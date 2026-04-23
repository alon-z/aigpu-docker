# syntax=docker/dockerfile:1.6

# Build args control the CUDA/PyTorch toolchain version.
# Defaults target Blackwell (sm_120) on CUDA 13.2 / cu128.
# Override for older hosts: e.g. CUDA_VERSION=12.4.1 UBUNTU_VERSION=22.04
# TORCH_INDEX=cu124 TORCH_ARCH_LIST=8.9 for an Ada (4090) build on a
# CUDA 12.4 driver.
ARG CUDA_VERSION=13.2.0
ARG UBUNTU_VERSION=24.04
ARG TORCH_INDEX=cu128
ARG TORCH_ARCH_LIST="8.9 12.0"

# =======================================================
# Stage 1: base-builder — apt deps shared by builders
# =======================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git ca-certificates \
        python3 python3-venv python3-dev python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# =======================================================
# Stage 2: comfy-builder — ComfyUI + SageAttention + nodes
# Cross-compiles SageAttention for Blackwell sm_120.
# =======================================================
FROM base-builder AS comfy-builder
ARG TORCH_INDEX
ARG TORCH_ARCH_LIST

WORKDIR /root
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /root/ComfyUI
RUN python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip wheel \
    && pip install torch torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
    && pip install -r requirements.txt \
    && pip install tiktoken sentencepiece triton \
    && pip install runpod requests websocket-client

# SageAttention (compile from source for the selected arch).
# Bypass torch's CUDA-version check — it's safe when the base image CUDA
# and the torch wheel CUDA disagree (e.g. 13.2 base + cu128 wheel), and a
# harmless no-op when they match.
RUN . /root/ComfyUI/venv/bin/activate \
    && TORCH_EXT=$(python3 -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.__file__)") \
    && cp "$TORCH_EXT" "${TORCH_EXT}.bak" \
    && sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass #raise RuntimeError(CUDA_MISMATCH_MESSAGE/' "$TORCH_EXT" \
    && git clone --depth=1 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention \
    && TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" pip install --no-build-isolation . \
    && cp "${TORCH_EXT}.bak" "$TORCH_EXT" \
    && rm "${TORCH_EXT}.bak" \
    && cd /root && rm -rf /tmp/SageAttention

# Custom nodes
WORKDIR /root/ComfyUI/custom_nodes
COPY custom_nodes.txt /tmp/custom_nodes.txt
RUN set -u; \
    fail_log=/tmp/clone-fail.log; \
    : > "$fail_log"; \
    grep -Ev '^[[:space:]]*(#|$)' /tmp/custom_nodes.txt \
      | while read -r url dir; do \
            [ -n "$dir" ] || dir=$(basename "$url" .git); \
            printf '%s\0%s\0' "$url" "$dir"; \
        done \
      | xargs -0 -n 2 -P 8 sh -c \
            'git clone --depth=1 "$1" "$2" || echo "$1" >> /tmp/clone-fail.log' _; \
    if [ -s "$fail_log" ]; then \
        echo "=================================================="; \
        echo "WARNING: these custom nodes failed to clone:"; \
        cat "$fail_log"; \
        echo "=================================================="; \
    fi; \
    rm -f "$fail_log" /tmp/custom_nodes.txt

# Patch ComfyUI-NAG for newer ComfyUI (chroma classes moved to flux)
RUN sed -i 's/from comfy.ldm.chroma.layers import DoubleStreamBlock, SingleStreamBlock/from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock/' \
        ComfyUI-NAG/chroma/layers.py \
    && sed -i 's/from comfy.ldm.chroma.layers import/from comfy.ldm.flux.layers import/' \
        ComfyUI-NAG/chroma/model.py

RUN . /root/ComfyUI/venv/bin/activate \
    && for d in */; do \
        [ -f "$d/requirements.txt" ] && pip install -r "$d/requirements.txt" 2>/dev/null || true; \
        [ -f "$d/install.py" ] && python3 "$d/install.py" 2>/dev/null || true; \
    done

RUN find /root/ComfyUI -type d -name '.git'        -prune -exec rm -rf {} + \
    && find /root/ComfyUI -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /root/ComfyUI -type f -name '*.pyc'             -delete \
    && rm -rf /root/.cache /tmp/*

# =======================================================
# Stage 3a: serverless — slim runtime for RunPod serverless
#   build with: docker build --target serverless -t <img>:serverless .
# =======================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS serverless

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Force pure-Python protobuf — native C++ impl segfaults on proto-plus enum init
# (triggered by ComfyUI-utils-nodes' google.generativeai import).
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
        libgl1 libglib2.0-0 libgles2 libegl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=comfy-builder /root/ComfyUI /root/ComfyUI

RUN mkdir -p \
    /root/ComfyUI/models/checkpoints/flux \
    /root/ComfyUI/models/vae/qwen \
    /root/ComfyUI/models/vae/zit \
    /root/ComfyUI/models/clip/qwen \
    /root/ComfyUI/models/text_encoders/zit \
    /root/ComfyUI/models/diffusion_models/gguf \
    /root/ComfyUI/models/diffusion_models \
    /root/ComfyUI/models/loras/qwen \
    /root/ComfyUI/models/clip_vision \
    /root/ComfyUI/models/model_patches \
    /root/ComfyUI/models/upscale_models \
    /root/ComfyUI/models/ultralytics/bbox

COPY serverless/handler.py     /handler.py
COPY serverless/start.sh       /start.sh
COPY serverless/test_input.json /test_input.json
RUN chmod +x /start.sh

WORKDIR /
CMD ["/start.sh"]

# =======================================================
# Stage 3b: pod — full-featured runtime (default build target)
# =======================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS pod

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Force pure-Python protobuf — native C++ impl segfaults on proto-plus enum init
# (triggered by ComfyUI-utils-nodes' google.generativeai import).
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git ca-certificates gnupg \
        python3 python3-venv python3-pip \
        libgl1 libglib2.0-0 libgles2 libegl1 aria2 rclone tmux \
        fonts-dejavu-core fonts-liberation vim \
        # cloud-provider injected at startup
        openssh-server openssh-client htop nano xauth \
        lsb-release systemd systemd-sysv \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY --from=comfy-builder /root/ComfyUI /root/ComfyUI

RUN mkdir -p \
    /root/ComfyUI/models/checkpoints/flux \
    /root/ComfyUI/models/vae/qwen \
    /root/ComfyUI/models/vae/zit \
    /root/ComfyUI/models/clip/qwen \
    /root/ComfyUI/models/text_encoders/zit \
    /root/ComfyUI/models/diffusion_models/gguf \
    /root/ComfyUI/models/diffusion_models \
    /root/ComfyUI/models/loras/qwen \
    /root/ComfyUI/models/clip_vision \
    /root/ComfyUI/models/model_patches \
    /root/ComfyUI/models/upscale_models \
    /root/ComfyUI/models/ultralytics/bbox

WORKDIR /root
