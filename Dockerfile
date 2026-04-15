FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------------
# System dependencies
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git ca-certificates gnupg \
        python3 python3-venv python3-dev python3-pip \
        build-essential libgl1 libglib2.0-0 aria2 rclone tmux \
        fonts-dejavu-core fonts-liberation vim \
    && curl -fsSL https://deb.nodesource.com/setup_23.x | bash - \
    && apt-get install -y nodejs \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# ComfyUI + custom nodes
# -------------------------------------------------------
WORKDIR /root
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /root/ComfyUI
RUN python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip wheel \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    && pip install tiktoken sentencepiece triton

# SageAttention (compile from source for Blackwell sm_120)
# Bypass CUDA version check (system has 13.2, PyTorch built with 12.8)
RUN . /root/ComfyUI/venv/bin/activate \
    && TORCH_EXT=$(python3 -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.__file__)" 2>/dev/null) \
    && cp "$TORCH_EXT" "${TORCH_EXT}.bak" \
    && sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass #raise RuntimeError(CUDA_MISMATCH_MESSAGE/' "$TORCH_EXT" \
    && git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention \
    && TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation . \
    && cp "${TORCH_EXT}.bak" "$TORCH_EXT" \
    && cd /root && rm -rf /tmp/SageAttention

WORKDIR /root/ComfyUI/custom_nodes
COPY custom_nodes.txt /tmp/custom_nodes.txt
RUN set -eu; \
    grep -Ev '^[[:space:]]*(#|$)' /tmp/custom_nodes.txt \
      | while read -r url dir; do \
            [ -n "$dir" ] || dir=$(basename "$url" .git); \
            printf '%s\0%s\0' "$url" "$dir"; \
        done \
      | xargs -0 -n 2 -P 8 git clone --depth=1; \
    rm /tmp/custom_nodes.txt

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

# -------------------------------------------------------
# ai-toolkit (ostris)
# -------------------------------------------------------
WORKDIR /root
RUN git clone https://github.com/ostris/ai-toolkit.git

WORKDIR /root/ai-toolkit
RUN git submodule update --init --recursive \
    && python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip wheel \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    && pip install --upgrade accelerate transformers diffusers huggingface_hub

WORKDIR /root/ai-toolkit/ui
RUN npm install

# -------------------------------------------------------
# Create models directory structure
# -------------------------------------------------------
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
