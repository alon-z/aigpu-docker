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
    && pip install tiktoken sentencepiece

WORKDIR /root/ComfyUI/custom_nodes
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
    && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git \
    && git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
    && git clone https://github.com/rgthree/rgthree-comfy.git \
    && git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git \
    && git clone https://github.com/kijai/ComfyUI-Florence2.git \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
    && git clone https://github.com/city96/ComfyUI-GGUF.git \
    && git clone https://github.com/cubiq/ComfyUI_essentials.git

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
