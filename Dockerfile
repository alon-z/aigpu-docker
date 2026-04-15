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
    && git clone https://github.com/cubiq/ComfyUI_essentials.git \
    # --- new nodes --- \
    && git clone https://github.com/chflame163/ComfyUI_LayerStyle.git \
    && git clone https://github.com/yolain/ComfyUI-Easy-Use.git \
    && git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
    && git clone https://github.com/crystian/ComfyUI-Crystools.git \
    && git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git \
    && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
    && git clone https://github.com/Gourieff/comfyui-reactor-node.git \
    && git clone https://github.com/chrisgoringe/cg-use-everywhere.git \
    && git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git \
    && git clone https://github.com/giriss/comfy-image-saver.git \
    && git clone https://github.com/changethecon/SeedVarianceEnhancer.git \
    && git clone https://github.com/angelbottomless/ComfyUI-LogicUtils.git \
    && git clone https://github.com/crt-nodes/CRT-Nodes.git \
    && git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git \
    && git clone https://github.com/VirtuosoResearch/ComfyUI-Virtuoso-Nodes.git \
    && git clone https://github.com/vsLinx/ComfyUI-vsLinx-Nodes.git \
    && git clone https://github.com/Writili/ComfyUI-WtlNodes.git \
    && git clone https://github.com/kijai/ComfyUI-KJNodes.git \
    && git clone https://github.com/filliptm/ComfyUI_Fill-Nodes.git \
    && git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git \
    && git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git \
    && git clone https://github.com/calcuis/gguf.git ComfyUI-gguf-calcuis \
    && git clone https://github.com/digitaljohn/ComfyUI-ProPost.git \
    && git clone https://github.com/alexopus/ComfyUI-Image-Saver.git \
    && git clone https://github.com/miosp/ComfyUI-FBCNN.git \
    && git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod.git \
    && git clone https://github.com/skatardude10/ComfyUI-Optical-Realism.git \
    && git clone https://github.com/Slartibart23/comfyui-sentence-filter.git \
    # --- video/audio/prompts --- \
    && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    && git clone https://github.com/melMass/comfy_mtb.git \
    && git clone https://github.com/kijai/ComfyUI-MMAudioWrapper.git \
    && git clone https://github.com/ChenDarYen/ComfyUI-NAG.git \
    && git clone https://github.com/Alectriciti/comfyui-adaptiveprompts.git \
    && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

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
