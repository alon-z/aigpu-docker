#!/usr/bin/env bash
set -e

COMFY_DIR="/root/ComfyUI"
VENV="$COMFY_DIR/venv"
VOLUME="${RUNPOD_VOLUME:-/runpod-volume}"
EXTRA_PATHS="$COMFY_DIR/extra_model_paths.yaml"

# shellcheck disable=SC1090
. "$VENV/bin/activate"

# Detect models root on the network volume. Users organize the volume
# differently: models directly at the root, under models/, or under
# ComfyUI/models/. Checkpoints presence is the strongest signal; fall back
# to diffusion_models (Flux-style) if checkpoints dir is absent.
detect_models_base() {
    local override="${MODELS_BASE_PATH:-}"
    if [ -n "$override" ]; then
        if [ -d "$override" ]; then
            echo "$override"
            return
        fi
        echo "serverless: MODELS_BASE_PATH=$override does not exist, falling through to auto-detect" >&2
    fi

    [ -d "$VOLUME" ] || return

    for candidate in \
        "$VOLUME/ComfyUI/models" \
        "$VOLUME/models" \
        "$VOLUME"; do
        if [ -d "$candidate/checkpoints" ] || \
           [ -d "$candidate/diffusion_models" ] || \
           [ -d "$candidate/unet" ]; then
            echo "$candidate"
            return
        fi
    done
}

write_extra_model_paths() {
    local base="$1"
    cat > "$EXTRA_PATHS" <<YAML
runpod_volume:
    base_path: $base
    checkpoints: checkpoints/
    clip: clip/
    clip_vision: clip_vision/
    configs: configs/
    controlnet: controlnet/
    diffusers: diffusers/
    diffusion_models: diffusion_models/
    embeddings: embeddings/
    gligen: gligen/
    hypernetworks: hypernetworks/
    loras: loras/
    photomaker: photomaker/
    style_models: style_models/
    text_encoders: text_encoders/
    unet: unet/
    upscale_models: upscale_models/
    vae: vae/
    vae_approx: vae_approx/
    ultralytics: ultralytics/
    ultralytics_bbox: ultralytics/bbox/
    ultralytics_segm: ultralytics/segm/
    sams: sams/
    custom_nodes: custom_nodes/
YAML
}

MODELS_BASE="$(detect_models_base || true)"
if [ -n "$MODELS_BASE" ]; then
    echo "serverless: wiring ComfyUI models search path to $MODELS_BASE"
    write_extra_model_paths "$MODELS_BASE"

    # Point HF cache at the volume so custom nodes that hf-download at runtime
    # persist across workers instead of re-downloading into ephemeral scratch.
    export HF_HOME="${HF_HOME:-$VOLUME/huggingface}"
    mkdir -p "$HF_HOME" 2>/dev/null || true
else
    echo "serverless: no network volume with recognizable model layout; using image-baked models only"
fi

echo "serverless: verifying GPU..."
if ! python3 -c "import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))"; then
    echo "serverless: CUDA init failed - worker cannot run." >&2
    exit 1
fi

echo "serverless: launching ComfyUI on 127.0.0.1:8188..."
COMFY_ARGS=(
    --disable-auto-launch
    --disable-metadata
    --listen 127.0.0.1
    --port 8188
)
[ -f "$EXTRA_PATHS" ] && COMFY_ARGS+=(--extra-model-paths-config "$EXTRA_PATHS")

python3 -u "$COMFY_DIR/main.py" "${COMFY_ARGS[@]}" &
COMFY_PID=$!

trap 'kill -TERM $COMFY_PID 2>/dev/null || true' EXIT

echo "serverless: starting RunPod handler (comfy pid=$COMFY_PID)..."
exec python3 -u /handler.py
