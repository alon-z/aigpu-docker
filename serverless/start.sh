#!/usr/bin/env bash
set -e

COMFY_DIR="/root/ComfyUI"
VENV="$COMFY_DIR/venv"

# Activate the ComfyUI venv so both comfy and the handler share torch/runpod.
# shellcheck disable=SC1090
. "$VENV/bin/activate"

echo "serverless: verifying GPU..."
if ! python3 -c "import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))"; then
    echo "serverless: CUDA init failed — worker cannot run." >&2
    exit 1
fi

echo "serverless: launching ComfyUI on 127.0.0.1:8188..."
python3 -u "$COMFY_DIR/main.py" \
    --disable-auto-launch \
    --disable-metadata \
    --listen 127.0.0.1 \
    --port 8188 &
COMFY_PID=$!

# If ComfyUI dies, take the whole container down so RunPod marks it unhealthy.
trap 'kill -TERM $COMFY_PID 2>/dev/null || true' EXIT

echo "serverless: starting RunPod handler (comfy pid=$COMFY_PID)..."
exec python3 -u /handler.py
