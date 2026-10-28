# Experimental ComfyUI + AI Toolkit image

Image: `ghcr.io/alon-z/aigpu-docker:toolkit-cu124` (after CI succeeds).
This is an optional image; the existing `latest*` and `serverless*` Dockerfiles
and build settings are unchanged.

## Shared GPU stack, isolated app dependencies

- CUDA 12.4.1 / Ubuntu 22.04 base, intended for RTX 3090/4090 rental hosts.
- One system installation of PyTorch `2.13.0+cu126`, torchvision
  `0.28.0+cu126`, torchaudio `2.11.0+cu126`, and Triton `3.7.1`.
- Both app venvs use Python's `--system-site-packages`; neither needs another
  PyTorch/CUDA installation. Constraints prevent ordinary pip installs from
  replacing the shared stack.
- ComfyUI: `v0.33.4` tag commit `7a131a3afadc8200120f67f9236311a2c48b7445`.
  Its tokenizer dependencies stay separate from Toolkit's Transformers 5 stack.
- AI Toolkit: `b69b98314557f7c8b08e11bbb14e48d959e6ab26`, with
  TorchCodec explicitly installed as `0.15.0+cu126` instead of the PyPI CUDA 13 build.
- SageAttention: `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, built against
  the shared PyTorch with CUDA 12.4 for SM 8.6/8.9.

This is **not an officially supported modern-ComfyUI CUDA configuration**.
NVIDIA CUDA 12.x minor-version compatibility allowed the Toolkit primitive
GPU tests to pass on driver `550.144.03`; newer-driver features and PTX paths
may still fail on other workloads or hosts. No full ComfyUI workflow or
LoRA-training success is claimed from those primitive tests.

ComfyUI defaults to PyTorch attention with dynamic VRAM disabled. Its CUDA-13
Comfy Kitchen kernels are unavailable and its eager fallback must be used.
Explicit CLI switches can select another attention backend or
`--enable-dynamic-vram` when testing on a capable host. Custom nodes follow
`custom_nodes.txt` heads, as in the existing image; their dependency installs
are best-effort and workflow-specific compatibility still needs checking.

## Build and checks

Runs on the **host** from the repository root:

```bash
docker build -f Dockerfile.toolkit --target toolkit -t aigpu-toolkit:test .
docker run --rm --entrypoint /root/ComfyUI/venv/bin/python \
  aigpu-toolkit:test /opt/toolkit/smoke_test.py
```

CI builds/loads this variant, runs the CPU import/dependency/shared-stack
checks and a UI/jobs-API check, then publishes it. Failed checks prevent this
variant's image tags from being pushed. CI logs include uncompressed image
size; no size estimate should be treated as a measured result.

Run on a **GPU host** after pulling the image:

```bash
docker run --rm --gpus all --entrypoint /root/ComfyUI/venv/bin/python \
  ghcr.io/alon-z/aigpu-docker:toolkit-cu124 /opt/toolkit/smoke_test.py --gpu
```

This adds BF16 attention/backward, Triton execution, SageAttention GPU-kernel,
and AdamW8bit optimizer checks. A real ComfyUI workflow and a short training
job are still required before relying on the image for either workload.

## Startup and access

Runs **inside the container**:

```bash
cd /root/ai-toolkit/ui
npm run start
```

Node.js/FFmpeg and the production UI are already built. SQLite's Node addon is
compiled for Ubuntu 22.04. The empty UI database is initialized during build.
The existing Clore startup command `npm run build_and_start` is remapped in
this image to launch the prebuilt UI without installing or rebuilding.

The UI file server binds **127.0.0.1:8675**, not every interface. Access it
through an SSH tunnel; use the current rental's SSH host and port:

```bash
ssh -N -L 8675:127.0.0.1:8675 -p SSH_PORT root@SSH_HOST
```

Then open `http://localhost:8675`. Do not use the manager's automatic
install/update flow: its hardware policy rejects driver-550 hosts and updates
can replace this tested candidate stack. Rebuild the image to change versions.
Models, datasets, and training output are not baked into the image and need
separate persistence/backups before cancelling a rental.
