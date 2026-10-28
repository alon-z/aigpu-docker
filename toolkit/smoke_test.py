#!/usr/bin/env python3
"""Runs inside the Toolkit image. CPU checks by default; --gpu on a GPU host."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path('/root')
SHARED_CHECK = """
from pathlib import Path
from importlib.metadata import version
import torch, torchvision, torchaudio
assert torch.__version__ == '2.13.0+cu126', torch.__version__
assert version('torchvision') == '0.28.0+cu126'
assert version('torchaudio') == '2.11.0+cu126'
assert version('triton') == '3.7.1'
assert Path(torch.__file__).parent == Path('/usr/local/lib/python3.10/dist-packages/torch'), torch.__file__
print('PASS shared PyTorch/CUDA:', torch.__version__, torch.__file__)
"""
GPU_CHECK = """
import torch.nn.functional as F
assert torch.cuda.is_available()
x = torch.randn(128, 128, device='cuda')
y = torch.compile(lambda t: t.sin() + t.cos())(x)
torch.testing.assert_close(y, x.sin() + x.cos())
q = torch.randn(1, 2, 128, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
F.scaled_dot_product_attention(q, q, q).float().square().mean().backward()
assert torch.isfinite(q.grad).all()
torch.cuda.synchronize()
print('PASS GPU BF16 attention/backward and Triton execution')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    checks = {
        'ComfyUI': """
from importlib.metadata import distributions
installed = {dist.metadata['Name'].lower() for dist in distributions()}
assert 'runpod' not in installed, 'RunPod SDK is not part of this image'
assert 'albumentationsx' not in installed
assert version('albumentations') == '2.0.8'
assert version('albucore') == '0.0.24'
assert version('onnx') == '1.22.0'
assert version('protobuf') == '5.29.6'
from comfy.cli_args import args
assert args.disable_dynamic_vram and args.use_pytorch_cross_attention
args.cpu = True
import comfy.sd
import sageattention
print('PASS ComfyUI core and SageAttention imports')
""",
        'ai-toolkit': """
import torchcodec
assert version('torchcodec') == '0.15.0+cu126'
print('PASS TorchCodec import')
""",
    }
    for app, check in checks.items():
        directory = ROOT / app
        python = directory / 'venv/bin/python'
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        code = SHARED_CHECK + check
        if args.gpu:
            code += GPU_CHECK
            if app == 'ComfyUI':
                code += """
from sageattention import sageattn
out = sageattn(q.detach(), q.detach(), q.detach(), is_causal=False)
assert out.shape == q.shape and torch.isfinite(out).all()
print('PASS SageAttention GPU kernel')
"""
            else:
                # The upstream registry eagerly imports OmniGen2, which queries
                # CUDA device properties at import time even when testing Krea2.
                code += """
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
print('PASS Toolkit model-registry and Krea2 imports')
import bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(128, 128, device='cuda'))
optimizer = bnb.optim.AdamW8bit([p])
p.square().mean().backward()
optimizer.step()
assert torch.isfinite(p).all()
print('PASS AdamW8bit GPU optimizer step')
"""
        subprocess.run([str(python), '-c', code], cwd=directory, check=True)
        if app == 'ai-toolkit' and not args.gpu:
            print('SKIP GPU-dependent Toolkit model-registry/Krea2 import; run with --gpu', flush=True)


if __name__ == '__main__':
    main()
