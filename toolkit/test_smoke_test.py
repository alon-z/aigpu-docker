#!/usr/bin/env python3
"""Runs on the host: python3 toolkit/test_smoke_test.py. No GPU/packages needed."""

import ast
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

import smoke_test

for gpu in (False, True):
    output = io.StringIO()
    argv = ['smoke_test.py', '--gpu'] if gpu else ['smoke_test.py']
    with patch.object(sys, 'argv', argv), patch.object(smoke_test.subprocess, 'run') as run, redirect_stdout(output):
        smoke_test.main()
    calls = run.call_args_list
    assert len(calls) == 4
    code = {}
    for call in calls:
        command = call.args[0]
        assert call.kwargs['check'] is True
        if command[1] == '-c':
            ast.parse(command[2])
            code[call.kwargs['cwd'].name] = command[2]
        else:
            assert command[1:] == ['-m', 'pip', 'check']
    toolkit = code['ai-toolkit']
    assert 'import torchcodec' in toolkit
    assert ('from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model' in toolkit) == gpu
    assert ('import bitsandbytes' in toolkit) == gpu
    assert ('SKIP GPU-dependent Toolkit' in output.getvalue()) == (not gpu)
    for script in code.values():
        assert ('assert torch.cuda.is_available()' in script) == gpu

print('PASS CPU/GPU check separation; dependency checks retained in both modes')
