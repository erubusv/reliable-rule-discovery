#!/usr/bin/env python3
"""Fail-fast CUDA runtime check for this workspace."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys


def print_status(label: str, value: object, quiet: bool) -> None:
    if not quiet:
        print(f"{label}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that CUDA is usable from Python.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ok = True

    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print_status("nvidia-smi exit_code", smi.returncode, args.quiet)
        if not args.quiet and smi.stdout:
            print(smi.stdout.strip())
        ok = ok and smi.returncode == 0
    except FileNotFoundError:
        print_status("nvidia-smi", "not found", args.quiet)
        ok = False

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        cu_ret = cuda.cuInit(0)
        print_status("cuInit", cu_ret, args.quiet)
        ok = ok and cu_ret == 0
    except Exception as exc:
        print_status("cuInit", repr(exc), args.quiet)
        ok = False

    try:
        import torch

        print_status("torch", torch.__version__, args.quiet)
        print_status("torch.version.cuda", torch.version.cuda, args.quiet)
        print_status("torch.cuda.is_available", torch.cuda.is_available(), args.quiet)
        print_status("torch.cuda.device_count", torch.cuda.device_count(), args.quiet)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            x = torch.arange(1024, device=device, dtype=torch.float32)
            y = float((x * x).sum().detach().cpu())
            print_status("cuda:0", torch.cuda.get_device_name(0), args.quiet)
            print_status("tensor_test_sum", round(y, 3), args.quiet)
        ok = ok and torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception as exc:
        print_status("torch_cuda", repr(exc), args.quiet)
        ok = False

    if not ok and args.quiet:
        print("CUDA check failed", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
