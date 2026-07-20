from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class BuildExt(_build_ext):
    def run(self) -> None:
        super().run()
        nvcc = shutil.which("nvcc")
        if nvcc is None or os.environ.get("CRBSTPP_DISABLE_CUDA") == "1":
            return
        output_dir = (
            Path("src") / "crbstpp"
            if self.inplace
            else Path(self.build_lib) / "crbstpp"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            nvcc,
            "-O3",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-gencode",
            "arch=compute_89,code=[sm_89,compute_89]",
            str(Path("native") / "pricing_cuda.cu"),
            "-lcublas",
            "-o",
            str(output_dir / "libcrbstpp_cuda.so"),
        ]
        subprocess.run(command, check=True)


setup(
    ext_modules=[
        Extension(
            "crbstpp._cpu_native",
            ["native/cpu_native.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
