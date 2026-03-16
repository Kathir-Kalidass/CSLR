"""
Generate TensorRT Build Command
===============================
Produces a recommended trtexec command for building a TensorRT engine
from ONNX with optional INT8 profile settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TensorRT trtexec command")
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine")
    parser.add_argument("--min-shape", default="input:1x16x512", help="Min dynamic shape")
    parser.add_argument("--opt-shape", default="input:1x32x512", help="Opt dynamic shape")
    parser.add_argument("--max-shape", default="input:4x96x512", help="Max dynamic shape")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16")
    parser.add_argument("--int8", action="store_true", help="Enable INT8")
    parser.add_argument("--calib", default="", help="Optional calibration cache path")
    args = parser.parse_args()

    onnx = Path(args.onnx).resolve()
    engine = Path(args.engine).resolve()

    cmd = [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--minShapes={args.min_shape}",
        f"--optShapes={args.opt_shape}",
        f"--maxShapes={args.max_shape}",
        "--workspace=4096",
    ]

    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")
        if args.calib:
            cmd.append(f"--calib={Path(args.calib).resolve()}")

    print(" ".join(cmd))


if __name__ == "__main__":
    main()
