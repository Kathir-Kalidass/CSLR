"""
Export ONNX + Dynamic INT8 Quantization
======================================
Creates ONNX model and quantized ONNX model for low-latency inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from app.core.logging import logger
from app.models.load_model import load_sequence_model


def export_onnx(model_path: str, onnx_path: str, opset: int = 17) -> None:
    model = load_sequence_model(device="cpu")
    model.eval()

    dummy = torch.randn(1, 32, 512)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
    )

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX and INT8-quantized ONNX")
    parser.add_argument("--model-path", required=True, help="Source model checkpoint path")
    parser.add_argument("--onnx-path", required=True, help="Output ONNX path")
    parser.add_argument("--int8-path", required=True, help="Output INT8 ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    onnx_path = str(Path(args.onnx_path).resolve())
    int8_path = str(Path(args.int8_path).resolve())

    logger.info("Exporting ONNX to %s", onnx_path)
    export_onnx(args.model_path, onnx_path, args.opset)

    logger.info("Quantizing ONNX to INT8: %s", int8_path)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )

    logger.info("ONNX INT8 export complete")


if __name__ == "__main__":
    main()
