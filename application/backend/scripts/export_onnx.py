"""
Export Model to ONNX
For cross-platform deployment
"""

import torch
import argparse
import os
from typing import Optional
from app.models.load_model import load_sequence_model
from app.core.logging import logger


def export_to_onnx(
    model,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 13
):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        example_input: Example input for tracing
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create example input if not provided
    if example_input is None:
        # (B, T, feature_dim)
        example_input = torch.randn(1, 32, 512)
    
    # Export to ONNX
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (example_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    logger.info(f"Model exported to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully")
    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset version"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_sequence_model(device="cpu")
    
    # Export
    export_to_onnx(
        model=model,
        output_path=args.output_path,
        opset_version=args.opset_version
    )


if __name__ == "__main__":
    main()
