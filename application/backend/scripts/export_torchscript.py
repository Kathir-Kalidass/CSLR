"""
Export Model to TorchScript
For faster inference
"""

import torch
import argparse
import os
from typing import Any, Optional, cast
from app.models.load_model import load_sequence_model
from app.core.logging import logger


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    example_input: Optional[torch.Tensor] = None
) -> Any:
    """
    Export model to TorchScript
    
    Args:
        model: PyTorch model
        output_path: Path to save TorchScript model
        example_input: Example input for tracing
    """
    model.eval()
    
    # Create example input if not provided
    if example_input is None:
        # (B, T, feature_dim)
        example_input = torch.randn(1, 32, 512)
    
    # Trace model
    logger.info("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)  # type: ignore
    
    # Save
    logger.info(f"Saving TorchScript model to {output_path}")
    torch.jit.save(traced_model, output_path)
    
    logger.info("Export complete!")
    
    # Verify
    logger.info("Verifying exported model...")
    loaded_model = cast(Any, torch.jit.load(output_path))
    output1 = model(example_input)
    output2 = loaded_model(example_input)
    
    diff = torch.abs(output1 - output2).max().item()
    logger.info(f"Max difference: {diff}")
    
    return traced_model


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
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
        help="Path to save TorchScript model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_sequence_model(device=args.device)
    
    # Export
    export_to_torchscript(
        model=model,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
