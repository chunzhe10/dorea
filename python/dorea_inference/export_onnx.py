"""Export RAUNE-Net to ONNX with onnxsim simplification.

Usage:
    python -m dorea_inference.export_onnx \
        --weights models/raune_net/weights_95.pth \
        --models-dir models/raune_net \
        --output raune_net.onnx
"""

import sys
import argparse
from pathlib import Path

import torch


def export_raune_onnx(
    weights: str,
    models_dir: str,
    output: str,
    opset: int = 17,
) -> str:
    """Export RAUNE-Net to simplified ONNX.

    Args:
        weights: Path to .pth weights file.
        models_dir: Directory containing models/ package with raune_net.py.
        output: Output .onnx file path.
        opset: ONNX opset version (default 17, TRT 10.x supports 9-20).

    Returns:
        Path to the written ONNX file.
    """
    import onnx
    from onnxsim import simplify

    models_dir = Path(models_dir)
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from models.raune_net import RauneNet

    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 540, 960)

    raw_path = output + ".raw" if not output.endswith(".raw") else output
    torch.onnx.export(
        model,
        dummy,
        raw_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    raw_model = onnx.load(raw_path)
    simplified, ok = simplify(raw_model)
    if not ok:
        raise RuntimeError("onnxsim simplification failed — check model for unsupported ops")

    onnx.save(simplified, output)

    raw = Path(raw_path)
    if raw.exists() and raw_path != output:
        raw.unlink()

    onnx.checker.check_model(onnx.load(output))

    return output


def main():
    parser = argparse.ArgumentParser(description="Export RAUNE-Net to ONNX")
    parser.add_argument("--weights", required=True, help="Path to weights .pth file")
    parser.add_argument("--models-dir", required=True, help="Directory containing models/ package")
    parser.add_argument("--output", required=True, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    path = export_raune_onnx(args.weights, args.models_dir, args.output, args.opset)
    print(f"Exported to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
