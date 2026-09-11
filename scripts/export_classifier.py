import argparse
from pathlib import Path
import timm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export classifier model to ONNX format.")
    parser.add_argument("model_name", type=str, help="Name of the timm model architecture.")
    parser.add_argument("ckpt_path", type=Path, help="Path to the input checkpoint file.")
    parser.add_argument("num_classes", type=int, help="Number of classes for the classifier model.")
    parser.add_argument("output", type=Path, help="Path to the output ONNX  model file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export (default: 1).")
    return parser.parse_args()       


def export(model_name: str, ckpt_path: Path, num_classes: int, path_out: Path, batch_size: int):

    model = timm.create_model(
            model_name, 
            num_classes=num_classes,
            cache_dir="../models"
            )

    ckpt = torch.load(ckpt_path)
    state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        path_out,
    )

if __name__ == "__main__":
    args = parse_args()
    export(args.model_name, args.ckpt_path, args.num_classes, args.output, args.batch_size)