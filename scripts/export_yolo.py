from ultralytics import YOLO
import argparse
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format.")
    parser.add_argument("input", type=Path, help="Path to the input YOLO model file.")
    parser.add_argument("output", type=Path, help="Path to the output ONNX model file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export (default: 1).")
    return parser.parse_args()      


def export(path_in: Path, path_out: Path, batch_size: int):

    model = YOLO(path_in, task="detect")

    exported = Path(model.export(
        format="onnx",
        batch=batch_size,
        dynamic=False,
        simplify=True,
    ))

    shutil.move(exported, path_out)


if __name__ == "__main__":
    args = parse_args()
    export(args.input, args.output, args.batch_size)