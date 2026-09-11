from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import pandas as pd
from scipy.special import softmax
from pathlib import Path
from utils import get_asset_path


gpu_providers = [
    "CUDAExecutionProvider",    
    "CoreMLExecutionProvider",  
    "CPUExecutionProvider"       
]

cpu_providers = [ 
    "CPUExecutionProvider"       
]


class DetectionModel:

    def __init__(self, use_gpu: bool, batch_size: int, threshold: float):

        if use_gpu:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                self.device = "cuda"
            elif "CoreMLExecutionProvider" in providers:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        self.batch_size = batch_size
        self.threshold = threshold
        self.model = YOLO(get_asset_path(f"assets/models/yolo26x_b{batch_size}.onnx"), task="detect")

    def __call__(self, x: list) -> tuple[list[list], int]:
        short = self.batch_size - len(x)
        x.extend(short * x[0:1])
        results = self.model(x, device=self.device)[:self.batch_size - short] 
        output = []
        box_count = 0
        for batch in results:
            boxes = []
            for data in batch.boxes.data:
                data = data.cpu()
                if data[-1] == 14.0 and data[-2] >= self.threshold:
                    boxes.append(data[:4].tolist())
            box_count += len(boxes)
            output.append(boxes)
        return output, box_count


class ClassificationModel:

    def __init__(self, use_gpu: bool, batch_size: int, threshold: float):

        providers = gpu_providers if use_gpu else cpu_providers
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = ort.InferenceSession(get_asset_path(f"assets/models/classifier_b{batch_size}.onnx"), providers=providers)
        self.class_mapping = pd.read_csv(get_asset_path("assets/classes/class_mapping.csv"))
    
    def __call__(self, x: list) -> list:
        short = self.batch_size - len(x)
        x.extend(short * x[0:1])       
        batch = np.stack(x, axis=0)
        batch = batch[..., ::-1] # BGR to RGB
        batch = self.scale_norm_permute(batch)
        input_name = self.model.get_inputs()[0].name
        pred = self.model.run(None, {input_name: batch})[0][:self.batch_size-short]
        pred = softmax(pred, axis=1)
        pred_indices = np.argmax(pred, axis=1)
        pred_confs = np.max(pred, axis=1)
        pred_names = []
        pred_probs = []
        for idx, conf in zip(pred_indices, pred_confs):
            if conf >= self.threshold:
                name = self.class_mapping.loc[self.class_mapping["class id"] == idx, "label"].values[0]
            else:
                name = "Unknown"
            pred_names.append(name)
            pred_probs.append(conf)
        return pred_names, pred_probs

    @staticmethod
    def scale_norm_permute(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32) / 255.0
        target_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        target_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - target_means) / target_stds
        return arr.transpose(0, 3, 1, 2)