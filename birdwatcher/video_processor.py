import threading
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from copy import deepcopy
import pandas as pd
from model_runner import DetectionModel, ClassificationModel
from utils import get_asset_path    
from constants import ALLOWED_BATCH_SIZES


class VideoProcessor:

    def __init__(self):
        self.class_mapping = pd.read_csv(get_asset_path("assets/classes/class_mapping.csv"))

    def process(
            self,
            video_path: Path,
            output_dir: Path,
            sample_interval: float,
            device: str,
            batch_size: int,
            box_conf_threshold: float,
            class_conf_threshold: float,
            progress_callback: Callable,
            cancel_event: threading.Event,
            multi_colour: bool = False,
    ):

        ### 1. set up ###

        # load video, check okay, and get frame rate
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video could not be opened")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_rate = max(1, int(fps * sample_interval))
        total_samples = np.ceil(total_frames / sampling_rate)

        # modify batch size if it is less than total samples
        if total_samples < batch_size:
            batch_size = [b for b in ALLOWED_BATCH_SIZES if b <= total_samples][-1]

        # create output directory
        output_dir.mkdir(exist_ok=True, parents=False)

        ### 2. detection stage ###

        # initialise detection model
        model = DetectionModel(device=device, batch_size=batch_size, threshold=box_conf_threshold)

        # read frames at frame rate and run detection in batches
        batch = []
        detections = []
        total_boxes = 0
        for i, frame_idx in enumerate(range(0, total_frames, sampling_rate)):
            if cancel_event.is_set():
                return False
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            batch.append(frame)
            if len(batch) == batch_size or i + 1 == total_samples:
                pred, box_count = model(batch)
                detections.extend(pred)
                batch = []
                total_boxes += box_count
                progress_callback(i / total_samples, "Finding birds")
        
        # clear detection model
        del model
        
        ### 3. classification stage ###

        # initialise classification model
        model = ClassificationModel(device=device, batch_size=batch_size, threshold=class_conf_threshold)

        # classify each box in each frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video could not be opened")
        batch, batch_helper = [], []
        classifications = deepcopy(detections)
        probabilities = deepcopy(detections)
        total_boxes_batched = 0
        total_boxes_processed = 0
        for i, frame_boxes in enumerate(detections):
            if cancel_event.is_set():
                return False
            if len(frame_boxes) > 0:
                frame_idx = i * sampling_rate
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _, frame = cap.read()
                for j, box in enumerate(frame_boxes):
                    batch_helper.append((i, j))
                    resized_frame = self.crop_to_square_box_and_resize(frame, box)
                    batch.append(resized_frame)
                    total_boxes_batched += 1
                    total_boxes_processed += 1
                    if len(batch) == batch_size or total_boxes_batched == total_boxes:
                        pred, probs = model(batch)
                        for name, prob, indices in zip(pred, probs, batch_helper):
                            classifications[indices[0]][indices[1]] = name
                            probabilities[indices[0]][indices[1]] = prob
                        batch, batch_helper = [], []
                        progress_callback(total_boxes_processed / total_boxes, "Classifying birds")

        # clear classification model
        del model
           
        ### 4. generate outputs ###
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video could not be opened")
        output_frames = sum(bool(frame_boxes) for frame_boxes in detections)
        output_frames_processed = 0
        if output_frames == 0:
            progress_callback(1.0, "Saving output")
        for i, frame_boxes in enumerate(detections):
            if cancel_event.is_set():
                return False
            if len(frame_boxes) > 0:
                frame_idx = i * sampling_rate
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _, frame = cap.read()
                marked_frame = self.draw_frame(
                    frame,
                    frame_boxes,
                    classifications[i],
                    probabilities[i],
                    multi_colour,
                )
                seconds = int(frame_idx / fps)
                hours, seconds_remaining = divmod(seconds, 3600)    
                minutes, seconds = divmod(seconds_remaining, 60)
                timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
                for name in classifications[i]:
                    name_dir = (output_dir / name)
                    name_dir.mkdir(exist_ok=True)
                    file_path = name_dir / f"{timestamp}.png"
                    cv2.imwrite(file_path, marked_frame)
                output_frames_processed += 1
                progress_callback(
                    output_frames_processed / output_frames,
                    "Saving output",
                )
                    
    def draw_frame(
            self,
            frame: np.ndarray,
            boxes: list,
            names: list,
            probs: list,
            multi_colour: bool = False,
    ) -> np.ndarray:
        for box, name, prob in zip(boxes, names, probs):
            if name == "Unknown":
                R, G, B = 255, 255, 255
            elif multi_colour:
                R = int(self.class_mapping.loc[self.class_mapping["label"] == name, "r"].values[0])
                G = int(self.class_mapping.loc[self.class_mapping["label"] == name, "g"].values[0])
                B = int(self.class_mapping.loc[self.class_mapping["label"] == name, "b"].values[0])
            else:
                R, G, B = 255, 0, 0
            x1, y1, x2, y2 = [int(x) for x in box]
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(frame.shape[1], x2 + 5)
            y2 = min(frame.shape[0], y2 + 5) 
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color=(B,G,R), thickness=4
                )
            tag = name.replace("_", " ") + (f" [{np.round(100*prob, 0):.0f}%]" if name != "Unknown" else "")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            text_thickness = 4
            (text_width, text_height), baseline = cv2.getTextSize(
                tag, font, font_scale, text_thickness
            )

            text_x = min(x1, max(0, frame.shape[1] - text_width))
            text_y = y1 - 15 - baseline
            if text_y - text_height - baseline < 0:
                text_y = y2 + 15 + text_height + baseline
            text_y = min(max(text_y, text_height + baseline), frame.shape[0])
            cv2.putText(
                frame, 
                tag, 
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=(B,G,R), 
                thickness=text_thickness
            )
        return frame

    @staticmethod
    def crop_to_square_box_and_resize(arr: np.ndarray, box: list) -> np.ndarray:
        
        x1 = int(np.floor(box[0]))
        y1 = int(np.floor(box[1]))
        x2 = int(np.ceil(box[2]))
        y2 = int(np.ceil(box[3]))

        size = max(x2 - x1 + 1, y2 - y1 + 1)
        x_mid = (x1 + x2) // 2
        y_mid = (y1 + y2) // 2

        x_start = x_mid - size // 2
        y_start = y_mid - size // 2
        x_end = x_start + size
        y_end = y_start + size

        pad_left = max(0, -x_start)
        pad_top = max(0, -y_start)
        pad_right = max(0, x_end - arr.shape[1])
        pad_bottom = max(0, y_end - arr.shape[0])

        padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        cropped = padded[y_start + pad_top:y_end + pad_top, x_start + pad_left:x_end + pad_left]
        resized = cv2.resize(cropped, (224,224), interpolation=cv2.INTER_CUBIC)

        return resized