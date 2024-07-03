import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from .utils import Sort


class Predictor:
    def __init__(self, model_path: str, device: str = 'cuda', draw: bool = False, verbose: bool = False):
        self.draw = draw
        self.verbose = verbose
        os.environ['YOLO_VERBOSE'] = 'False'
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path, task='detect').to(self.device)
        self.sort = Sort()

    def predict(self, image: np.ndarray, thresh: tuple[float, int]) -> tuple[np.ndarray, list]:
        assert image.shape == (640, 640, 3), "Input image must be 640x640x3"
        img = torch.tensor(image, dtype=torch.float16).to(self.device).unsqueeze(0).permute(0, 3, 1, 2) / 255
        outputs = self.model(img, verbose=self.verbose)
        output = outputs[0].cpu()
        output_image, data = self.process_outputs(output, image, thresh)
        return output_image, data

    def process_outputs(self, outputs: Results, image: np.ndarray, thresh: tuple[float, int]) -> tuple[np.ndarray, list]:
        boxes = outputs.boxes
        detections = []
        data = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            label = int(box.cls.numpy())
            score = float(box.conf.numpy())
            if score > thresh[0]:
                label_with_confidence = f'{label} {score:.2f}'
                detections.append((x1, y1, x2, y2))
                data.append((x1, y1, x2, y2, label_with_confidence))

        result = list()
        tracks = self.sort.update(detections)  # TODO: We aren't actually using this yet...
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            label_with_confidence = next((d[4] for d in data if (d[0], d[1], d[2], d[3]) == (x1, y1, x2, y2)), None)
            result.append((x1, y1, x2, y2, label_with_confidence))
            if label_with_confidence and self.draw:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(image, label_with_confidence, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image, result
