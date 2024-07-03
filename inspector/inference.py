import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Predictor:
    def __init__(self, model_path: str, device: str = 'cuda', draw: bool = False, verbose: bool = False):
        self.draw = draw
        self.verbose = verbose
        os.environ['YOLO_VERBOSE'] = 'False'
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path, task='detect').to(self.device)
        pass

    def predict(self, image: np.ndarray, thresh: float = .4) -> tuple[np.ndarray, list]:
        """
        Perform inference using YOLO
        """
        assert image.shape == (640, 640, 3), "Input image must be 640x640x3"

        img = torch.tensor(image, dtype=torch.float16).to(self.device).unsqueeze(0).permute(0, 3, 1, 2) / 255
        outputs = self.model(img, verbose=self.verbose)
        output = outputs[0].cpu()
        output_image, data = self.process_outputs(output, image, thresh)

        return output_image, data

    def process_outputs(self, outputs: Results, image: np.ndarray, thresh: float = .4) -> tuple[np.ndarray, list]:
        """
        Process model outputs for mark-up.

        https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        """
        boxes = outputs.boxes
        data = list()

        for i, box in enumerate(boxes):
            # Get coordinates and class label
            x1, y1, x2, y2 = box.xyxy[0]
            label = int(box.cls.numpy())
            score = float(box.conf.numpy())
            if score > thresh:
                label_with_confidence = f'{label} {score:.2f}'
                data.append((x1, y1, x2, y2, label_with_confidence))
                if self.draw:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                    cv2.putText(image, label_with_confidence, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image, data
