import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Predictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Load the model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path, task='detect').to(self.device)
        # self.model.half()

    def predict(self, image: np.ndarray) -> np.array:
        """
        Perform inference using YOLO
        """
        assert image.shape == (640, 640, 3), "Input image must be 640x640x3"

        # Normalize the image
        img = torch.tensor(image, dtype=torch.float16).to(self.device).unsqueeze(0).permute(0, 3, 1, 2) / 255
        # Perform inference.
        outputs = self.model(img)
        output = outputs[0].cpu()
        # Run markup
        # TODO: We need to transform the markup and place it on the original image from before the ROI zoom.
        output_image = self.process_outputs(output, image)

        return output_image

    @staticmethod
    def process_outputs(outputs: Results, image: np.ndarray) -> np.ndarray:
        """
        Process model outputs for mark-up.

        https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        """
        boxes = outputs.boxes

        for i, box in enumerate(boxes):
            # Get coordinates and class label
            x1, y1, x2, y2 = box.xyxy[0]
            label = int(box.cls.numpy())
            score = float(box.conf.numpy())
            label_with_confidence = f'{label} {score:.2f}'

            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put the label and confidence score
            cv2.putText(image, label_with_confidence, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        pass
        return image
