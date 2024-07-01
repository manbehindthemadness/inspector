import torch
import cv2
import numpy as np


class Predictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Load the model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dict = torch.load(model_path, map_location=self.device)
        self.model = self.model_dict['model']
        self.model.eval()  # Set the model to evaluation mode

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.half()

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image using mean and std
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        image = (image - mean) / std
        return image

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference using YOLO
        """
        assert image.shape == (640, 640, 3), "Input image must be 640x640x3"

        # Normalize the image
        img = self.normalize_image(image)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = torch.tensor(img, dtype=torch.float16).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)

        boxes, masks, scores, class_labels = self.process_outputs(outputs)

        output_image = self.draw_boxes(image, boxes, masks, scores, class_labels)

        return output_image

    def process_outputs(self, outputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process model outputs for mark-up.
        """
        boxes = []
        masks = []
        scores = []
        class_labels = []

        detections = outputs[0]
        proto = outputs[1]

        # Assuming detections have shape [1, num_detections, 5]
        # and proto contains mask information separately
        detections = detections[0]  # Remove batch dimension
        conf_mask = detections[:, 4] > 0.5  # Apply confidence threshold
        filtered_detections = detections[conf_mask]

        for detection in filtered_detections:
            x1, y1, x2, y2 = detection[:4].cpu().numpy()
            score = detection[4].cpu().numpy()
            # Here, we need to infer class label from another source if not provided
            class_label = 0  # Default to class 0 if not available
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_labels.append(class_label)
            masks.append(proto)  # Assuming proto contains mask data

        # Move tensors to CPU and convert to numpy arrays
        return np.array(boxes.cpu()), np.array(masks.cpu()), np.array(scores.cpu()), np.array(class_labels.cpu())

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
        """
        Draw image mark-up.
        """
        for box, mask, score, class_label in zip(boxes, masks, scores, class_labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_label}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # If masks are available and valid, add them to the image
            if mask is not None:
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                image[mask > 0.5] = [0, 255, 0]  # Example mask overlay

        return image
