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

        output_image = self.draw_boxes_and_masks(image, boxes, masks, scores, class_labels)

        return output_image

    @staticmethod
    def process_outputs(outputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process model outputs for mark-up.
        """
        boxes = []
        masks = []
        scores = []
        class_labels = []

        # Assuming outputs have shape (batch_size, num_anchors, grid_height, grid_width, num_attributes)
        # num_attributes includes (x, y, w, h, confidence, class_probs)
        for r in outputs:
            for box in r.boxes:
                x1, y1, x2, y2 = box[:4].cpu().numpy()
                score = box[4].cpu().numpy()
                class_label = box[5].cpu().numpy()
                mask = r.masks  # Assuming r.masks contains the mask information
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_labels.append(class_label)
                masks.append(mask)

        return np.array(boxes), np.array(masks), np.array(scores), np.array(class_labels)

    @staticmethod
    def draw_boxes_and_masks(image: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
        """
        Draw image mark-up with boxes and masks.
        """
        for box, mask, score, class_label in zip(boxes, masks, scores, class_labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_label}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw mask
            mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
            mask_binary = mask_resized > 0.5
            colored_mask = np.zeros_like(image)
            colored_mask[y1:y2, x1:x2][mask_binary] = [0, 255, 0]  # Green mask
            image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

        return image
