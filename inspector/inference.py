import torch
import cv2
import numpy as np


class Predictor:
    def __init__(self, model_path, device='cpu'):
        # Load the model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dict = torch.load(model_path, map_location=self.device)
        self.model = self.model_dict['model']
        self.model.eval()  # Set the model to evaluation mode

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, image):
        assert image.shape == (640, 640, 3), "Input image must be 640x640x3"

        # Preprocess the image
        img = image / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # Change to CxHxW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = torch.tensor(img, dtype=torch.float16).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(img)

        # Post-process the outputs to get boxes, scores, and class labels
        boxes, scores, class_labels = self.process_outputs(outputs)

        # Draw boxes, scores, and class labels on the image
        output_image = self.draw_boxes(image, boxes, scores, class_labels)

        return output_image

    @staticmethod
    def process_outputs(outputs):
        # Placeholder function to process the model outputs
        # This will vary depending on your specific YOLO model implementation
        # You will need to adapt this function to extract boxes, scores, and class labels
        boxes = []
        scores = []
        class_labels = []
        # Example processing, adjust as necessary
        for output in outputs:
            for detection in output:
                box = detection[:4].cpu().numpy()
                score = detection[4].cpu().numpy()
                class_label = detection[5].cpu().numpy()
                boxes.append(box)
                scores.append(score)
                class_labels.append(class_label)
        return boxes, scores, class_labels

    @staticmethod
    def draw_boxes(image, boxes, scores, class_labels):
        for box, score, class_label in zip(boxes, scores, class_labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_label}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
