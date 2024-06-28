import cupy as cup
import cv2
import numpy as np


cached_frame = None
debug_pipeline = False


def crop_and_resize_frame(frame, boxes, uncenter=False):
    global cached_frame

    try:
        # Find the largest box
        largest_box = None
        largest_area = 0

        for box in boxes:
            y1 = int(frame.shape[0] * box[0])
            y2 = int(frame.shape[0] * box[2])
            x1 = int(frame.shape[1] * box[1])
            x2 = int(frame.shape[1] * box[3])

            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        if largest_box is None and debug_pipeline:
            raise ValueError("No boxes provided")

        # Expand the largest box into a square
        x1, y1, x2, y2 = largest_box
        box_width = x2 - x1
        box_height = y2 - y1
        square_side = max(box_width, box_height)

        if uncenter:
            # Optional: adjust the box to un-center it during the transformation
            if box_width > box_height:
                y1 = max(0, y1 - (square_side - box_height))
            else:
                x1 = max(0, x1 - (square_side - box_width))

        # Ensure the box is within the frame boundaries
        x2 = min(frame.shape[1], x1 + square_side)
        y2 = min(frame.shape[0], y1 + square_side)
        x1 = max(0, x2 - square_side)
        y1 = max(0, y2 - square_side)

        cropped_frame = frame[y1:y2, x1:x2]

        if cropped_frame.size == 0 and debug_pipeline:
            raise ValueError("Cropped frame is empty")

        # Resize the cropped frame to 640x640 while preserving aspect ratio
        target_size = 640
        h, w = cropped_frame.shape[:2]

        if h == 0 or w == 0 and debug_pipeline:
            raise ValueError("Invalid dimensions of cropped frame")

        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        resized_cropped_frame = cv2.resize(cropped_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a blank 640x640 image and place the resized frame at the center
        final_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        final_frame[start_y:start_y + new_h, start_x:start_x + new_w] = resized_cropped_frame

        # Update the cached frame
        cached_frame = final_frame

        return final_frame

    except Exception as e:
        if debug_pipeline:
            print(f"Error occurred: {e}")
        if cached_frame is not None and debug_pipeline:
            print("Using cached frame")
            return cached_frame
        else:
            raise e


def plot_boxes(frame, boxes, colors, scores):
    color_black = (0, 0, 0)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        y1 = (frame.shape[0] * box[0]).astype(int)
        y2 = (frame.shape[0] * box[2]).astype(int)
        x1 = (frame.shape[1] * box[1]).astype(int)
        x2 = (frame.shape[1] * box[3]).astype(int)
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
        cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)


def init_cupy() -> cup.ndarray:
    """
    This just forces cupy to load its resources.
    """
    a = cup.array([1, 1])
    a + 1
    return a
