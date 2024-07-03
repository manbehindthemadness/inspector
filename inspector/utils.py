import cv2
import numpy as np
import cupy as cp

debug_pipeline = False
show_target = True


def crop_and_resize_frame(
        frame: np.ndarray, boxes: np.ndarray,
        uncenter: bool = False, target_size: int = 640,
) -> tuple[np.ndarray, tuple[int, int, int, int], int, list[tuple[int, int, int, int]]]:
    """
    This will take the largest ROI from the source capture and create a square mat that matches the model inputs.

    NOTE: This is a good candidate for GPU acceleration.
    """

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
                pass_box = box

        if largest_box is None:
            raise ValueError("No boxes provided")
        else:
            boxes = [pass_box]  # noqa

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

        if cropped_frame.size == 0:
            raise ValueError("Cropped frame is empty")

        # Resize the cropped frame to 640x640 while preserving aspect ratio
        h, w = cropped_frame.shape[:2]

        if h == 0 or w == 0:
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
        result = final_frame

    except Exception as e:
        if debug_pipeline:
            print(f"Error occurred: {e}")
        result = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = 0, 0, 0, 0  # If there's an error, the coordinates will be (0, 0)

    return result, (x1, y1, x2, y2), target_size, boxes


def plot_boxes(
        frame: np.ndarray, boxes: [np.ndarray, list[tuple[int, int, int, int]]], colors: [np.ndarray, None],
        scores: [np.ndarray, None], color: [tuple, None] = None):
    """
    This is an image mark-up used for debugging the source capture.
    """

    def overlap(b1, b2):
        """
        Yeah, this doesn't seem to work...

        (x1, y1, x2, y2), (x1, y1, x2, y2)
        """
        return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

    label_coords = list()
    color_black = (0, 0, 0)
    for i in range(len(boxes)):
        box = boxes[i]
        if isinstance(box, np.ndarray):
            y1 = (frame.shape[0] * box[0]).astype(int)
            y2 = (frame.shape[0] * box[2]).astype(int)
            x1 = (frame.shape[1] * box[1]).astype(int)
            x2 = (frame.shape[1] * box[3]).astype(int)
        elif isinstance(box, list) or isinstance(box, tuple):
            y1, y2, x1, x2, _ = box
        else:
            raise TypeError(f"expected list, tuple or np.ndarray, not {type(box)}")
        if colors is not None:
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        if color is None:
            color = (0, 0, 0)

        score = str()
        if scores is not None:
            score = f"{scores[i]:.2f}"
        elif len(box) == 5:
            score = box[-1]

        (w1, h1), _ = cv2.getTextSize(score, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        lc_x1, lc_y1, lc_x2, lc_y2 = x1, y1, x1 + 50 + h1, y1 - 15

        relocate = False
        for j in range(len(label_coords)):
            relocate = False
            if overlap(label_coords[j], (lc_x1, lc_y1, lc_x2, lc_y2)):
                relocate = True
        in_view = True
        if relocate:
            print('relocating')
            distance = lc_y2 - lc_y1
            lc_y1 += distance
            if lc_y1 < 0:
                in_view = False
            else:
                lc_y2 += distance

        cl = color  # Change the color of the target diagnostic frame.
        if show_target and not i:
            cl = (172, 47, 117)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cl, 1)
        if in_view:
            lc_coord = (lc_x1, lc_y1, lc_x2, lc_y2)
            label_coords.append(lc_coord)
            cv2.rectangle(frame, (lc_x1, lc_y1), (lc_x2, lc_y2), color, -1)
            cv2.putText(frame, score, (x1 + 10, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_black)


def transform_boxes(
        boxes: list[tuple[int, int, int, int, str]],  # List of boxes as [(x1, y1, x2, y2, label)]
        origin_box: tuple[int, int, int, int],  # The bounding box extracted from the original image (x1, y1, x2, y2).
        resized_size: tuple[int, int] = (640, 640)  # The resized value of the origin_box (x, y).
):
    """
    Transforms detection boxes from the cropped and resized region back to the coordinates of the original image.

    Args:
    - boxes: List of bounding boxes detected in the cropped and resized region.
             Each box is represented as a tuple (x1, y1, x2, y2, label).
    - origin_box: Tuple of (x1, y1, x2, y2) representing the bounding box of the cropped area in the original image.
    - resized_size: Tuple of (resized_width, resized_height) representing the size of the cropped region after resizing.

    Returns:
    - transformed_boxes: List of bounding boxes transformed to the original image coordinates in (y1, y2, x1, x2, label)
    """
    transformed_boxes = []
    orig_x1, orig_y1, orig_x2, orig_y2 = origin_box

    if show_target:
        # Add origin dummy box for troubleshooting
        transformed_boxes.append((orig_y1, orig_y2, orig_x1, orig_x2, 'target'))

    # Calculate the width and height of the original and resized bounding box
    original_box_width = orig_x2 - orig_x1
    original_box_height = orig_y2 - orig_y1
    resized_width, resized_height = resized_size

    # Calculate the scaling factors from resized to original box size
    scale_x = original_box_width / resized_width
    scale_y = original_box_height / resized_height

    for box in boxes:
        x1, y1, x2, y2, label = box

        # Transform the coordinates to the original image
        orig_x1_transformed = int(x1 * scale_x) + orig_x1
        orig_y1_transformed = int(y1 * scale_y) + orig_y1
        orig_x2_transformed = int(x2 * scale_x) + orig_x1
        orig_y2_transformed = int(y2 * scale_y) + orig_y1

        transformed_boxes.append(
            (orig_y1_transformed, orig_y2_transformed, orig_x1_transformed, orig_x2_transformed, label)
        )

    return transformed_boxes


class KalmanFilter:
    """
    You guessed it, a super-fancy filter in the classical style of Rudolf E. Kálmán.

    https://arxiv.org/pdf/1611.06467
    """
    def __init__(self):
        self.dt = 1.0
        self.F = cp.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=cp.float32)
        self.H = cp.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=cp.float32)
        self.P = cp.eye(4, dtype=cp.float32)
        self.Q = cp.eye(4, dtype=cp.float32) * 0.01
        self.R = cp.eye(2, dtype=cp.float32) * 0.1
        self.x = cp.zeros((4, 1), dtype=cp.float32)

    def predict(self):
        """
        Apply temporal metrics.
        """
        self.x = cp.dot(self.F, self.x)
        self.P = cp.dot(cp.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z: float):
        """
        Update metrics with newly discovered values.
        """
        y = z - cp.dot(self.H, self.x)
        s = cp.dot(self.H, cp.dot(self.P, self.H.T)) + self.R
        k = cp.dot(cp.dot(self.P, self.H.T), cp.linalg.inv(s))
        self.x = self.x + cp.dot(k, y)
        i = cp.eye(self.F.shape[1], dtype=cp.float32)
        self.P = cp.dot(i - cp.dot(k, self.H), self.P)


class Track:
    """
    Let's track some objects babah!
    """
    def __init__(self, bbox):
        self.bbox = cp.array(bbox, dtype=cp.float32)
        self.kf = KalmanFilter()
        self.time_since_update = 0
        self.id = 0
        self.hits = 0

    def update(self, bbox: cp.ndarray):
        """
        Add newly found object tracks.
        """
        self.bbox = cp.array(bbox, dtype=cp.float32)
        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        """
        Apply kalman temporal predictions.
        """
        self.kf.predict()
        self.time_since_update += 1


class Sort:
    """
    The aptly named class that orders our detections.
    """
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []

    def update(self, detections):
        """
        Updates our tracks with new detections.
        """
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.tracks.append(Track(detections[i]))
            return self.tracks

        updated_tracks = []
        detections = cp.array(detections, dtype=cp.float32)

        track_boxes = cp.array([t.bbox.get() for t in self.tracks], dtype=cp.float32)
        distance_matrix = cp.zeros((len(track_boxes), len(detections)), dtype=cp.float32)

        for t, trk in enumerate(track_boxes):
            for d, det in enumerate(detections):
                distance_matrix[t, d] = cp.sqrt((trk[0] - det[0]) ** 2 + (trk[1] - det[1]) ** 2)

        row_ind, col_ind = cp.unravel_index(cp.argsort(distance_matrix.ravel()), distance_matrix.shape)
        row_ind = row_ind.get().tolist()
        col_ind = col_ind.get().tolist()

        used_rows = set()
        used_cols = set()
        for r, c in zip(row_ind, col_ind):
            if r in used_rows or c in used_cols:
                continue
            self.tracks[r].update(detections[c].get())
            updated_tracks.append(self.tracks[r])
            used_rows.add(r)
            used_cols.add(c)

        unused_tracks = [i for i in range(len(self.tracks)) if i not in used_rows]
        unused_detections = [i for i in range(len(detections)) if i not in used_cols]

        for i in unused_tracks:
            self.tracks[i].predict()
            if self.tracks[i].time_since_update > self.max_age:
                continue
            updated_tracks.append(self.tracks[i])

        for i in unused_detections:
            new_track = Track(detections[i].get())
            updated_tracks.append(new_track)

        self.tracks = [t for t in updated_tracks if t.hits >= self.min_hits or t.time_since_update <= self.max_age]

        return self.tracks
