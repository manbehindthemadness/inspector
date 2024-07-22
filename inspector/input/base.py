import os
import cv2
import numpy as np
import time
from inspector.utils import plot_boxes, crop_and_resize_frame, transform_boxes, get_utc_datetime_now_ticks


class CaptureBase:
    """
    Load an image from file and process it.
    """
    term = False

    image_dir = '~/Pictures/'

    counter = None
    fps = None
    start_time = None
    np.random.seed(0)
    colors_full = np.random.randint(255, size=(100, 3), dtype=int)

    cam = None
    focus = 100
    auto_focus = True

    osd_timer = 0
    osd_message = str()
    osd_opacity = 0.4
    last_opacity = float(osd_opacity)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
            self,
            debug: bool = False,
            preprocess: bool = True,
            draw: bool = True,
    ):
        self.debug = debug
        self.preprocess = preprocess
        self.draw = draw
        self.threshold = 0.2

        self.plot_boxes = plot_boxes
        self.transform_boxes = transform_boxes

        print(cv2.getBuildInformation())

    def _exit(self):
        """
        Bails us out of the run time.
        """
        self.term = True

    def _toggle_preprocessing(self):
        """
        Aptly named.
        """
        self.preprocess = not self.preprocess
        self._set_osd_message(f'preprocessing set to: {self.preprocess}')

    def _set_osd_message(self, message: str, timer: int = 15, upper: bool = True):
        """
        This will set out on-screen message along with the timer.
        """
        if upper:
            message = message.upper()
        self.osd_message = message
        print(self.osd_message)
        self.osd_timer = timer

    def _capture_image(self, name: str, images: list[np.ndarray]):
        """
        If used in the viewer callback we can selectively capture images.
        images will be a list of images that will be saved as <name>_1.jpg, <name_2.jpg>...
        """
        self.image_dir = os.path.expanduser('~/Pictures/')
        message = str()
        name += f"_{get_utc_datetime_now_ticks()}"
        for idx, image in enumerate(images):
            target_file = f"{self.image_dir}{name}_{idx}.jpg"
            cv2.imwrite(target_file, image)
            message += f"SAVED: {target_file}\n"
        self._set_osd_message(message, upper=False)

    @staticmethod
    def add_noise(image: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
        """
        Add monochromatic Gaussian noise to the image.
        """
        noise = np.random.normal(mean, std, image.shape[:2]).astype(np.uint8)
        noise = np.stack((noise, noise, noise), axis=-1)
        noisy_image = cv2.subtract(image, noise)
        return noisy_image

    def prepare_regions(
            self,
            frame: np.ndarray,
            thresh: float,
            detection_scores: np.ndarray,
            detection_boxes: np.ndarray,
            callback: any = None
    ):
        """
        Gets the boxes and detections ready for rendering.
        """

        # keep boxes bigger than threshold
        mask = detection_scores >= self.threshold
        boxes = detection_boxes[mask]
        colors = self.colors_full[mask]
        scores = detection_scores[mask]

        if not self.preprocess:
            boxes = list()
        focus_frame, origins, target_size, boxes = crop_and_resize_frame(frame, boxes)
        data = None
        if callback:
            focus_frame, data = callback(focus_frame, thresh)
        return focus_frame, boxes, colors, scores, origins, target_size, data

    def viewer(
            self,
            frame: np.ndarray,
            focus_frame: np.ndarray,
            data: list,
            origins: tuple[int, int, int, int],
            target_size: int,
            callback: any = None,
            extra: str = str(),
    ):
        """
        Render text and display final output on the screen.
        """

        def draw_centered_osd_message(image: np.ndarray, backdrop: bool = True):
            """
            Draws an On-Screen Display (OSD) message centered on the given image.

            Args:
                image (np.ndarray): The image on which to draw the message.
                backdrop (bool): If True, a semi-transparent black backdrop is drawn behind
                                    the text for better visibility.
            """
            osd_origins = list()
            x1s, x2s, y1s, y2s = list(), list(), list(), list()
            font = self.font
            font_scale = 0.5
            font_thickness = 1

            text_color = (255, 255, 255)
            backdrop_color = (0, 0, 0)

            lines = self.osd_message.split('\n')
            (text_width, text_height), baseline = cv2.getTextSize(lines[0], font, font_scale, font_thickness)
            line_height = text_height + baseline
            total_text_height = line_height * len(lines)
            start_y = (image.shape[0] - total_text_height) // 2 + text_height

            for i, line in enumerate(lines):
                (line_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                start_x = (image.shape[1] - line_width) // 2
                y = start_y + i * line_height

                if backdrop:
                    x1s.append(start_x - 5)
                    y1s.append(y - text_height - 5)
                    x2s.append(start_x + line_width + 5)
                    y2s.append(y + baseline + 5)

                osd_origins.append((line, start_x, y))

            if backdrop:
                top_left, bottom_right = (min(x1s), min(y1s)), (max(x2s), max(y2s))
                sub_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                backdrop_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                backdrop_rect[:] = backdrop_color
                blended_rect = cv2.addWeighted(sub_img, 1 - self.osd_opacity, backdrop_rect, self.osd_opacity, 0)
                image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blended_rect

            for i in osd_origins:
                line, start_x, y = i
                cv2.putText(
                    image, line, (start_x, y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA
                )

        if data is not None:  # Draw the boxes from the YOLO model.
            cropped_size = target_size, target_size
            yolo_boxes = self.transform_boxes(data, origins, cropped_size)
            if self.draw:
                self.plot_boxes(frame, yolo_boxes, None, None, color=(0, 255, 0))

        # show fps and predicted count
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(self.fps)
        label_fps += extra
        (w1, h1), _ = cv2.getTextSize(label_fps, self.font, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(
            frame, label_fps, (2, frame.shape[0] - 4), self.font, 0.4, color_black, lineType=cv2.LINE_AA
        )

        if callback:
            callback()

        if self.osd_timer > 0:
            draw_centered_osd_message(frame)
        else:
            self.osd_opacity = float(self.last_opacity)

        # show frame
        cv2.namedWindow('Localizer', cv2.WINDOW_NORMAL, )
        cv2.resizeWindow('Localizer', 800, 800)
        cv2.imshow("Localizer", frame)
        if self.debug:
            cv2.imshow("Focused", focus_frame)

        self.osd_timer -= 1
        self.counter += 1
        if (time.time() - self.start_time) > 1:
            self.fps = self.counter / (time.time() - self.start_time)
            self.counter = 0
            self.start_time = time.time()

    def set_times(self):
        """
        Resets our counters.
        """
        self.start_time = time.time()
        self.counter = 0
        self.fps = 0

    def _run(self, image_path: str, callback: any = None, thresh: float = .4, fuzz: [tuple[int, int], None] = None):
        """
        This is the application capture loop.
        """

        original_frame = cv2.imread(image_path)
        if original_frame is None:
            raise FileNotFoundError(f"Image at path {image_path} not found")

        original_frame = cv2.resize(original_frame, (640, 640))  # resize to 640x640 if needed

        self.set_times()

        while True:
            frame = np.array(original_frame)  # Duplicate the image mat.
            if fuzz:
                mean, std = fuzz
                frame = self.add_noise(frame, mean, std)

            # Simulate detection outputs
            detection_boxes = np.random.rand(100, 4)  # Dummy detection boxes
            detection_scores = np.random.rand(100)  # Dummy detection scores

            focus_frame, boxes, colors, scores, origins, target_size, data = self.prepare_regions(
                frame, thresh, detection_scores, detection_boxes, callback
            )

            # draw boxes
            self.plot_boxes(frame, boxes, colors, scores)

            self.viewer(
                frame, focus_frame, data, origins, target_size,
            )

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
