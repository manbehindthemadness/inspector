import cv2
import numpy as np
from inspector.input.base import CaptureBase


class StillImage(CaptureBase):
    """
    Allows us to play with single images.
    """
    fuzz = False

    mean = 1.0
    std = 0.5

    def __init__(self, debug: bool = False, preprocess: bool = False, draw: bool = True):
        super().__init__(debug, preprocess, draw)
        self.debug = debug
        self.preprocess = preprocess
        self.draw = draw

    @staticmethod
    def get_help():
        """
        This just returns a string of useful help information.
        """
        result = ("Keyboard:\n\n"
                  "Esc - Exit program\n"
                  "H - Display this message\n\n"
                  "Numpad:\n\n"
                  "[ (7) mean +0.1 ][ (8) ------ ][ (9) std +0.1 ]\n\n"
                  "[ (4) ------ ][ (5) ------ ][ (6) ------- ]\n\n"
                  "[ (1) mean -0.1 ][ (2) ------ ][ (3) std -0.1 ]\n\n"
                  "[ (0) toggle noise ]\n"
                  "plus - raise detection threshold 0.1\n"
                  "minus - lower detection threshold 0.1")
        return result

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add monochromatic Gaussian noise to the image.
        """
        if self.fuzz:
            noise = np.random.normal(self.mean, self.std, image.shape[:2]).astype(np.uint8)
            noise = np.stack((noise, noise, noise), axis=-1)
            noisy_image = cv2.subtract(image, noise)
            return noisy_image
        else:
            return image

    def _set_mean(self, mean: float):
        """
        This will adjust the mean values of the noise.
        """
        self.mean += mean
        self.mean = np.around(np.clip(self.mean, 0.0, 1.0), decimals=1)
        self._set_osd_message(f"fuzz: {self.fuzz}\n\nmean: {self.mean}\n\nstd: {self.std}")

    def _set_std(self, std: float):
        """
        This will adjust the mean values of the noise.
        """
        self.std += std
        self.std = np.around(np.clip(self.std, 0.0, 1.0), decimals=1)
        self._set_osd_message(f"fuzz: {self.fuzz}\n\nmean: {self.mean}\n\nstd: {self.std}")

    def _toggle_fuzz(self):
        """
        Enables or disables artificial noise.
        """
        self.fuzz = not self.fuzz
        self._set_osd_message(f"fuzz: {self.fuzz}\n\nmean: {self.mean}\n\nstd: {self.std}")

    def _noise_control(self):
        """
        This will allow us to control the introduction of artificial noise for testing.
        """
        key = cv2.waitKey(1) & 0xFF
        match key:
            case 27:  # Esc
                self._set_osd_message('exiting application')
                self._exit()
            case 55:  # Num 7
                self._set_mean(0.1)
            case 57:  # Num 9
                self._set_std(0.1)
            case 49:  # Num 1
                self._set_mean(-0.1)
            case 51:  # Num 3
                self._set_std(-0.1)
            case 48:  # Num 0
                self._toggle_fuzz()
            case 104:  # Keyboard H
                self.last_opacity = float(self.osd_opacity)
                self.osd_opacity = 1.0
                self._set_osd_message(self.get_help(), 120, upper=False)
        self._set_thresh(key)

    def run(self, image_path: str, callback: any = None):
        """
        Main loop.
        """
        self._run(
            image_path=image_path,
            callback=callback,
            preprocess_callback=self.add_noise,
            key_callback=self._noise_control
        )
        cv2.destroyAllWindows()
