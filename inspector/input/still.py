from inspector.input.base import CaptureBase


class StillImage(CaptureBase):
    """
    Allows us to play with single images.
    """
    def __init__(self, debug: bool = False, preprocess: bool = True, draw: bool = True):
        super().__init__(debug, preprocess, draw)
        self.debug = debug
        self.preprocess = preprocess
        self.draw = draw

        self.run = self._run
