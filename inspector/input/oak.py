import cv2
import depthai as dai
import numpy as np
import blobconverter
from inspector.input.base import CaptureBase

CAMERA_MODELS = [
    'oak-d',
    'oak-d pro',
    'oak-d sr',
    'oak-d lr',
]


class Camera(CaptureBase):
    """
    Load up the camera pipeline and resources.`
    """
    AF = True

    plain_image = None
    marked_image = None
    image_name = 'cap'

    control_queue = None
    ctrl = None

    def __init__(self, debug: bool = False, preprocess: bool = True, draw: bool = True,
                 camera_model: CAMERA_MODELS = 'oak-d'):
        super().__init__(debug, preprocess, draw)
        self.debug = debug
        self.preprocess = preprocess
        self.draw = draw
        self.nn_path = blobconverter.from_zoo(
            name="mobile_object_localizer_192x192",
            zoo_type="depthai",
            shaves=6
        )
        self.nn_width = 192
        self.nn_height = 192
        self.preview_width = 800
        self.preview_height = 800

        self.threshold = 0.2

        pipeline = dai.Pipeline()

        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(self.nn_path)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        # Color camera
        self.cam = cam = pipeline.create(dai.node.ColorCamera)
        match camera_model:
            case 'oak-d sr':
                cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
                self.AF = False
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setFps(25)

        control_in = pipeline.createXLinkIn()
        control_in.setStreamName('control')
        control_in.out.link(cam.inputControl)

        # Create manip
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(self.nn_width, self.nn_height)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        manip.initialConfig.setKeepAspectRatio(True)

        # Link preview to manip and manip to nn
        cam.preview.link(manip.inputImage)
        manip.out.link(detection_nn.input)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("cam")
        xout_rgb.input.setBlocking(False)
        cam.preview.link(xout_rgb.input)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        xout_nn.input.setBlocking(False)
        detection_nn.out.link(xout_nn.input)

        xout_manip = pipeline.create(dai.node.XLinkOut)
        xout_manip.setStreamName("manip")
        xout_manip.input.setBlocking(False)
        manip.out.link(xout_manip.input)

        self.pipeline = pipeline

        self.ctrl = dai.CameraControl()

    def _toggle_focus(self):
        """
        Toggles autofocus.
        """
        self.auto_focus = not self.auto_focus
        self._set_osd_message(f'toggle auto focus {self.auto_focus}')
        match self.auto_focus:
            case True:
                self.ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                self.ctrl.setAutoFocusTrigger()  # Trigger autofocus
            case False:
                self.ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                self._set_focus()
        self.control_queue.send(self.ctrl)

    def _set_focus(self, value_shift: int = 0):
        """
        This will send an autofocus adjustment to the camera.
        """
        if not self.auto_focus:
            self.focus += value_shift
            self.focus = np.clip(self.focus, 0, 255)
            self.ctrl.setManualFocus(self.focus)
            self.control_queue.send(self.ctrl)
            self._set_osd_message(f'focus {value_shift}, now {self.focus}')
        else:
            self._set_osd_message('manual focus disabled')

    def _focus_control(self):
        """
        This is a callback allowing us to manually control the camera focus.
        """
        key = cv2.waitKey(1) & 0xFF
        match key:
            case 32:
                images = [self.plain_image, self.marked_image]
                if all(element is not None for element in images):
                    self._capture_image(self.image_name, images)
                else:
                    self._set_osd_message('no images available to save')
            case 27:
                self._set_osd_message('exiting application')
                self._exit()
            case 149:
                self._set_focus(1)
            case 151:
                self._set_focus(8)
            case 154:
                self._set_focus(32)
            case 156:
                self._set_focus(-1)
            case 153:
                self._set_focus(-8)
            case 155:
                self._set_focus(-32)
            case 158:
                self._toggle_focus()

    def run(self, callback: any = None, thresh: float = .4):
        """
        This is the application capture loop.
        """

        with dai.Device(self.pipeline) as device:
            q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
            q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            self.control_queue = device.getInputQueue('control')

            self.set_times()

            while not self.term:
                in_cam = q_cam.get()
                in_nn = q_nn.get()
                in_manip = q_manip.get()

                self.focus = in_cam.getLensPosition()  # noqa

                frame = in_cam.getCvFrame()  # noqa
                self.plain_image = np.array(frame)
                frame_manip = in_manip.getCvFrame()  # noqa
                frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)

                # get outputs
                detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))  # noqa
                detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))  # noqa

                focus_frame, boxes, colors, scores, origins, target_size, data = self.prepare_regions(
                    frame, thresh, detection_scores, detection_boxes, callback
                )

                # draw boxes
                self.plot_boxes(frame, boxes, colors, scores)
                if self.debug and self.draw:
                    self.plot_boxes(frame_manip, boxes, colors, scores)

                self.viewer(
                    frame, focus_frame, data, origins, target_size, self._focus_control
                )
                self.marked_image = np.array(frame)

            cv2.destroyAllWindows()
