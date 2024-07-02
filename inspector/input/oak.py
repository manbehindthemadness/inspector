import cv2
import depthai as dai
import numpy as np
import blobconverter
import time
from inspector.utils import plot_boxes, crop_and_resize_frame, transform_boxes


class Camera:
    """
    Load up the camera pipeline and resources.`
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
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
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setFps(25)

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

    def run(self, callback: any = None):
        """
        This is the application capture loop.
        """

        with dai.Device(self.pipeline) as device:
            np.random.seed(0)
            colors_full = np.random.randint(255, size=(100, 3), dtype=int)

            q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
            q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            start_time = time.time()
            counter = 0
            fps = 0

            while True:
                in_cam = q_cam.get()
                in_nn = q_nn.get()
                in_manip = q_manip.get()

                frame = in_cam.getCvFrame()  # noqa
                frame_manip = in_manip.getCvFrame()  # noqa
                frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)

                # get outputs
                detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))  # noqa
                detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))  # noqa

                # keep boxes bigger than threshold
                mask = detection_scores >= self.threshold
                boxes = detection_boxes[mask]
                colors = colors_full[mask]
                scores = detection_scores[mask]

                focus_frame, origins, target_size, boxes = crop_and_resize_frame(frame, boxes)
                data = None
                if callback:
                    focus_frame, data = callback(focus_frame)

                # draw boxes
                plot_boxes(frame, boxes, colors, scores)
                plot_boxes(frame_manip, boxes, colors, scores)

                if data is not None:  # Draw the boxes from the YOLO model.
                    cropped_size = target_size, target_size
                    yolo_boxes = transform_boxes(data, origins, cropped_size)
                    plot_boxes(frame, yolo_boxes, None, None, color=(0, 255, 0))

                # show fps and predicted count
                color_black, color_white = (0, 0, 0), (255, 255, 255)
                label_fps = "Fps: {:.2f}".format(fps)
                (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
                cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
                cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                            0.4, color_black)

                # show frame
                cv2.imshow("Localizer", frame)
                if self.debug:
                    cv2.imshow("Manip + NN", frame_manip)
                    cv2.imshow("Focused", focus_frame)

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time)

                    counter = 0
                    start_time = time.time()

                if cv2.waitKey(1) == ord('q'):
                    break
