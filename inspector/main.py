import os
from .input.oak import Camera
from .inference import Predictor

cwd = os.path.dirname(os.path.abspath(__file__))

cam = Camera()
predictor = Predictor(
    model_path=f'{cwd}/models/medium.pt',
    device='cuda'
)

cam.run(callback=predictor.predict)
