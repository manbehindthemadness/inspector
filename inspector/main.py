import os
from .input.oak import Camera
from .inference import Predictor

threshold = 0.4
cwd = os.path.dirname(os.path.abspath(__file__))

cam = Camera(debug=False)
predictor = Predictor(
    model_path=f'{cwd}/models/medium.pt',
    # draw=True,
)

cam.run(callback=predictor.predict, thresh=threshold)
