import os
from inspector.inference import Predictor

cwd = os.path.dirname(os.path.abspath(__file__))

USE_OAK = True  # Toggle this to use the camera or a still image.
MODEL_PATH = f'{cwd}/models/medium.pt'
IMAGE_PATH = 'test.png'
kwargs = dict()

if USE_OAK:
    from .input.oak import Camera
else:
    from .input.still import StillImage as Camera
    kwargs.update({
        'image_path': IMAGE_PATH,  # or whatever still image you choose.
    })

cam = Camera(
)
predictor = Predictor(
    model_path=MODEL_PATH,
)

kwargs.update({
    'callback': predictor.predict,
})

cam.run(**kwargs)
