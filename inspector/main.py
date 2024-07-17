import os
from inspector.inference import Predictor

USE_OAK = True  # Toggle this to use the camera or a still image.
threshold = 0.4  # Detection threshold
kwargs = dict()

if USE_OAK:
    from .input.oak import Camera
else:
    from .input.still import StillImage as Camera
    kwargs.update({
        'image_path': 'test.png',  # or whatever still image you choose.
        'fuzz': (1.0, 0.5)  # This adds fake camera transistor noise.
    })


cwd = os.path.dirname(os.path.abspath(__file__))

cam = Camera(
    debug=False,
    # preprocess=False,
    # draw=False
)
predictor = Predictor(
    model_path=f'{cwd}/models/medium.pt',
    # draw=True,
)

kwargs.update({
    'callback': predictor.predict,
    'thresh': threshold
})

cam.run(**kwargs)
