
import numpy as np
from PIL import Image, ImageOps

from tensorflow.keras.models import load_model, Sequential, Model

IMG_HEIGHT = 50
IMG_WIDTH = 120

# load the image
img = Image.open('./markkuu/right-curve.jpg')
img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
img = ImageOps.grayscale(img)
img = np.expand_dims(img, axis=2)

# load the model
model = load_model('trained-model.h5')

prediction = model.predict(np.array([img]))
print(prediction)