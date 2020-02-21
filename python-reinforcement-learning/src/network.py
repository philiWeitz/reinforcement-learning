import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential

IMG_HEIGHT = 50
IMG_WIDTH = 120
IMG_DEPTH_DIM = 1

INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH_DIM)

def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), data_format='channels_last', input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, (3, 3)))
    model.add(Conv2D(32, (5, 5), strides=2))
    model.add(MaxPooling2D(pool_size=(3,3), padding='valid'))
    model.add(Flatten())
    model.add(Dense(120, activation='linear'))
    model.add(Dense(6, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prediction_to_motion(prediction):
    vertical = np.argmax(prediction[:3])
    horizontal = np.argmax(prediction[3:])

    if vertical == 0:
        vertical = "FORWARD"
    elif vertical == 1:
        vertical = "BACKWARDS"
    else:
        vertical = "STOP"

    if horizontal == 0:
        horizontal = "LEFT"
    elif horizontal == 1:
        horizontal = "RIGHT"
    else:
        horizontal = "STOP"
    
    motion = {}
    motion['vertical'] = vertical
    motion['horizontal'] = horizontal

    return motion

def predict(image):
    img = np.expand_dims(image, axis=2)
    prediction = model.predict(np.array([img]))
    return prediction_to_motion(prediction[0])

model = create_model()