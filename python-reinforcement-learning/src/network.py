import time
import random

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
OUTPUT_SHAPE = 3

GAMMA = 0.99

max_discount_rewards_mean = 1


def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), data_format='channels_last', input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, (3, 3)))
    model.add(Conv2D(32, (5, 5), strides=2))
    model.add(MaxPooling2D(pool_size=(3,3), padding='valid'))
    model.add(Flatten())
    model.add(Dense(200, activation='sigmoid'))   
    model.add(Dense(60, activation='sigmoid'))   
    model.add(Dense(OUTPUT_SHAPE, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def prediction_to_motion(prediction):
    horizontal = np.argmax(prediction)

    if horizontal == 0:
        horizontal = "LEFT"
    elif horizontal == 1:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"
    
    motion = {}
    motion['vertical'] = "FORWARD"
    motion['horizontal'] = horizontal

    return motion

def action_to_vector(action):
    result = [0,0,0]

    if action['horizontal'] == "LEFT":
        result[0] = 1.0
    elif action['horizontal'] == "RIGHT":
        result[1] = 1.0
    else:
        result[2] = 1.0

    return result

def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)

def train(state_record, action_record, discount_reward_record):
    global max_discount_rewards_mean
    print("Max Mean:", max_discount_rewards_mean)

    # standartdize the rewards
    discount_rewards = np.array(discount_reward_record)
    max_discount_rewards_mean = max(max_discount_rewards_mean, discount_rewards.mean())

    discount_rewards -= max_discount_rewards_mean
    discount_rewards /= discount_rewards.std()
    discount_rewards = discount_rewards.squeeze()
    print("Discounted Rewards:", discount_rewards);
    
    X = np.array([expand_image_dimension(image) for image in state_record])
    y = np.array([action_to_vector(action) for action in action_record])
 
    return model.train_on_batch([X, discount_reward_record], y)


# epsilon of 0 -> no random moves, epsilon of 100 -> 100% random moves
def predict(image, epsilon = 0):
    randNumber = random.randint(0, 100)

    # make a random move 
    if randNumber < epsilon:
        random_prediction = [ random.random() for i in range(OUTPUT_SHAPE) ]
        return prediction_to_motion(random_prediction)

    # else as the network to make a prediction
    img = expand_image_dimension(image)
    prediction = model.predict(np.array([img]))
    return prediction_to_motion(prediction[0])


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(discounted_r))):
        running_add =  r[t] + running_add * GAMMA # belman equation
        discounted_r[t] = running_add
    return discounted_r


model = create_model()