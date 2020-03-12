import time

import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout
from tensorflow.keras.models import load_model, Sequential, Model


IMG_HEIGHT = 50
IMG_WIDTH = 120
IMG_DEPTH_DIM = 1

INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH_DIM)
OUTPUT_SHAPE = 3

GAMMA = 0.99


def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def get_discounted_rewards(memory):
    discounted_rewards = np.zeros(len(memory.rewards.data))
    running_add = 0

    for i in reversed(range(len(discounted_rewards))):
        if memory.terminals.data[i] == True:
            running_add = memory.rewards.data[i]
        else:
            running_add = memory.rewards.data[i] + running_add * GAMMA # belman equation
        discounted_rewards[i] = running_add

    # standardize the rewards
    discounted_rewards -= discounted_rewards.mean()
    discounted_rewards /= discounted_rewards.std()
    discounted_rewards = discounted_rewards.squeeze()
    return discounted_rewards


def get_adjusted_actions(memory, rewards):
    actions = np.array(memory.actions.data)

    gradients = np.gradient(actions, axis=0)

    for i in range(len(actions)):
        action = actions[i]
        reward = rewards[i]
        gradient = gradients[i]
        selected_action_idx = np.argmax(action)

        learning_rate = 0.1 if reward > 0 else -0.1
        action[selected_action_idx] += learning_rate * gradient[selected_action_idx] * reward

        next_idx = (selected_action_idx + 1) % 3
        action[next_idx] += learning_rate * gradient[next_idx] * reward

        next_idx = (next_idx + 1) % 3
        action[next_idx] += learning_rate * gradient[next_idx] * reward

    return actions


class Agent:

    def __init__(self):
        # only save the model if the step count is larger than the current max
        self.max_step_count = 200

        self.step = 0
        self.training = True
        self.create_model()


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), data_format='channels_last', input_shape=INPUT_SHAPE))
        model.add(Conv2D(32, (5, 5), strides=2))
        model.add(MaxPooling2D(pool_size=(3,3), padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='linear'))
        model.add(Dropout(0.25))
        model.add(Dense(OUTPUT_SHAPE, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.model = model


    def train(self, memory):
        steps_taken = len(memory.actions.data)
        if (steps_taken > self.max_step_count):
            print("Saving model...")
            self.model.save('trained-model.h5')
            self.max_step_count = steps_taken

        print("Training...")
        discounted_rewards = get_discounted_rewards(memory)
        adjusted_actions = get_adjusted_actions(memory, discounted_rewards)

        X = np.array(memory.observations.data)
        y = np.array(adjusted_actions)

        # increase the step size
        self.step += 1

        result = self.model.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True)
        return result.history["loss"][-1]


    def predict_move(self, image):
        prediction = self.model.predict(np.array([image]))
        # prediction is an array of arrays
        return prediction[0]
