import time

import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D
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


class Agent:

    def __init__(self):
        self.step = 0
        self.training = True

        self.create_model()
                
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.3, value_test=.05, nb_steps=100)
        self.policy = policy
        self.policy._set_agent(self)


    def create_model(self):
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

        self.model = model


    def train(self, memory):
        print("Training...")
        discounted_rewards = get_discounted_rewards(memory);
        # print(discounted_rewards)

        X = np.array(memory.observations.data)
        y = np.array(memory.actions.data)

        # increase the step size
        self.step += 1

        return self.model.train_on_batch([X, discounted_rewards], y)


    def predict_move(self, image):
        q_values = self.model.predict(np.array([image]))

        # select an action
        position = self.policy.select_action(q_values=q_values[0])
        # set the action
        action = np.zeros(OUTPUT_SHAPE)
        action[position] = 1

        return action
