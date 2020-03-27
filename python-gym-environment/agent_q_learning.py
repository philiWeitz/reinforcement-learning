import functools
import numpy as np
import keras.backend as K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization, TimeDistributed, LSTM


LEARNING_RATE = 0.001

GAMMA = 0.95
NR_OF_ACTIONS = 3


class AgentTemporalDifference():
    def __init__(self):
        self.model = self.build_model()
        self.reset()


    def reset(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.selected_action_memory = []


    def build_model(self):
        input = Input(shape=(96, 96, 3), name='img_in')
               
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)

        x = Dense(128, activation='relu')(x)
        steering = Dense(NR_OF_ACTIONS, activation='linear')(x)

        model = Model(inputs=[input], outputs=[steering])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_absolute_error')

        print(model.summary())  
        return model

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        return self.model.predict(state)[0]


    def store_transaction(self, observation, reward):
        action_vector = self.choose_action(observation)

        # select an action
        if sum(action_vector) < 0.01:
            self.selected_action_memory.append(np.random.choice(NR_OF_ACTIONS))
        else:
            self.selected_action_memory.append(np.argmax(action_vector))

        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        self.action_memory.append(action_vector)


    def learn(self, is_finish_reached=False):
        rewards = self.reward_memory
        actions = self.action_memory
        state_memory = self.state_memory
        selected_action_memory = self.selected_action_memory

        # calculate the expected q values
        expected_q_values = []

        # bellman equation calculation
        for reward_idx in reversed(range(len(rewards))):
            if len(expected_q_values) <= 0:
                expected_q_values.insert(0, 1)
            else:
                reward = rewards[reward_idx]
                discount = expected_q_values[0] * GAMMA
                expected_q_values.insert(0, reward + discount)

        # update the values
        for action_idx in range(len(actions)):
            action = actions[action_idx]
            selected_idx = selected_action_memory[action_idx]
            action[selected_idx] = expected_q_values[action_idx]

        # learn the new values
        X = np.array(state_memory)
        y = np.array(actions)

        # print some stats
        print('-----------------')
        # print('Observations received:', len(state_memory))
        # print('Max Q value:', y.max())
        # print('Min Q value:', y.min())

        result = self.model.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True)
        return result.history["loss"][-1]


    def get_current_action(self):
        result = [0.0, 1.0, 0.0]

        if (len(self.action_memory) < 1):
            return result

        action_idx = self.selected_action_memory[-1]

        # left steering
        if action_idx == 0:
            result[0] = -0.8
        # right steering
        elif action_idx == 2:
            result[0] = 0.8

        return result


    def save_prediction_model(self):
        self.model.save('car-racing-model.h5')