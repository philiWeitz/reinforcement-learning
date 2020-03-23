import functools
import numpy as np
import keras.backend as K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization, TimeDistributed, LSTM


GAMMA = 0.99
NR_OF_ACTIONS =3


class AgentTemporalDifference():
    def __init__(self):
        self.T = 0
        self.policy = self.build_policy_network()
        self.reset()


    def reset(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


    def build_policy_network(self):
        input = Input(shape=(50, 120, 1), name='img_in')
               
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)

        x = Dense(512, activation='linear')(x)
        steering = Dense(NR_OF_ACTIONS, activation='linear')(x)

        policy = Model(inputs=[input], outputs=[steering])
        policy.compile(optimizer='adam', loss='mean_absolute_error')
        
        return policy

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        return self.policy.predict(state)[0]


    def store_transaction(self, observation, action, reward):
        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        self.action_memory.append(action)


    def get_reward(self, is_on_track, is_terminal_state):
        return 0 if is_on_track else -0.1


    def learn(self, is_finish_reached=False):
        self.T += 1
        learning_rate = 0.001

        rewards = self.reward_memory
        actions = self.action_memory
        state_memory = self.state_memory

        value_final_state = 5.0 if is_finish_reached else -5.0

        # calculate the temporal difference
        rewards_gamma_tmp = np.array([value_final_state])
        rewards_discounted = np.array([])

        rewards.reverse()
        for reward in rewards:
            rewards_gamma_tmp *= GAMMA 
            rewards_gamma_tmp = np.append(rewards_gamma_tmp, reward)
            rewards_discounted = np.append(sum(rewards_gamma_tmp), rewards_discounted)

        rewards.reverse()

        # calculate the update
        state_values = [max(action) for action in actions]
        value_delta = (rewards_discounted - state_values) * learning_rate

        # update the values
        for action_idx in range(len(actions)):
            update = value_delta[action_idx]
            action = actions[action_idx]
            selected_idx = np.argmax(action)
            action[selected_idx] += update

        # learn the new values
        X = np.array(state_memory)
        y = np.clip(actions, -5, 5)

        result = self.policy.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True)
        return result.history["loss"][-1]


    def get_steps_count(self):
        return len(self.action_memory)


    def get_current_action(self):
        if (len(self.action_memory) < 1):
            return [0.0]

        action_idx = np.argmax(self.action_memory[-1])

        # left steering
        if action_idx == 0:
            return [-0.8]
        # right steering
        if action_idx == 2:
            return [0.8]
        # center steering
        return [0.0]
    
    def save_prediction_model(self):
        self.policy.save('model.h5')