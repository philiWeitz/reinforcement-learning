import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization, TimeDistributed, LSTM


LEARNING_RATE = 0.001

GAMMA = 0.99
NR_OF_ACTIONS = 3


class Agent():
    def __init__(self):
        self.T = 0
        self.model = self.build_model()
        self.reset()


    def reset(self):
        self.max_tiles_count = 0
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.selected_action_memory = []
        self.on_track_frame_counter = 0


    def build_model(self):
        input = Input(shape=(50, 120, 1), name='img_in')
               
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)

        x = Dense(128, activation='relu')(x)
        steering = Dense(NR_OF_ACTIONS, activation='relu')(x)

        model = Model(inputs=[input], outputs=[steering])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

        print(model.summary())  
        return model

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        return self.model.predict(state)[0]


    def store_transaction(self, observation,, is_on_track, is_final_state):
        action_vector = self.choose_action(observation)
        reward = self.get_reward(is_on_track, is_terminal_state)
        
        # select an action
        self.selected_action_memory.append(np.argmax(action_vector))

        if is_on_track:
            self.on_track_frame_counter += 1

        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        self.action_memory.append(action_vector)


    def get_reward(self, is_on_track, is_terminal_state):
        nr_of_tiles_done = round(self.on_track_frame_counter / 10)
        if nr_of_tiles_done > self.max_tiles_count:
            self.max_tiles_count = nr_of_tiles_done
            return 3.0
        return -0.1


    def learn(self, is_finish_reached=False):
        self.T += 1

        rewards = self.reward_memory
        actions = self.action_memory
        state_memory = self.state_memory
        selected_action_memory = self.selected_action_memory

        # calculate the expected q values
        expected_q_values = []

        # bellman equation calculation
        for reward_idx in reversed(range(len(rewards))):
            if len(expected_q_values) <= 0:
                expected_q_values.insert(0, 0)
            else:
                reward = rewards[reward_idx-1]
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

        # log tensorboard data every 40 runs
        callbacks = [tf.keras.callbacks.TensorBoard('logs')] if (self.T % 40) == 39 else []

        result = self.model.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True, callbacks=callbacks)
        return result.history["loss"][-1]


    def get_steps_count(self):
        return len(self.action_memory)


    def get_current_action(self):
        if (len(self.action_memory) < 1):
            return [0.0]

        action_idx = self.selected_action_memory[-1]

        # left steering
        if action_idx == 0:
            return [-0.8]
        # right steering
        if action_idx == 2:
            return [0.8]
        # center steering
        return [0.0]
    
    
    def save_model(self):
        self.model.save('model.h5')