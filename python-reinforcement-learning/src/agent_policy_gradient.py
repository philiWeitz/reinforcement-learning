import numpy as np
import keras.backend as K

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization

# Taken from https://www.youtube.com/watch?v=IS0V8z8HXrM


NR_OF_ACTIONS = 1

class AgentPolicyGradient():
    def __init__(self):
        self.GAMMA=0.99
        self.G = 0
        self.lr = 0.0001
        self.policy, self.predict = self.build_policy_network()
        self.reset()


    def reset(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


    def build_policy_network(self):
        input = Input(shape=(50, 120, 1), name='img_in')
        advantages = Input(shape=[1])
        
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        props = Dense(NR_OF_ACTIONS, activation='linear')(x)

        def custom_loss_function(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * advantages)

        policy = Model(inputs=[input, advantages], outputs=[props])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss_function)

        predict = Model(inputs=[input], outputs=[props])
        return policy, predict

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        return self.predict.predict(state)[0]


    def store_transaction(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        print("Observations for training:", len(state_memory))
        actions = np.clip(action_memory, -1, 1)

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1

            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount

            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean) / std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)
        return cost


    def get_steps_count(self):
        return sum(self.action_memory)


    def get_current_action(self):
        return self.action_memory[-1]