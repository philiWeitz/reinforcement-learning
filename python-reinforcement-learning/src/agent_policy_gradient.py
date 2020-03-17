import numpy as np
import keras.backend as K

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization

# Taken from https://www.youtube.com/watch?v=IS0V8z8HXrM

LR = 0.1
OPTIMIZER_LR = 0.00001

NR_OF_ACTIONS = 1

class AgentPolicyGradient():
    def __init__(self):
        self.GAMMA=0.99
        self.G = 0
        self.policy, self.predict = self.build_policy_network()
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

        x = Dense(1024, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        props = Dense(NR_OF_ACTIONS, activation='linear')(x)

        def custom_loss_function(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik)

        policy = Model(inputs=[input], outputs=[props])
        policy.compile(optimizer=Adam(lr=OPTIMIZER_LR), loss=custom_loss_function)

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

        # discount the rewards
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1

            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount

            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        rewards = (G-mean) / std

        # calculate new actions
        gradients = np.gradient(actions, axis=0)
        adjusted_actions = actions.copy()

        for i in range(len(actions)):
            action = adjusted_actions[i]
            reward = rewards[i]
            gradient = gradients[i]

            # learning_rate = LR if reward > 0 else (LR * -1)
            action[0] += LR * gradient[0] * reward

        X = np.array(state_memory)
        y = np.array(adjusted_actions)

        result = self.policy.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True)
        return result.history["loss"][-1]


    def get_steps_count(self):
        return sum(self.action_memory)


    def get_current_action(self):
        return self.action_memory[-1]