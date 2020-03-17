import numpy as np
import keras.backend as K

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization

# Taken from https://www.youtube.com/watch?v=IS0V8z8HXrM

GAMMA = 0.99

LR = 0.01
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

        x = Dense(1024, activation='linear')(x)
        x = Dense(256, activation='linear')(x)
        props = Dense(NR_OF_ACTIONS, activation='linear')(x)

        policy = Model(inputs=[input], outputs=[props])
        policy.compile(optimizer='adam', loss='mean_squared_error')

        predict = Model(inputs=[input], outputs=[props])
        return policy, predict

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        return self.predict.predict(state)[0]


    def store_transaction(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def get_discounted_rewards(self):
        discounted_rewards = np.zeros(len(self.reward_memory))
        running_add = 0

        for i in reversed(range(len(discounted_rewards))):
            running_add = self.reward_memory[i] + running_add * GAMMA # belman equation
            discounted_rewards[i] = running_add

        # standardize the rewards
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()
        discounted_rewards = discounted_rewards.squeeze()
        return discounted_rewards


    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)

        if (len(action_memory) < 2):
            print("Invalid terminal state")
            return 0

        print("Observations for training:", len(state_memory))
        actions = np.clip(action_memory, -1, 1)

        # discount the rewards
        discounted_rewards = self.get_discounted_rewards()
        action_updates = np.zeros_like(actions)
       
        for i in range(len(actions)):
            reward = discounted_rewards[i]

            if (reward < 0):
                noise = np.random.randn()
                action_updates[i] = LR * reward + noise * LR

        # print(action_updates)
        X = np.array(state_memory)
        y = np.array(actions + action_updates)

        result = self.policy.fit(X, y, batch_size=512, epochs=1, verbose=0, shuffle=True)
        return result.history["loss"][-1]


    def get_steps_count(self):
        return sum(self.action_memory)


    def get_current_action(self):
        return self.action_memory[-1]