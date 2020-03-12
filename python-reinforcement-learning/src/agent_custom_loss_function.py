import numpy as np
import keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization

# Taken from https://www.youtube.com/watch?v=IS0V8z8HXrM

IMG_HEIGHT = 50
IMG_WIDTH = 120
IMG_DEPTH_DIM = 1

INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH_DIM)
OUTPUT_SHAPE = 3
NR_OF_ACTIONS = OUTPUT_SHAPE



class AgentNextGen():
    def __init__(self):
        self.GAMMA=0.99
        self.G = 0
        self.lr = 0.001
        self.policy, self.predict = self.build_policy_network()


    def build_policy_network(self):
        input = Input(shape=(50, 120, 1), name='img_in')
        advantages = Input(shape=[1])
        
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = BatchNormalization()(x)
        props = Dense(OUTPUT_SHAPE, activation='softmax')(x)

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
        propabilities = self.predict.predict(state)[0]
        print(propabilities)
        action = np.random.choice(NR_OF_ACTIONS, p=propabilities)
        return action


    def train(self, memory):
        state_memory = np.array(memory.observations.data)
        action_memory = np.array(memory.actions.data)
        reward_memory = np.array(memory.rewards.data)

        actions = np.zeros([len(action_memory), NR_OF_ACTIONS])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory, dtype='float')
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1

            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.GAMMA
            
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean) / std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)
        print("Cost:", cost)
