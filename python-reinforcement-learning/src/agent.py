import os
import gym
import pylab
import time
import itertools
import numpy as np
from statistics import mean

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Flatten, MaxPooling2D, Conv2D

from keras import backend as K

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


GAMMA = 0.99
EPISODES = 1000
LR_ACTOR = 0.001
LR_CRITIC = 0.005

EPOCHS = 4
MINI_BATCH_SIZE = 16
BATCH_SIZE = EPOCHS * MINI_BATCH_SIZE

CLIPPING = 0.1
ENTROPY_BETA = 0.00
# smoothing factor
ALPHA = 0.95

HEIGHT = 50
WIDTH = 120
DEPTH = 1

NR_OF_ACTIONS = 3
INPUT_SHAPE = (HEIGHT, WIDTH, DEPTH)


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def ppo_loss_function(advantage, old_prediction):
    def loss(y_true, y_pred):
        # y_true one hot encoded actions
        # pred is a softmax vector. 
        # prob is the probability of the taken aciton.
        prob = y_true * y_pred
        old_prob = y_true * old_prediction

        # create the ratio based on log probability
        ratio = K.exp(K.log(prob + 1e-10) - K.log(old_prob + 1e-10))

        clip_ratio = K.clip(ratio, min_value=(1 - CLIPPING), max_value=(1 + CLIPPING))
        surrogate1 = ratio * advantage
        surrogate2 = clip_ratio * advantage

        # add the entropy loss to avoid getting stuck on local minima
        entropy_loss = (prob * K.log(prob + 1e-10))
        ppo_loss = -K.mean(K.minimum(surrogate1,surrogate2) + ENTROPY_BETA * entropy_loss)
        return ppo_loss

    return loss


def get_generalized_advantage_estimations(reward_mem, value_mem, mask_mem, next_state_value):
    gae = 0
    return_mem = []
    episode_length = len(reward_mem)
   
    for t in reversed(range(episode_length)):
        value = value_mem[t]
        value_prime = next_state_value if (t+1) >= episode_length else value_mem[t+1]
        
        delta = reward_mem[t] + GAMMA * value_prime * mask_mem[t] - value
        gae = delta + GAMMA * ALPHA * mask_mem[t] * gae
        
        return_value = gae + value  
        return_mem.insert(0, return_value)
        
    return np.array(return_mem)


class PPOAgent:
    def __init__(self, state_size=INPUT_SHAPE, action_size=NR_OF_ACTIONS, gamma=GAMMA, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, load_model_from_file=False):
        self.load_model_from_file = load_model_from_file

        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        
        self.training_loss = []
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.reset()

        if self.load_model_from_file:
            self.actor.load_weights('model/ppo_actor.h5')
            self.critic.load_weights('model/ppo_critic.h5')
     
    def build_actor(self):
        advantage = Input(shape=(1,), name='advantage_input')
        old_prediction = Input(shape=(self.action_size,), name='old_prediction_input')
        loss = ppo_loss_function(advantage=advantage, old_prediction=old_prediction)

        state_input = Input(shape=self.state_size, name='state_input')
        serial = state_input
        serial = Conv2D(16, (3, 3), data_format='channels_last')(serial)
        serial = Conv2D(32, (5, 5), strides=2)(serial)
        serial = MaxPooling2D(pool_size=(3,3), padding='valid')(serial)
        serial = Flatten()(serial)
        serial = Dense(64, activation='relu', kernel_initializer='he_uniform')(serial)
        output = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(serial)
        
        actor = Model(inputs=[state_input, advantage, old_prediction], outputs=[output])
        actor.compile(loss=[loss], optimizer=Adam(lr=self.lr_actor))
        # actor.summary()
        return actor

    def build_critic(self):
        state_input = Input(shape=self.state_size, name='state_input')
        serial = state_input
        serial = Conv2D(16, (3, 3), data_format='channels_last')(serial)
        serial = Conv2D(32, (5, 5), strides=2)(serial)
        serial = MaxPooling2D(pool_size=(3,3), padding='valid')(serial)
        serial = Flatten()(serial)
        serial = Dense(64, activation='relu', kernel_initializer='he_uniform')(serial)
        output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(serial)

        critic = Model(inputs=[state_input], outputs=[output])
        critic.compile(loss='mse', optimizer=Adam(lr=self.lr_actor))
        # critic.summary()
        return critic
    
    def get_value(self, state):
        state_expanded = np.array(state)[np.newaxis, ::]
        return self.critic.predict(state_expanded)[0][0]

    def get_action(self, state):
        state_expanded = np.array(state)[np.newaxis, ::]
        probability = self.actor.predict([state_expanded, np.zeros((1, 1)), np.zeros((1, self.action_size))])[0]
        action_idx = np.random.choice(self.action_size, 1, p=probability)[0]
        return action_idx, probability

    def one_hot_ecode_actions(self, actions):
        length = len(actions)
        result = np.zeros((length, self.action_size))
        result[range(length), actions] = 1
        return result
    
    def train_model(self, states, advantages, actions, probabilities, gaes):
        one_hot_encoded_actions = self.one_hot_ecode_actions(actions)

        actor_loss = self.actor.fit(
            [states, advantages, probabilities],
            [one_hot_encoded_actions],
            verbose=False, shuffle=True, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE, validation_split=0.2)
        
        critic_loss = self.critic.fit(
            [states],
            [gaes],
            verbose=False, shuffle=True, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE, validation_split=0.2)
        self.training_loss = [mean(actor_loss.history['val_loss']), mean(critic_loss.history['val_loss'])]

    def save_models(self):
        try:
            os.mkdir('model')
        except OSError:
            pass

        self.actor.save("model/ppo_actor.h5")
        self.critic.save("model/ppo_critic.h5")

    def reset(self):
        self.T = 0
        self.value_mem = []
        self.state_mem = []
        self.action_mem = []
        self.reward_mem = []
        self.mask_mem = []
        self.probability_mem = []
        self.batch_reward_mem = []

    def store_transition(self, value, state_stack, action, reward, done, probability):
        self.value_mem.append(value)
        self.state_mem.append(state_stack)
        self.action_mem.append(action)
        self.batch_reward_mem.append(reward)
        self.mask_mem.append(0 if done == True else 1)
        self.probability_mem.append(probability)

    def learn(self, is_finish_reached):
        self.T += 1

        # add a small bonus on top when goal is reached
        if is_finish_reached:
            bonus = 200.0 / len(self.batch_reward_mem) 
            self.batch_reward_mem = np.array(self.batch_reward_mem)
            self.batch_reward_mem[np.where(self.batch_reward_mem > 0)] += bonus
   
        self.reward_mem.extend(self.batch_reward_mem)
        self.batch_reward_mem = []
        
        # don't train if batch size is not reached yet
        # if (len(self.state_mem) < BATCH_SIZE):
        #     return

        if (self.T < 3):
            return

        # the value of this state is not yet added to the value memory
        next_state_value = self.value_mem[-1]
        state_mem = np.array(self.state_mem)
        value_mem = np.array(self.value_mem)
        action_mem = np.array(self.action_mem)
        probability_mem = np.array(self.probability_mem)

        gaes = get_generalized_advantage_estimations(self.reward_mem, value_mem, self.mask_mem, next_state_value)
        advantages = gaes - value_mem
        advantages = normalize(advantages)
  
        self.train_model(state_mem, advantages, action_mem, probability_mem, gaes)
        self.reset()

    def get_steps_count(self):
        return len(self.action_mem)

    def get_reward_sum(self):
        return sum(self.batch_reward_mem)

    def get_current_action(self):
        if (len(self.action_mem) < 1):
            return [0.0]

        action_idx = self.action_mem[-1]

        # left steering
        if action_idx == 0:
            return [-0.8]
        # right steering
        if action_idx == 2:
            return [0.8]
        # center steering
        return [0.0]