import os
import gym
import pylab
import time
import itertools
import numpy as np
from statistics import mean

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Flatten, MaxPooling2D, Conv2D, Dropout

from keras import backend as K

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


GAMMA = 0.99
LR_ACTOR = 0.001
LR_CRITIC = 0.005

EPOCHS = 4

CLIPPING = 0.1
ENTROPY_BETA = 0.001
# smoothing factor
ALPHA = 0.95

HEIGHT = 50
WIDTH = 120
DEPTH = 2

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


class Buffer:
    def __init__(self, action_size):
        self.action_size = action_size
        self.clear()

    def length(self):
        return len(self.actions)

    def clear(self):
        self.values = np.array([])
        self.actions = np.array([], dtype='int')
        self.probabilities = np.array([])

    def append(self, value, action, probability):
        self.values = np.append(self.values, value)
        self.actions = np.append(self.actions, action)
        if len(self.probabilities) > 0:
            self.probabilities = np.append(self.probabilities, np.array(probability)[np.newaxis, ::], axis=0)
        else:
            self.probabilities = np.array(probability)[np.newaxis, ::]

    def get_advantages(self, rewards, masks):
        gaes = get_generalized_advantage_estimations(rewards, self.values, masks, self.values[-1])
        advantages = gaes - self.values
        return normalize(advantages), gaes

    def one_hot_ecode_actions(self):
        length = len(self.actions)
        result = np.zeros((length, self.action_size))
        result[range(length), self.actions] = 1
        return result
    


class PPOAgent:
    def __init__(self, state_size=INPUT_SHAPE, gamma=GAMMA, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, load_model_from_file=False):
        self.load_model_from_file = load_model_from_file

        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.state_size = state_size
        self.angle_action_size = 3
        self.throttle_action_size = 2
        self.value_size = 1
        
        self.training_loss = []
        
        self.actor, self.actor_predict = self.build_actor()
        self.critic = self.build_critic()
        self.reset()

        if self.load_model_from_file:
            self.actor.load_weights('model/ppo_actor.h5')
            self.critic.load_weights('model/ppo_critic.h5')
     
    def build_actor(self):
        angle_advantage = Input(shape=(1,), name='angle_advantage')
        angle_old_prediction = Input(shape=(self.angle_action_size,), name='angle_old_prediction')        
        angle_loss = ppo_loss_function(advantage=angle_advantage, old_prediction=angle_old_prediction)

        throttle_advantage = Input(shape=(1,), name='throttle_advantage')
        throttle_old_prediction = Input(shape=(self.throttle_action_size,), name='throttle_old_prediction')        
        throttle_loss = ppo_loss_function(advantage=throttle_advantage, old_prediction=throttle_old_prediction)

        state_input = Input(shape=self.state_size, name='state_input')
        serial = state_input
        serial = Conv2D(16, (3, 3), data_format='channels_last')(serial)
        serial = Conv2D(32, (5, 5), strides=2)(serial)
        serial = MaxPooling2D(pool_size=(3,3), padding='valid')(serial)
        serial = Flatten()(serial)
        serial = Dense(64, activation='relu', kernel_initializer='he_uniform')(serial)
        output_angle = Dense(self.angle_action_size, activation='softmax', kernel_initializer='he_uniform', name='angle')(serial)
        output_throttle = Dense(self.throttle_action_size, activation='softmax', kernel_initializer='he_uniform', name='throttle')(serial)
        
        actor = Model(inputs=[state_input, angle_advantage, angle_old_prediction, throttle_advantage, throttle_old_prediction], outputs=[output_angle, output_throttle])
        actor.compile(loss=[angle_loss, throttle_loss], optimizer=Adam(lr=self.lr_actor))
        actor.summary()

        # no need to compile it because we will not do a back propagation to it
        actor_predict = Model(inputs=[state_input], outputs=[output_angle, output_throttle])

        return actor, actor_predict

    def build_critic(self):
        state_input = Input(shape=self.state_size, name='state_input')
        serial = state_input
        serial = Conv2D(16, (3, 3), data_format='channels_last')(serial)
        serial = Conv2D(32, (5, 5), strides=2)(serial)
        serial = MaxPooling2D(pool_size=(3,3), padding='valid')(serial)
        serial = Flatten()(serial)
        serial = Dense(64, activation='relu', kernel_initializer='he_uniform')(serial)
        output_angle = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform', name='angle')(serial)
        output_throttle = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform', name='throttle')(serial)

        critic = Model(inputs=[state_input], outputs=[output_angle, output_throttle])
        critic.compile(loss=['mse', 'mse'], optimizer=Adam(lr=self.lr_actor))
        # critic.summary()
        return critic
    
    def get_value(self, state):
        state_expanded = np.array(state)[np.newaxis, ::]
        prediction = self.critic.predict(state_expanded)
        return [prediction[0][0][0], prediction[1][0][0]]

    def get_action(self, state):
        state_expanded = np.array(state)[np.newaxis, ::]
        probability = self.actor_predict.predict([state_expanded])

        angle_prob = probability[0][0]
        throttle_prob = probability[1][0]

        angle_action_idx = np.random.choice(self.angle_action_size, 1, p=angle_prob)[0]
        throttle_action_idx = np.random.choice(self.throttle_action_size, 1, p=throttle_prob)[0]

        return [angle_action_idx, throttle_action_idx], [angle_prob,throttle_prob]

    def train_model(self, states):
        batch_size = int((len(states) * 0.6) / EPOCHS)

        angle_adv, angle_gaes = self.angle_buffer.get_advantages(self.reward_mem, self.mask_mem)
        throttle_adv, throttle_gaes = self.throttle_buffer.get_advantages(self.reward_mem, self.mask_mem)

        actor_loss = self.actor.fit(
            {
                'state_input': states,
                'angle_advantage': angle_adv, 'angle_old_prediction': self.angle_buffer.probabilities,
                'throttle_advantage': throttle_adv, 'throttle_old_prediction': self.throttle_buffer.probabilities,
            },
            {
                'angle': self.angle_buffer.one_hot_ecode_actions(),
                'throttle': self.throttle_buffer.one_hot_ecode_actions()
            },
            verbose=False, shuffle=True, epochs=EPOCHS, batch_size=batch_size, validation_split=0.2)
        
        critic_loss = self.critic.fit(
            [states],
            {
                'angle': angle_gaes,
                'throttle': throttle_gaes
            },
            verbose=False, shuffle=True, epochs=EPOCHS, batch_size=batch_size, validation_split=0.2)
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
        self.last_throttle = 1.0
        self.state_mem = []
        self.reward_mem = np.array([])
        self.mask_mem = np.array([])
        self.batch_reward_mem = np.array([])
        self.angle_buffer = Buffer(self.angle_action_size)
        self.throttle_buffer = Buffer(self.throttle_action_size)

    def store_transition(self, value, state, action, reward, done, probability):
        self.state_mem.append(state)
        self.batch_reward_mem = np.append(self.batch_reward_mem, reward)
        self.mask_mem = np.append(self.mask_mem, 0 if done == True else 1)
        self.angle_buffer.append(value[0], action[0], probability[0])
        self.throttle_buffer.append(value[1], action[1], probability[1])

    def learn(self, is_finish_reached):
        self.T += 1

        # add a small bonus on top when goal is reached
        if is_finish_reached:
            bonus = 200.0 / len(self.batch_reward_mem) 
            self.batch_reward_mem = np.array(self.batch_reward_mem)
            self.batch_reward_mem[np.where(self.batch_reward_mem > 0.95)] += bonus
   
        self.reward_mem = np.append(self.reward_mem, self.batch_reward_mem)
        self.batch_reward_mem = np.array([])

        if (self.T < 3):
            return
  
        self.train_model(np.array(self.state_mem))
        self.reset()

    def get_steps_count(self):
        return self.angle_buffer.length()

    def get_reward_sum(self):
        return sum(self.batch_reward_mem)

    def get_previous_throttle(self):
        return self.last_throttle

    def get_current_action(self):
        if (self.angle_buffer.length() < 1):
            return [0.0, 1.0]

        angle_action_idx = self.angle_buffer.actions[-1]
        throttle_action_idx = self.throttle_buffer.actions[-1]

        angle = 0.0
        throttle = 0.2

        # left steering
        if angle_action_idx == 0:
            angle = -0.8

        # right steering
        if angle_action_idx == 2:
            angle = 0.8

        # full speed
        if throttle_action_idx == 0:
            throttle = 1.0

        self.last_throttle = throttle
        return [angle, throttle]