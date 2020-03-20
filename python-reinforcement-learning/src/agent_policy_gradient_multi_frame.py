import numpy as np
import keras.backend as K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization, TimeDistributed, LSTM


GAMMA = 0.99
LR = 0.05

NR_OF_ACTIONS = 2
STEERING_IDX = 0

FRAMES = 2

def moving_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    result = (cumsum[N:] - cumsum[:-N]) / float(N)
    return np.append(x[0:N-1].flatten(), result)


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
        cnnModel = Sequential()
        cnnModel.add(Conv2D(16, (3, 3), data_format='channels_last'))
        cnnModel.add(Conv2D(32, (5, 5), strides=2))
        cnnModel.add(MaxPooling2D(pool_size=(3,3), padding='valid'))
        cnnModel.add(Dense(200, activation='linear'))
        cnnModel.add(Dense(100, activation='linear'))
        cnnModel.add(Flatten())

        # advantages = Input(shape=[1])

        inputLayer = Input(shape=(FRAMES, 50, 120, 1), name='input_layer')
        serial = inputLayer
        serial = TimeDistributed(cnnModel, input_shape=(50, 120, 1))(serial)
        serial = Dense(200, activation='linear')(serial)
        serial = Flatten()(serial)

        angle_out = Dense(units=1, activation='linear', name='angle_out')(serial)

        policy = Model(inputs=[inputLayer], outputs=[angle_out])
        policy.compile(optimizer='adam', loss={'angle_out': 'mean_absolute_percentage_error'})
        
        predict = Model(inputs=[inputLayer], outputs=[angle_out])
        return policy, predict

    
    def choose_action(self, observation):
        state = np.array(self.get_observation_buffer(observation))
        state = state[np.newaxis, :]
        prediction = self.predict.predict(state)
        return prediction[STEERING_IDX]


    def store_transaction(self, observation, action, reward):
        state = self.get_observation_buffer(observation)
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def get_observation_buffer(self, current_observation):
        result = []
        for i in range(1, FRAMES):
            if len(self.state_memory) > 0:
                result.append(self.state_memory[-i][-1])
            else:
                result.append(current_observation)
        result.append(current_observation)
        return result


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


    def learn(self, is_finish_reached=False):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)

        if (len(action_memory) < 2):
            print("Invalid terminal state")
            return 0

        print("Observations for training:", len(state_memory))
        steering_actions = np.clip(action_memory, -0.8, 0.8)
        steering_actions_smooth = moving_average(steering_actions, 3)
        steering_actions_gradient_sum = sum(np.gradient(steering_actions_smooth))

        advantages = self.get_discounted_rewards()
        training_frame_idxs = np.where(advantages <= 0)

        # only adjust actions if it has negative examples
        if not is_finish_reached:
            steering_action_changes = abs(advantages) * steering_actions_gradient_sum * LR
            for action_idx in range(len(steering_actions)):
                steering_actions[action_idx][0] += steering_action_changes[action_idx]
        else:
            training_frame_idxs = np.arange(len(advantages))
            steering_actions = action_memory

        # network parameter
        X = np.array(state_memory[training_frame_idxs])
        angle = steering_actions[training_frame_idxs]
    
        # shuffling seems to be very important!
        result = self.policy.fit(X, {
            'angle_out': angle
        }, batch_size=512, epochs=1, shuffle=True, verbose=0)
        
        return result.history["loss"][-1]


    def get_steps_count(self):
        return len(self.action_memory)


    def get_current_action(self):
        return self.action_memory[-1]

    
    def save_prediction_model(self):
        self.predict.save('model.h5')