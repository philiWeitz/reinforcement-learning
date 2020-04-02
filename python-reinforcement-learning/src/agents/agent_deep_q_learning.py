import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Dropout, BatchNormalization, TimeDistributed, LSTM


LR = 0.00001
GAMMA = 0.99

EPSILON = 1.00
EPSILON_MIN = 0.01
EPSILON_DECREASE = 0.95

NR_OF_ACTIONS = 3
TRAINING_BATCH_SIZE = 16
MAX_MEMORY_SIZE = 1000

HEIGHT = 50
WIDTH = 120
DEPTH = 1
INPUT_SHAPE = (HEIGHT, WIDTH, DEPTH)


class Agent():
    def __init__(self):
        self.reset()
        self.T = 0
        self.epsilon = EPSILON
        self.memory = ReplayBuffer(MAX_MEMORY_SIZE, INPUT_SHAPE, NR_OF_ACTIONS)
        self.model = self.build_model()
        # self.A = self.build_model()
        # self.B = self.build_model()
        # self.mix_models()


    def reset(self):
        self.is_terminal_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.state_memory = []
        self.reward_sum = 0

    
    # def mix_models(self):
        # if np.random.choice(2) == 0:
        #     self.model = self.A
        #     self.q_next_model = self.B
        # else:
        #     self.model = self.B
        #     self.q_next_model = self.A


    def build_model(self):
        input = Input(shape=INPUT_SHAPE, name='img_in')
               
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)

        x = Dense(1024, activation='linear')(x)
        steering = Dense(NR_OF_ACTIONS, activation='linear')(x)

        model = Model(inputs=[input], outputs=[steering])
        model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
        model.summary()
        return model

    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        if np.random.random() < self.epsilon:
            action = np.random.choice(NR_OF_ACTIONS)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions[0])
        return action


    def store_transaction(self, observation, is_on_track, is_terminal_state):
        action = self.choose_action(observation)
        reward = self.get_reward(is_on_track, is_terminal_state)

        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.state_memory.append(observation)
        self.is_terminal_state_memory.append(is_terminal_state)


    def get_reward(self, is_on_track, is_terminal_state):
        reward = 1 if is_on_track else -1
        self.reward_sum += reward
        return reward


    def learn(self, is_finish_reached=False):
        # TODO: adjust the rewards of the current episode

        # put all learnings into the buffer
        for idx in range(len(self.state_memory) - 1):
            state = self.state_memory[idx]
            next_state = self.state_memory[idx+1]
            action = self.action_memory[idx]
            reward = self.reward_memory[idx]
            self.memory.store_transition(state, action, reward, next_state, False)

        # don't train if buffer is too small
        if self.memory.mem_cntr < TRAINING_BATCH_SIZE:
            self.reset()
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(TRAINING_BATCH_SIZE)
        action_space = [i for i in range(NR_OF_ACTIONS)]
        action_values = np.array(action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_current = self.model.predict(state)
        q_next = self.model.predict(new_state)
        q_target = q_current.copy()

        # update the target actions
        batch_index = np.arange(TRAINING_BATCH_SIZE, dtype=np.int32)
        # q_target[batch_index, action_indices] = reward + GAMMA * np.max(q_next, axis=1) * done
        q_target[batch_index, action_indices] = reward + GAMMA * q_next[batch_index, action_indices] * done
        
        result = self.model.fit(state, q_target, shuffle=True, batch_size=TRAINING_BATCH_SIZE, epochs=1, verbose=0)
        self.T += 1

        # some stats
        print('Round:', self.T, ' - Total rewards:', round(self.reward_sum, 2), ', Epsilon:', round(self.epsilon, 2), ', Total actions:', len(self.reward_memory))

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECREASE)
        self.reset()
        return result.history["loss"][-1]


    def get_current_action(self):
        result = [0.0, 0.0]

        if len(self.action_memory) <= 0:
            return result
        
        action = self.action_memory[-1]

        # left steering
        if action == 0:
            result[0] = -0.8
        # right steering
        if action == 2:
            result[0] = 0.8
        # center steering
        return result
    
    
    def save_model(self):
        self.model.save('model-dq.h5')
