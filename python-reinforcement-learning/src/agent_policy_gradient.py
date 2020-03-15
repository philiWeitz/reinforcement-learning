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


def get_terminal_index_of_longest_path(terminals):
    indexs = np.where(terminals == True)[0] + 1
    result = np.split(terminals, indexs)

    path_length = [len(x) for x in result]
    longest_path_idx = np.argmax(path_length)

    start_idx = sum(path_length[:longest_path_idx])
    end_idx = sum(path_length[:longest_path_idx+1]) - 1
    
    return start_idx, end_idx


class AgentPolicyGradient():
    def __init__(self):
        self.GAMMA=0.99
        self.G = 0
        self.lr = 0.0001
        self.policy, self.predict = self.build_policy_network()


    def build_policy_network(self):
        input = Input(shape=(50, 120, 1), name='img_in')
        advantages = Input(shape=[1])
        
        x = input
        x = Conv2D(16, (3, 3), data_format='channels_last')(x)
        x = Conv2D(32, (5, 5), strides=2)(x)
        x = MaxPooling2D(pool_size=(3,3), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(100, activation='relu')(x)
        props = Dense(OUTPUT_SHAPE, activation='softmax')(x)

        def custom_loss_function(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * advantages)

        policy = Model(inputs=[input, advantages], outputs=[props])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss_function)

        predict = Model(inputs=[input], outputs=[props])
        return policy, predict

    
    def predict_move(self, observation):
        state = observation[np.newaxis, :]
        propabilities = self.predict.predict(state)[0]
        # print(propabilities)
        action = np.random.choice(NR_OF_ACTIONS, p=propabilities)
        # action = np.argmax(propabilities)
        return propabilities, action


    def train(self, memory):
        state_memory = np.array(memory.observations)
        state_memory_mirrowed = np.array(memory.observations_mirrowed)
        action_memory = np.array(memory.actions)
        reward_memory = np.array(memory.rewards)
        terminals = np.array(memory.terminals)
        selected_idxs = memory.selected_action_idx

        # only keep the longest route for training
        start_idx, end_idx = get_terminal_index_of_longest_path(terminals)
        state_memory = state_memory[start_idx:end_idx+1]
        state_memory_mirrowed = state_memory_mirrowed[start_idx:end_idx+1]
        action_memory = action_memory[start_idx:end_idx+1]
        reward_memory = reward_memory[start_idx:end_idx+1]
        terminals = terminals[start_idx:end_idx+1]
        selected_idxs = selected_idxs[start_idx:end_idx+1]

        # # add an extra reward to the longest route
        # start_idx, end_idx = get_terminal_index_of_longest_path(terminals)
        # reward_memory[end_idx] = 10

        actions = np.zeros([len(action_memory), NR_OF_ACTIONS])
        # one hot encoding
        for i in range(len(actions)):
            selected_idx_for_action = selected_idxs[i]
            actions[i][selected_idx_for_action] = 1.0

        running_add = 0.0
        G = np.zeros_like(reward_memory, dtype='float')

        for i in reversed(range(len(reward_memory))):
            if terminals[i] == True:
                running_add = reward_memory[i]
            else:
                # belman equation
                running_add = reward_memory[i] + running_add * self.GAMMA
            G[i] = running_add

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean) / std

        # TODO: is this worth it?
        # G += 0.6

        # mirrow the actions
        actions_mirrowed = actions.copy()
        actions_mirrowed[:,[0, 1]] = actions_mirrowed[:,[1, 0]]
        actions = np.append(actions,actions_mirrowed, axis=0)

        # combine observations with mirrow observations
        state_memory = np.append(state_memory,state_memory_mirrowed, axis=0)

        # double G
        G = np.append(G,G, axis=0)

        cost = self.policy.train_on_batch([state_memory, G], actions)
        print("Cost:", cost)
        return cost