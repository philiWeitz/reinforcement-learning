import sys
import gym
import pylab
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model

import tensorflow as tf
# This is important! It speeds up the training by a lot
tf.compat.v1.disable_eager_execution()

# reference: https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/4-actor-critic

GAMMA = 0.99
EPISODES = 1000
LR_ACTOR = 0.001
LR_CRITIC = 0.005


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model_from_file = False

        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        if self.load_model_from_file:
            self.actor = load_model('./logs/cartpole_actor.h5')
            self.critic = load_model('./logs/cartpole_critic.h5')
        else:
            self.actor = self.build_actor()
            self.critic = self.build_critic()

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR_ACTOR))
        actor.summary()
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(lr=LR_CRITIC))
        critic.summary()
        return critic

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward - value + GAMMA * (next_value)
            target[0][0] = reward + GAMMA * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def save_models(self):
        self.actor.save("./logs/cartpole_actor.h5")
        self.critic.save("./logs/cartpole_critic.h5")


def plot_reward_history(rewards):
    plt.figure(0)
    plt.cla()
    ax = sns.lineplot(data=np.array(rewards))
    ax.set_title('Reward History')
    plt.show(block=False)
    plt.pause(0.001)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size)

    scores = []

    for episode_idx in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        state_mem = []
        action_mem = []
        reward_mem = []
        next_state_mem = []
        done_mem = []

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            state_mem.append(state)
            action_mem.append(action)
            reward_mem.append(reward)
            next_state_mem.append(next_state)
            done_mem.append(done)

            score += reward
            state = next_state

            if done:
                for i in range(len(state_mem)):
                    agent.train_model(state_mem[i], action_mem[i], reward_mem[i], next_state_mem[i], done_mem[i])

                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)

                print('Episode score:', score)
                plot_reward_history(scores)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save("./logs/cartpole_actor.h5")
                    agent.critic.save("./logs/cartpole_critic.h5")
                    sys.exit()

        # save the model when ever the last 2 runs are close to optimum
        if len(scores) > 2 and np.mean(scores[-2]) > 490:
            agent.save_models()