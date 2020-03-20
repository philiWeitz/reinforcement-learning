import gym
import matplotlib.pyplot as plt
import numpy as np

from agent_policy_gradient import AgentPolicyGradient

if __name__ == '__main__':
    agent = AgentPolicyGradient()

    env = gym.make('LunarLander-v2')
    score_history = []

    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            # env.render()
            observation_, reward, done, info = env.step(action)
            agent.store_transaction(observation, action, reward)
            observation = observation_
            score += reward

        score_history.append(score)
        agent.learn()

        print('Episode ', i, 'score %.1f' % score, 'average score %.1f' % np.mean(score_history[-100:]))