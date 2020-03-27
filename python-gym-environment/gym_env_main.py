import gym
import matplotlib.pyplot as plt
import numpy as np

from agent_policy_gradient import AgentPolicyGradient
from agent_q_learning import AgentTemporalDifference


def is_terminal_state(reward_history):
    return True if max(reward_history[-20:]) < 0 else False


if __name__ == '__main__':
    agent = AgentTemporalDifference()
 
    env = gym.make('CarRacing-v0')
    score_history = []

    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        total_score = 0
        observation = env.reset()

        while not done:
            action = agent.get_current_action()
            # env.render()
            
            if (i % 20) == 19:
                env.render()

            observation, reward, done, info = env.step(action)
            agent.store_transaction(observation, reward)
        
            total_score += reward
            done = is_terminal_state(agent.reward_memory)

        score_history.append(total_score)
        agent.learn(False)

        if (i % 20) == 19:
            # env.render()
            agent.save_prediction_model()

        print('Episode ', i, 'score %.1f' % total_score, 'average score %.1f' % np.mean(score_history[-100:]))