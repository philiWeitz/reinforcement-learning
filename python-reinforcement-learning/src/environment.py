import numpy as np
from visualization import Visualization
from agent import PPOAgent
# from agent_with_throttle import PPOAgent

def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def preprocess_image(image):
    return image / 255


def action_to_motion(action):
    clipped_steering = np.clip(action[0], -1.0, 1.0)
    clipped_throttle = np.clip(action[1], 0, 1.0)

    motion = {}
    motion['steering'] = round(float(clipped_steering), 3)
    motion['acceleration'] = round(float(clipped_throttle), 3)

    return motion


class Environment:
    def __init__(self):
        self.agent = PPOAgent(load_model_from_file=False)
        self.visualization = Visualization()
        self.is_terminal_state = False


    def add_movement(self, move_model):
        is_on_track = move_model['isOnTrack']
        self.is_terminal_state = move_model['isTerminalState']
        self.is_finish_reached = move_model['isFinishReached']

        # get the observation
        colors = move_model['colors']
        gray_scale_image = np.reshape(colors, (50, 120))
        gray_scale_image = np.flip(gray_scale_image, 0)
        self.visualization.add_image(gray_scale_image)

        # show the input image
        # self.visualization.show_agent_input_image(gray_scale_image)

        gray_scale_image = expand_image_dimension(gray_scale_image)
        gray_scale_image = preprocess_image(gray_scale_image)

        # this will hopefully give the agent an understanding of speed
        prev_throttle = self.agent.get_previous_throttle()
        previous_throttle_state = np.full_like(gray_scale_image, prev_throttle / 10.0)

        state = np.append(gray_scale_image, previous_throttle_state, axis=2)
        value = self.agent.get_value(state)
        action, probability = self.agent.get_action(state)
        done = self.is_terminal_state

        # reward = 1.0 if action == 1 else 0.9
        # reward = 1.0 if action[1] == 0 else 0.5
        # reward = reward + 0.2 if action[0] == 1 else reward
        reward = 1.0 if is_on_track else -0.1

        self.agent.store_transition(value, state, action, reward, done, probability)


    def train_model_on_batch(self):
        # step_count = self.agent.get_steps_count()
        # self.visualization.add_steps_value(step_count)
        # self.visualization.plot_steps_history()
        # self.visualization.show_random_agent_input_image()

        self.visualization.add_reward_value(self.agent.get_reward_sum())
        self.visualization.plot_reward_history()

        # if we reached the goal -> current model is already really good (save before retraining)
        if self.is_finish_reached:
            print('Saving model to file...')
            self.agent.save_models()

        self.agent.learn(self.is_finish_reached)

        # write video to file if finish is reached
        if self.is_finish_reached:
            print('Saving video to file...')
            self.visualization.frames_to_file()

        self.visualization.reset_image_buffer()


    def get_predicted_motion(self):
        return action_to_motion(self.agent.get_current_action())
