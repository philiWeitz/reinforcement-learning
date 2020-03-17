import numpy as np

from visualization import Visualization
from agent_policy_gradient import AgentPolicyGradient

from tensorflow.keras.applications.mobilenet import preprocess_input


def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def preprocess_image(image):
    return preprocess_input(image)


def action_to_motion(action):
    clipped_action = np.clip(action, -1, 1)

    motion = {}
    motion['steering'] = round(float(clipped_action[0]), 3)
    motion['acceleration'] = 1.0

    return motion


class Environment:
    def __init__(self):
        self.agent = AgentPolicyGradient()
        self.visualization = Visualization()
        self.is_terminal_state = False


    def add_movement(self, move_model):
        self.is_terminal_state = move_model['isTerminalState']

        # get the observation
        colors = move_model['colors']
        gray_scale_image = np.reshape(colors, (50, 120))
        gray_scale_image = np.flip(gray_scale_image, 0)

        # show the input image
        # self.visualization.show_agent_input_image(gray_scale_image)

        gray_scale_image = expand_image_dimension(gray_scale_image)
        gray_scale_image = preprocess_image(gray_scale_image)

        # predict action and store current observation
        action = self.agent.choose_action(gray_scale_image)

        reward = 1 if action[0] > -0.1 and action[0] < 0.1 else 0.4
        reward = reward if move_model['isOnTrack'] else reward * -1
        
        self.agent.store_transaction(gray_scale_image, action, reward)


    def train_model_on_batch(self):
        loss_value = self.agent.learn()
        step_count = self.agent.get_steps_count()

        # self.visualization.add_loss_value(loss_value)
        # self.visualization.plot_loss_history()

        # self.visualization.add_steps_value(step_count)
        # self.visualization.plot_steps_history()

        self.agent.reset()


    def get_predicted_motion(self):
        return action_to_motion(self.agent.get_current_action())
