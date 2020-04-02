import numpy as np
from visualization import Visualization
# from agents.agent_q_learning import Agent
from agents.agent_deep_q_learning import Agent
# from agents.agent_policy_gradient_multi_frame import AgentPolicyGradient


def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def preprocess_image(image):
    return image / 255


def action_to_motion(action):
    clipped_steering = np.clip(action[0], -1, 1)

    motion = {}
    motion['steering'] = round(float(clipped_steering), 3)
    motion['acceleration'] = 1.0

    return motion


class Environment:
    def __init__(self):
        self.agent = Agent()
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

        self.agent.store_transaction(gray_scale_image, is_on_track, self.is_terminal_state)


    def train_model_on_batch(self):
        # if we reached the goal -> current model is already really good (save before retraining)
        if self.is_finish_reached:
            print('Saving model to file...')
            self.agent.save_model()

        self.agent.learn(self.is_finish_reached)
        # self.visualization.add_loss_value(loss_value)
        # self.visualization.plot_loss_history()

        # write video to file if finish is reached
        if self.is_finish_reached:
            print('Saving video to file...')
            self.visualization.frames_to_file()

        # step_count = self.agent.get_steps_count()
        # self.visualization.plot_steering(self.agent.action_memory)

        # self.visualization.add_steps_value(step_count)
        # self.visualization.plot_steps_history()

        self.visualization.reset_image_buffer()
        self.agent.reset()


    def get_predicted_motion(self):
        return action_to_motion(self.agent.get_current_action())
