import numpy as np

from visualization import Visualization

from agent_temporal_difference import AgentTemporalDifference
# from agent_policy_gradient_multi_frame import AgentPolicyGradient

from tensorflow.keras.applications.mobilenet import preprocess_input


def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def preprocess_image(image):
    return (image - image.mean()) / 255


def action_to_motion(action):
    clipped_steering = np.clip(action[0], -1, 1)

    motion = {}
    motion['steering'] = round(float(clipped_steering), 3)
    motion['acceleration'] = 1.0

    return motion


def is_same_direction(previous_action, action):
    prev = np.argmax(previous_action)
    curr = np.argmax(action)
    return prev == curr
    


class Environment:
    def __init__(self):
        self.agent = AgentTemporalDifference()
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

        # predict action and store current observation
        reward = self.agent.get_reward(is_on_track, self.is_finish_reached)
        
        self.agent.store_transaction(gray_scale_image, reward)


    def train_model_on_batch(self):
        # if we reached the goal -> current model is already really good (save before retraining)
        if self.is_finish_reached:
            self.agent.save_prediction_model()

        loss_value = self.agent.learn(self.is_finish_reached)
        self.visualization.add_loss_value(loss_value)
        self.visualization.plot_loss_history()

        # write video to file if finish is reached
        if self.is_finish_reached:
            self.visualization.frames_to_file()

        # step_count = self.agent.get_steps_count()
        # self.visualization.plot_steering(self.agent.action_memory)

        # self.visualization.add_steps_value(step_count)
        # self.visualization.plot_steps_history()

        self.visualization.reset_image_buffer()
        self.agent.reset()


    def get_predicted_motion(self):
        return action_to_motion(self.agent.get_current_action())
