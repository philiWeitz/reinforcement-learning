import numpy as np

from agent import Agent
from visualization import Visualization
from rl.memory import SequentialMemory


def expand_image_dimension(image):
    return np.expand_dims(image, axis=2)


def action_to_motion(action):
    horizontal = np.argmax(action)

    if horizontal == 0:
        horizontal = "LEFT"
    elif horizontal == 1:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"

    motion = {}
    motion['vertical'] = "FORWARD"
    motion['horizontal'] = horizontal

    return motion


def motion_to_action(motion):
    action = [0,0,0]

    if motion['horizontal'] == "LEFT":
        action[0] = 1.0
    elif motion['horizontal'] == "RIGHT":
        action[1] = 1.0
    else:
        action[2] = 1.0

    return action


class Environment:
    def __init__(self):
        self.init()
        self.agent = Agent()
        self.batch_size = 1
        self.is_terminal_state = False
        self.visualization = Visualization()


    def init(self):
        self.memory = SequentialMemory(1000, window_length=1)
        self.current_episode = 0


    def add_move(self, move_model):
        # environment was not propperly reset -> ignore this frame
        if self.is_terminal_state and not move_model['isOnTrack']:
            return False

        # get the observation
        colors = move_model['colors']
        gray_scale_image = np.reshape(colors, (50, 120))
        gray_scale_image = np.flip(gray_scale_image, 0)

        # display the image
        # self.visualization.show_agent_input_image(gray_scale_image)

        gray_scale_image = expand_image_dimension(gray_scale_image)

        # is terminal state
        is_agent_on_track =  move_model['isOnTrack']
        self.is_terminal_state = not is_agent_on_track

        # get the next state prediction from network
        prediction = self.agent.predict_move(gray_scale_image)

        # each step gets a reward of 1
        reward = 1 if self.is_terminal_state else 1

        # lets store the current state
        self.memory.append(gray_scale_image, prediction, reward, self.is_terminal_state)

        # update the episode counter
        if self.is_terminal_state:
            self.current_episode += 1

        return True


    def train_model_on_batch(self):
        # only train if batch size is reached
        if self.current_episode >= self.batch_size:
            loss_value = self.agent.train(self.memory)
            self.visualization.add_loss_value(loss_value)
            self.visualization.plot_loss_history()

            self.visualization.add_steps_value(sum(self.memory.rewards.data) / self.batch_size)
            self.visualization.plot_steps_history()
            # reset environment
            self.init()


    def get_predicted_motion(self):
        return action_to_motion(self.memory.actions.data[-1])
