import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time


AGENT_IMAGE_TIMEOUT_SEC = 3 * 1000


class Visualization:

    def __init__(self):
      self.loss_history = np.array([])
      self.steps_history = np.array([])
      self.image_show_timeout = time.time() * 1000.0


    def add_loss_value(self, loss_value):
      self.loss_history = np.append(self.loss_history, loss_value)


    def add_steps_value(self, steps_value):
      self.steps_history = np.append(self.steps_history, steps_value)


    def plot_loss_history(self):
      plt.figure(0)
      ax = sns.lineplot(data=self.loss_history)
      ax.set_title('Loss History')
      plt.show(block=False)
      plt.pause(0.001)


    def plot_steps_history(self):
      plt.figure(1)
      ax = sns.lineplot(data=self.steps_history)
      ax.set_title('Steps History')
      plt.show(block=False)
      plt.pause(0.001)

    
    def show_agent_input_image(self, image):
      now = time.time() * 1000.0

      if now > self.image_show_timeout:
        plt.figure(2)
        plt.title('Agent Input Image')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)
        plt.pause(0.001)
        self.image_show_timeout = now + AGENT_IMAGE_TIMEOUT_SEC
