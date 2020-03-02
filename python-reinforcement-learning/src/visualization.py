import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualization:

    def __init__(self):
      self.loss_history = np.array([])
      self.steps_history = np.array([])


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