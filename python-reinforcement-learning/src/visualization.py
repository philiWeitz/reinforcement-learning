import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualization:

    def __init__(self):
      self.loss_history = np.array([])

    def add_loss_value(self, loss_value):
      self.loss_history = np.append(self.loss_history, loss_value)

    def plot_loss_history(self):
      plt.figure(0)
      ax = sns.lineplot(data=self.loss_history)
      ax.set_title('Loss History')
      plt.show(block=False)
      plt.pause(0.001)