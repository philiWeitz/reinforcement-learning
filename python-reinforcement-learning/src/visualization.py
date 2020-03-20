import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import cv2


AGENT_IMAGE_TIMEOUT_SEC = 1 * 1000


class Visualization:

    def __init__(self):
      self.loss_history = np.array([])
      self.steps_history = np.array([])
      self.image_show_timeout = time.time() * 1000.0
      self.image_buffer = []


    def reset_image_buffer(self):
      self.image_buffer = []


    def add_image(self, image):
      self.image_buffer.append(image)


    def add_loss_value(self, loss_value):
      self.loss_history = np.append(self.loss_history, loss_value)


    def add_steps_value(self, steps_value):
      self.steps_history = np.append(self.steps_history, steps_value)


    def plot_loss_history(self):
      plt.figure(0)
      plt.cla()
      ax = sns.lineplot(data=self.loss_history[-50:])
      ax.set_title('Loss History')
      plt.show(block=False)
      plt.pause(0.001)


    def plot_steps_history(self):
      plt.figure(1)
      ax = sns.lineplot(data=self.steps_history)
      ax.set_title('Steps History')
      plt.show(block=False)
      plt.pause(0.001)


    def plot_steering(self, actions):
      data = np.array(actions).flatten()
      plt.figure(2)
      plt.cla()
      ax = sns.lineplot(data=data)
      ax.set_title('Steering')
      plt.show(block=False)
      plt.pause(0.001)

    
    def show_agent_input_image(self, image):
      now = time.time() * 1000.0

      if now > self.image_show_timeout:
        plt.figure(3)
        plt.title('Agent Input Image')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)
        plt.pause(0.001)
        self.image_show_timeout = now + AGENT_IMAGE_TIMEOUT_SEC

    def frames_to_file(self):
      fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
      fps = 24
      out = cv2.VideoWriter('track-video.avi', fourcc, fps, (120, 50), False)
      
      for frame in self.image_buffer:
        frame_gray_scale = np.expand_dims(frame, axis=2)
        frame_gray_scale = np.array(frame_gray_scale, dtype=np.uint8)
        out.write(frame_gray_scale)
      
      out.release()
      self.image_buffer = []