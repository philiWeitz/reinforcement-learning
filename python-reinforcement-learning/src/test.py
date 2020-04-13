import cv2
import numpy as np
import pandas as pd

from agent import PPOAgent
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

VIDEO_FPS = 15

agent = PPOAgent(load_model_from_file=True)
data = pd.read_csv('test/data-set.csv')

image_frames = []
X = []
y = []

def preprocess_image(image):
    return image / 255

for index, row in data.iterrows():
    img = Image.open('test/' + row['file path'])
    img = img.resize((120, 50), Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img_expanded = np.expand_dims(np.array(img), axis=2)
    img_preprocessed = preprocess_image(img_expanded)

    image_frames.append(img_expanded)
    X.append(img_preprocessed)


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('test-video.avi', fourcc, VIDEO_FPS, (120, 50), False)

angle_predictions = [] 

# make the predictions for each frame
for frame in image_frames:
    frame_expanded = np.array(frame)[np.newaxis, :]
    action_idx, probability = agent.get_action(frame)

    if action_idx == 0:
        angle_predictions.append(-1)
    elif action_idx == 2:
        angle_predictions.append(1)
    else:
        angle_predictions.append(0)
  
angle_predictions = np.array(angle_predictions)
angle_predictions *= 50

# create the video
for frame_idx in range(len(image_frames)):
    frame = image_frames[frame_idx]
    angle = angle_predictions[frame_idx]
    angle_offset = min(110, max(10, int(60 + angle)))

    for a in range(10):
        frame[a][60][0] = 0
        frame[a][10][0] = 0
        frame[a][110][0] = 0
        
    for a in range(10):
        for b in range(6):
            frame[a][angle_offset+b-3][0] = 0

    out.write(frame)

out.release()

