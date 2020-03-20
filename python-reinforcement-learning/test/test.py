import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input


FRAMES = 2
VIDEO_FPS = 15

def get_frame_buffer(all_frames, frame_idx):
    result = []
    for i in range(1, FRAMES):
        if i == FRAMES-1 and frame_idx > 0 :
            result.append(all_frames[frame_idx-i])
        else:
            result.append(all_frames[i])
    
    result.append(all_frames[i])       
    return result

model = load_model('model.h5')
data = pd.read_csv('test/test-images/data-set.csv')
image_frames = []
X = []
y = []

for index, row in data.iterrows():
    img = Image.open('test/test-images/' + row['file path'])
    img = img.resize((120, 50), Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img_expanded = np.expand_dims(np.array(img), axis=2)
    img_preprocessed = preprocess_input(img_expanded)

    image_frames.append(img_expanded)
    X.append(img_preprocessed)


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('test-video.avi', fourcc, VIDEO_FPS, (120, 50), False)

angle_predictions = [] 

# make the predictions for each frame
for frame_idx in range(len(image_frames)):
    frame_buffer_for_prediction = get_frame_buffer(X, frame_idx)
    frame_buffer_for_prediction = np.array(frame_buffer_for_prediction)[np.newaxis, :]
    prediction = model.predict(frame_buffer_for_prediction)

    angle_predictions.append(prediction[0][0])
  
angle_predictions = np.array(angle_predictions)
print('Angle mean value:', angle_predictions.mean(), 'std:', angle_predictions.std())

# adjust the angle mean to be 0
angle_predictions = angle_predictions - angle_predictions.mean()
# scale the angle predictiions
angle_predictions /= angle_predictions.std() 
angle_predictions *= 60

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

