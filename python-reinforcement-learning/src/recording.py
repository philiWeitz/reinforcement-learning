import numpy as np
import os, os.path
import json

from PIL import Image
from socket_connection import SocketConnector
from agent_temporal_difference import AgentTemporalDifference

RECORDING_FOLDER =  './recordings/'

if not os.path.exists(RECORDING_FOLDER):
    os.mkdir(RECORDING_FOLDER)


user_input_memory = []
frame_buffer_memory = []


def frame_recording_to_file(frame_buffer, user_input):
    file_count = str(len(os.listdir(RECORDING_FOLDER)))
    folder_path = RECORDING_FOLDER + file_count

    # create the folder
    os.mkdir(folder_path)

    # save user input
    np.savetxt(folder_path + '/user-input.txt', user_input, delimiter=';', header='steering;acceleration')


    # save all frames
    for frame_idx in range(len(frame_buffer)):
        gray_scale_image = np.reshape(frame_buffer[frame_idx], (50, 120))
        gray_scale_image = np.flip(gray_scale_image, 0)

        img_file_name = str(frame_idx).zfill(5)
        img = Image.fromarray(gray_scale_image.astype(np.uint8))
        img.save(folder_path + '/' + img_file_name + '.jpg')


def on_data_received(json_object, connection):
    global user_input_memory
    global frame_buffer_memory

    image = json_object['colors']
    user_input = json_object['userInput']
    is_terminal_state = json_object['isTerminalState']

    frame_buffer_memory.append(image)      
    user_input_memory.append(user_input)             

    # no need to move further
    if is_terminal_state:
        frame_recording_to_file(frame_buffer_memory, user_input_memory)
        frame_buffer_memory = []
        user_input_memory = []

        print('Recording saved')
        connection.send("RESET".encode('utf-8'))
    else:
        connection.send(json.dumps({ 'steering': 0.0, 'acceleration': 0.0 }).encode('utf-8'))


def create_a_recording():
    socketConnector = SocketConnector('localhost', 11000)
    socketConnector.open_connection(on_data_received)


def create_a_base_model(agent):
    for folder in os.listdir(RECORDING_FOLDER):
        folder_path = RECORDING_FOLDER + folder + '/'
        files = os.listdir(folder_path)
        files.sort()

        X = []
        y = np.loadtxt(folder_path + 'user-input.txt', delimiter=';', skiprows=1)

        image_files = [ file for file in files if file.endswith('.jpg') ]
        for image_path in image_files:
            img = Image.open(folder_path + image_path)
            img_expanded = np.expand_dims(np.array(img), axis=2)
            X.append(img_expanded)

        # train the agent
        agent.learn_on_recording(np.array(X), np.array(y))
    
    # save the model
    agent.save_prediction_model


create_a_recording()

# agent = AgentTemporalDifference()
# create_a_base_model(agent)