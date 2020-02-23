import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import socket
import json
import sys

import time
import network

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ('localhost', 11000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

# shows an image every 5 seconds
show_image = False
image_show_timeout = time.time() * 1000.0

episode_counter = 0


def show_received_image(image):
    global image_show_timeout
    now = time.time() * 1000.0

    if (show_image and now > image_show_timeout) == True:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)
        plt.pause(0.001)
        image_show_timeout = now + (3 * 1000)

def run_socket_server():
    global episode_counter

    epsilon = 100
    data_buffer = ""

    reward_record = []
    state_record = []
    action_record = []

    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while connection:
            byte_data = connection.recv(10000)

            if byte_data:
                data = byte_data.decode('utf-8')
                data_buffer += data
                
                # find json end
                json_end_idx = data_buffer.find('}') + 1

                if json_end_idx > 0:
                    # we found a valid json object
                    json_string = data_buffer[slice(0,json_end_idx)]
                    data_buffer = data_buffer[slice(json_end_idx, len(data_buffer))]

                    try:
                        json_object = json.loads(json_string)

                        colors = json_object['colors']
                        gray_scale_image = np.reshape(colors, (50, 120))
                        gray_scale_image = np.flip(gray_scale_image, 0)

                        # make a prediction based on input image
                        motion = network.predict(gray_scale_image, epsilon=epsilon)
                        connection.send(json.dumps(motion).encode('utf-8'))

                        # print received image from unity
                        show_received_image(gray_scale_image)
                        
                        # get the reward
                        isAgentOnTrack =  json_object['isOnTrack']
                        reward_record.append(1.0 if isAgentOnTrack else -5.0)
                        
                        # add state and action record
                        state_record.append(gray_scale_image)
                        action_record.append(motion)

                        # episode is over :(
                        if not isAgentOnTrack:
                            network.train(state_record, action_record, reward_record)

                            state_record = []
                            action_record = []
                            reward_record = []

                            print("Episode:", episode_counter, ", Epsilon:", epsilon)
                            # reset environment
                            connection.send("RESET".encode('utf-8'))

                            # increase the episode
                            episode_counter += 1
                    except Exception as e:
                        print('unable to parse json from data string', e)
            else:
                break
            
    finally:
        # Clean up the connection
        connection.close()


while True:
    run_socket_server()
