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
image_show_timeout = time.time() * 1000.0

show_image = False
show_loss_history_plot = True
show_reward_sum_history_plot = True

# only impacts the decrease of epsilond througout the training
MAX_EPISODES = 200
# amount of episodes until we train the network
BATCH_SIZE = 10


# linearly decrese epsilon
def get_epsilon(episode_counter, max_epsiodes):
    # always leave at least 30% randomness
    return max(30, (100 - (episode_counter / max_epsiodes) * 100))

def show_loss_history(loss_history):
    if show_loss_history_plot:
        plt.figure(0)
        
        ax = sns.lineplot(data=np.array(loss_history))
        ax.set_title('Loss History')

        plt.show(block=False)
        plt.pause(0.001)


def show_reward_sum_history(reward_sum_history):
    if show_reward_sum_history_plot:
        plt.figure(1)
    
        ax = sns.lineplot(data=np.array(reward_sum_history))
        ax.set_title('Reward Sum History')

        plt.show(block=False)
        plt.pause(0.001)


def show_received_image(image):
    global image_show_timeout
    now = time.time() * 1000.0

    if (show_image and now > image_show_timeout) == True:
        plt.figure(2)
        plt.title('Agent Input Image')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show(block=False)
        plt.pause(0.001)
        image_show_timeout = now + (3 * 1000)


def run_socket_server():
    episode_counter = 0

    data_buffer = ""
    
    loss_history = []
    reward_sum_history = []

    reward_record = []
    state_record = []
    action_record = []
    discounted_reward_record = []

    overall_step_record = np.array([])


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

                        epsilon = get_epsilon(episode_counter, max_epsiodes=MAX_EPISODES)

                        # make a prediction based on input image
                        motion = network.predict(gray_scale_image, epsilon=epsilon)
                        connection.send(json.dumps(motion).encode('utf-8'))

                        # print received image from unity
                        show_received_image(gray_scale_image)
                        
                        # get the reward
                        isAgentOnTrack =  json_object['isOnTrack']
                        reward_record.append(1.0 if isAgentOnTrack else 1.0)
                        
                        # add state and action record
                        state_record.append(gray_scale_image)
                        action_record.append(motion)

                        # episode is over
                        if not isAgentOnTrack:
                            step_count = len(reward_record)

                            if step_count > 2:
                                reward_sum_history.append(step_count)
                                discounted_reward = network.discount_rewards(reward_record).tolist()
                                discounted_reward_record.extend(discounted_reward)

                                overall_step_record = np.append(overall_step_record, len(reward_record))
                                # increase the episode
                                episode_counter += 1

                            if step_count > 2 and (episode_counter % BATCH_SIZE) == 0:
                                # train and evaluate the model
                                loss = network.train(state_record, action_record, discounted_reward_record)
                                loss_history.append(loss)

                                print("Episode:", episode_counter)
                                print("Epsilon:", epsilon)

                                # show reward history
                                show_reward_sum_history(reward_sum_history)
                                # show loss function results
                                show_loss_history(loss_history)

                                state_record = []
                                action_record = []
                                discounted_reward_record = []

                            else:
                                time.sleep(0.5)

                            reward_record = []

                            # reset environment
                            print("---- RESET------------------------------------------------")
                            connection.send("RESET".encode('utf-8'))
                           
                    except Exception as e:
                        print('Error:', e)
            else:
                break
            
    finally:
        # Clean up the connection
        connection.close()


while True:
    try:
        run_socket_server()
    except Exception as e:
        print(e)
