import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import socket
import json
import sys

from environment import Environment

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ('localhost', 11000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

env = Environment()


def run_socket_server():
    data_buffer = ""
    
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
                        env.add_movement(json_object)
                        
                        # no need to move further
                        if env.is_terminal_state:
                            env.train_model_on_batch()
                            connection.send("RESET".encode('utf-8'))

                        # make prediction about the next move
                        else:
                            motion = env.get_predicted_motion()
                            connection.send(json.dumps(motion).encode('utf-8'))

                    except Exception as e:
                        print('Error:', e)
            else:
                break
            
    finally:
        # Clean up the connection
        connection.close()


if __name__ == '__main__':
    while True:
        try:
            run_socket_server()
        except Exception as e:
            print(e)
