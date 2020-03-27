import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import socket
import json
import sys

from environment import Environment
from socket_connection import SocketConnector

env = Environment()


def on_data_received(json_object, connection):
    env.add_movement(json_object)
                        
    # no need to move further
    if env.is_terminal_state:
        env.train_model_on_batch()
        connection.send("RESET".encode('utf-8'))

    # make prediction about the next move
    else:
        motion = env.get_predicted_motion()
        connection.send(json.dumps(motion).encode('utf-8'))


socketConnector = SocketConnector('localhost', 11000)
socketConnector.open_connection(on_data_received)