import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import socket
import json
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ('localhost', 11000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

# show the image every x times
image_counter = 0

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while connection:
            byte_data = connection.recv(200000)
            if byte_data:
                data = byte_data.decode('utf-8')
                # print(data);

                try:
                    data = json.loads(data)
                    print("parsed JSON object:", data['data'])

                    colors = data['colors']
                    gray_scale_image = np.reshape(colors, (50, 120))
                    gray_scale_image = np.flip(gray_scale_image, 0)

                    # Image debug output
                    if (image_counter % 5) == 0:
                        plt.imshow(gray_scale_image, cmap='gray', vmin=0, vmax=255)
                        plt.show(block=False)
                        plt.pause(0.1)

                    image_counter += 1
                except:
                    print('unable to parse json from data string')
            else:
                break
            
    finally:
        # Clean up the connection
        connection.close()