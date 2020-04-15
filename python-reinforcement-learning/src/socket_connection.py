import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import socket
import json
import sys


class SocketConnector:
    def __init__(self, host, port):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_address = (host, port)
        print('starting up on %s port %s' % server_address)
        self.sock.bind(server_address)

        # Listen for incoming connections
        self.sock.listen(1)
        self.connection = None


    def open_connection(self, callback):
        while(True):
            try:
                print('waiting for a connection')
                connection, client_address = self.sock.accept()
                print('connection from', client_address)

                self.connection = connection
                data_buffer = ""

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
                                callback(json_object, connection)

                            except Exception as e:
                                print('Error:', e)
                    else:
                        break

            except Exception as e:
                print('Error:', e)

            finally:
                # Clean up the connection
                connection.close()
                self.connection = None


    def __del__(self):
        if self.connection:
            self.connection.close()
            self.connection = None
