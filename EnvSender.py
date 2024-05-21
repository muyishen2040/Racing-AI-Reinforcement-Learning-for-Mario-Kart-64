#!/bin/python
import socket
import time
import gym, gym_mupen64plus
import errno
import pickle
import struct

def try_connect():
    while True:
        try:
            sock.connect((host, port))
            print("Connected")
            break
        except Exception as e:
            time.sleep(1)

def is_socket_closed(sock):
    try:
        # this will try to read bytes without blocking and also without removing them from buffer (peek only)
        data = sock.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
        if len(data) == 0:
            return True
    except socket.error as e:
        if e.errno == errno.EWOULDBLOCK:
            return False  # socket is open and reading from it would block
        elif e.errno == errno.ECONNRESET:
            return True  # socket was closed for some other reason
        else:
            return True
    return False

def send_data(data_to_send):
    data = pickle.dumps(data_to_send)
    length = struct.pack('>I', len(data))
    sock.sendall(length)
    sock.sendall(data)

def recv_data():
    data = sock.recv(4096)
    return data
    
host = '172.17.0.1'
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

env = gym.make('Mario-Kart-Luigi-Raceway-v0')

while True:
    if is_socket_closed(sock):
        print("Waiting for connection")
        try:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        except:
            pass
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try_connect()

    data = recv_data()
    action = pickle.loads(data)
    if action == "reset":
        obs = env.reset()
        send_data(obs)
    else:
        obs, rew, done, info = env.step(action)
        send_data((obs, rew, done, info))
