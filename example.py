#!/bin/python
import socket
import gym, gym_mupen64plus


host = '172.17.0.1'
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

print("NOOP waiting for green light")
for i in range(18):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0])

print("GO! ...drive straight as fast as possible...")
for i in range(50):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0])
    sock.sendall(str(obs).encode())
    action = sock.recv(1024).decode()
    env.step(eval(action))

print("Doughnuts!!")
for i in range(10000):
    if i % 100 == 0:
        print("Steps " + str(i))
    (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0])
    sock.sendall(str(obs).encode())
    action = sock.recv(1024).decode()
    env.step(eval(action))

sock.close()
env.close()

