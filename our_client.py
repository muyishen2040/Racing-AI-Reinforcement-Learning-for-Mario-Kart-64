import socket

host = '0.0.0.0'  # Listen on all network interfaces
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))
sock.listen(1)
print("Waiting for a connection...")
connection, client_address = sock.accept()

try:
    print("Connection from", client_address)
    while True:
        data = connection.recv(1024)
        if data:
            print("Received obs:", data.decode())
            action = '[0, 0, 1, 0, 0]'  # Example action, replace with actual action logic
            connection.sendall(action.encode())
        else:
            print('Connection closed or no data received')
            break
finally:
    connection.close()
    sock.close()