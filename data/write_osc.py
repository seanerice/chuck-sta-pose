from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from functools import partial
import time

# Constants
parts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
         'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
         'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee',
         'rightKnee', 'leftAnkle', 'rightAnkle']
numparts = len(parts)
part2idx = {p: i for i, p in enumerate(parts)}
idx2part = {i: p for i, p in enumerate(parts)}


def writelns(filename, lines):
    with open(filename, 'a') as file:
        for line in lines:
            file.write(line + "\n")


def handle_msg(addr, *args):
    xp, yp = msg2data(args)
    data_row = []
    for x, y in zip(xp, yp):
        data_row.append(x)
        data_row.append(y)
    text_data = " ".join([str(f) for f in data_row])
    writelns(addr.replace("/", "_") + ".txt", [text_data])


def msg2data(args):
    x_points = [-1 for i in range(numparts)]
    y_points = [-1 for i in range(numparts)]
    for i in range(0, len(args), 3):
        part = args[i]

        x = args[i + 1]
        x_points[part2idx[part]] = x

        y = args[i + 2]
        y_points[part2idx[part]] = y

    return x_points, y_points


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 6666

    dispatcher = Dispatcher()
    dispatcher.map("/pose/*", handle_msg)
    # dispatcher.set_default_handler(println)

    # Blocking server ensures messages handled in order
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.serve_forever()
