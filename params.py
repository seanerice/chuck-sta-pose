from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from functools import partial
import time
import math

ip = "127.0.0.1"
port = 6669
client = SimpleUDPClient(ip, port)

# Constants
parts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
         'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
         'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee',
         'rightKnee', 'leftAnkle', 'rightAnkle']
numparts = len(parts)
part2idx = {p: i for i, p in enumerate(parts)}
idx2part = {i: p for i, p in enumerate(parts)}

params = ['r_arm_height', 'l_arm_height']


def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def lerp(a, b, t):
    return (1.0 - t) * (a + b) * t


def inv_lerp(a, b, v):
    return (v - a) / (b - a)


def remap(in_range, out_range, v):
    t = inv_lerp(in_range[0], in_range[1], v)
    return lerp(out_range[0], out_range[1], t)


pose = {p: (-1, -1) for p in parts}


def update_pose(data):
    xp, yp = msg2data(data)
    for i, (x, y) in enumerate(zip(xp, yp)):
        if x > -1 and y > -1:
            pose[idx2part[i]] = (x, y)


def center_of_mass(pose):
    xp = [
        p[0] for p in [
            pose[part] for part in [
                "leftShoulder",
                "rightShoulder",
                "leftHip",
                "rightHip"]]]
    yp = [
        p[1] for p in [
            pose[part] for part in [
                "leftShoulder",
                "rightShoulder",
                "leftHip",
                "rightHip"]]]

    avgx = sum(xp) / 4
    avgy = sum(yp) / 4
    return avgx, avgy


def params_dict(pose):
    pdict = {}

    pdict["l_arm_height"] = clamp(
        pose["leftWrist"][1],
        pose["nose"][1],
        pose["leftHip"][1])
    pdict["l_arm_height"] = inv_lerp(pose["nose"][1],
                                     pose["leftHip"][1],
                                     pdict["l_arm_height"])
    return pdict


def writelns(filename, lines):
    with open(filename, 'a') as file:
        for line in lines:
            file.write(line + "\n")


def handle_msg(addr, *args):
    update_pose(args)
    pose_param = params_dict(pose)
    client.send_message("/fs1/lpf", [pose_param['l_arm_height'] + 300, 1])
    print(pose_param["l_arm_height"])


def msg2data(args):
    x_points = [-1 for i in range(numparts)]
    y_points = [-1 for i in range(numparts)]
    for i in range(0, len(args), 3):
        part = args[i]

        x = float(args[i + 1])
        x_points[part2idx[part]] = x

        y = float(args[i + 2])
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
