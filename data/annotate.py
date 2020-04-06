import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from getkey import getkey, keys
import json
import itertools


def writelns(filename, lines):
    with open(filename, 'a') as file:
        for line in lines:
            file.write(line + "\n")


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        rows = file.readlines()
        for row in rows:
            n_row = [float(n) for n in row.strip().strip("\n").split(" ")]
            data.append(n_row)
        return data


def read_annotations(filename):
    with open(filename, 'r') as file:
        return json.load(file)


classes = ["r_arm_raise", "l_arm_raise", "r_arm_lower", "l_arm_lower",
           "l_arm_open", "r_arm_open", "l_arm_close", "r_arm_close",
           "point_l", "point_r"]
label2idx = {p: i for i, p in enumerate(classes)}


def labels2data(labels):
    label_data = [0 for i in range(len(classes))]
    for l in labels:
        if l in label2idx:
            label_data[label2idx[l]] = 1
    return label_data


class InteractiveAnnotator():
    def __init__(self, data):
        self.data = data

        plt.ion()
        self.background = np.zeros((800, 800))
        self.window = plt.imshow(self.background, cmap='gray', vmin=0, vmax=1)

        self.time = 0
        self.running = True

        self.state = 0
        self.start = 0
        self.end = 0

        while self.running:
            self.update()

    def update(self):
        # Render
        img = self.background.copy()
        row = data[self.time]
        for i in range(0, len(row), 2):
            x, y = int(row[i]), int(row[i + 1])

            cv2.circle(img, (x, y), 5, (1), 2)
        self.window.set_data(img)

        # Input
        if self.state == 0:
            print(self.start)
            self.scrub_start()
        elif self.state == 1:
            self.scrub_end()
            print(self.start, self.end)
        elif self.state == 2:
            self.input_classes()

    def stop(self):
        self.running = False

    def scrub_start(self):
        key = getkey()
        if key == keys.LEFT:
            self.time -= 1
        elif key == keys.RIGHT:
            self.time += 1
        elif key == keys.ENTER:
            self.state = 1
        self.start = self.time

    def scrub_end(self):
        key = getkey()
        if key == keys.LEFT:
            self.time -= 1
        elif key == keys.RIGHT:
            self.time += 1
        elif key == keys.ENTER:
            self.state = 2
        self.end = self.time

    def input_classes(self):
        pass


def frange(start, end, step):
    assert(step != 0)
    sample_count = int(abs(end - start) / step)
    return itertools.islice(itertools.count(start, step), sample_count)


def lerp(t, frm, to):
    return (to - frm) * t + frm


if __name__ == "__main__":
    pd_file = sys.argv[1]
    an_file = sys.argv[2]
    pose_data = read_data(pd_file)
    annotations = read_annotations(an_file)

    names = [
        "r_arm_height",
        "l_arm_height",
        "r_arm_raise",
        "l_arm_raise",
        "r_arm_lower",
        "l_arm_lower"]
    an_vec = {c: i for i, c in enumerate(names)}

    an_rows = {}
    for an in annotations:
        name = an["Name"]
        start = an["StartTime"]
        end = an["EndTime"]
        for i in range(start, end):
            if i not in an_rows:
                an_rows[i] = [0 for _ in range(6)]

            if "ParamStart" in an:
                pstart = an["ParamStart"]
                pend = an["ParamEnd"]
                an_rows[i][an_vec[name]] = lerp(
                    (i - start) / (end - start - 1), pstart, pend)
            else:
                an_rows[i][an_vec[name]] = 1

    train_data = []
    for i, pose in enumerate(pose_data):
        if i in an_rows:
            data_row = pose + an_rows[i]
            text = " ".join([str(f) for f in data_row])
            writelns(sys.argv[3], [text])
