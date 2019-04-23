import os
import sys
import cv2
import numpy as np
import neurolab as nl

input_f = 'letter.data'
"""img_resize_factor = 12
start, end = 6, -1
height, width = 16, 8

with open(input_f, 'r') as f:
    for line in f.readlines():
        data = np.array([255*float(x) for x in line.split('\t')[start:end]])
        img = np.reshape(data, (height, width))
        img_scaled = cv2.resize(img, None, fx=img_resize_factor, fy=img_resize_factor)
        print(line)
        cv2.imshow('Img', img_scaled)

        c = cv2.waitKey()
        if c == 27:
            break"""

num_data = 50
orig_labels = 'omandig'
num_orig_labels = len(orig_labels)

num_train = int(0.9*num_data)
num_test = num_data - num_train

start = 6
end = -1

data = []
labels = []

with open(input_f, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels:
            continue

        label = np.zeros((num_orig_labels, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)

        if len(data) >= num_data:
            break

data = np.asfarray(data)
labels = np.array(labels).reshape(num_data, num_orig_labels)

num_dims = len(data[0])
nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_orig_labels])
nn.trainf = nl.train.train_gd

error_progress = nn.train(data[:num_train, :], labels[:num_train, :], epochs=10000, show=1000, goal=0.01)

pred_test = nn.sim(data[num_train:, :])
for i in range(num_test):
    print("\n Original: ", orig_labels[np.argmax(labels[i])])
    print("Predicted: ", orig_labels[np.argmax(pred_test[i])])