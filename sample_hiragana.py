# -*- coding: utf-8 -*-

import glob
import numpy as np
import cv2
import pickle
import chainer
import chainer.functions as F

dropout_ratio = 0.1

dirList = glob.glob('hiragana73/*')
wordList = {}
imageList = {}

count = 1
for dirName in dirList:
    print(dirName)
    fileList = glob.glob(dirName + '/*')
    matrix = np.zeros((48, 48, len(fileList)))
    matrixColor = np.zeros((48, 48, 3, len(fileList)))
    for i in range(len(fileList)):
        fname = fileList[i]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        matrixColor[:, :, :, i] = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        matrix[:, :, i] = img
    wordList[str(count)] = matrix
    imageList[str(count)] = matrixColor
    count += 1
    
model = pickle.load(open('latest.model', 'rb'))

# テスト用バッチ作成
def make_batch(num):
    batch = np.zeros((73, 1, 48, 48))
    batch = batch.astype(np.float32)
    output = np.zeros((73))
    output = output.astype(np.int32)
    for i in range(73):
        dataList = wordList[str(i+1)]
        index = num
        data = dataList[:, :, index]
        data /= np.max(data)
        batch[i, :, :, :] = data
        output[i] = i
    return batch, output

"""
i = 0
batch, output = make_batch(i)
x = chainer.Variable(batch)
t = chainer.Variable(output)
x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
x = F.relu(model.conv3(x))
x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=False)
y = model.l2(x)
acc = F.accuracy(y, t)
y = F.softmax(y).data
"""

resultMatrix = []
accMatrix = []
for i in range(100):
    batch, output = make_batch(i)
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
    x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
    x = F.relu(model.conv3(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=False)
    y = model.l2(x)
    acc = F.accuracy(y, t).data
    y = F.softmax(y).data
    
    accMatrix.append(acc)
    resultMatrix.append(y)

accSum = 0
for i in range(100):
    accSum += accMatrix[i]