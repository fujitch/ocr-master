# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
import chainer
import chainer.functions as F
import random
from chainer import optimizers
import pickle

# 初期値設定
training_epochs = 10000
dropout_ratio = 0.1

# 学習用データの読み込み

dirList = glob.glob('hiragana73/*')
wordList = {}

count = 1
for dirName in dirList:
    print(dirName)
    fileList = glob.glob(dirName + '/*')
    matrix = np.zeros((48, 48, len(fileList)))
    for i in range(len(fileList)):
        fname = fileList[i]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        matrix[:, :, i] = img
    wordList[str(count)] = matrix
    count += 1


# 学習モデル作成
model = chainer.FunctionSet(conv1=F.Convolution2D(1, 20, 5),
                            conv2=F.Convolution2D(20, 50, 5),
                            conv3=F.Convolution2D(50, 100, 4),
                            l1=F.Linear(3600, 256),
                            l2=F.Linear(256, 73))

# optimizerをAdamに設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習バッチ作成
def make_batch():
    batch = np.zeros((73, 1, 48, 48))
    batch = batch.astype(np.float32)
    output = np.zeros((73))
    output = output.astype(np.int32)
    for i in range(73):
        dataList = wordList[str(i+1)]
        index = random.randint(0, dataList.shape[2]-1)
        data = dataList[:, :, index]
        data /= np.max(data)
        batch[i, :, :, :] = data
        output[i] = i
    return batch, output

# 本番トレーニング
for epoch in range(training_epochs):
    optimizer.zero_grads()
    batch, output = make_batch()
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
    x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
    x = F.relu(model.conv3(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=True)
    y = model.l2(x)
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    loss.backward()
    optimizer.update()
    print('epoch:' + str(epoch) + '  acc:' + str(acc.data) + '  loss:' + str(loss.data))
    """
    if epoch%100 == 0:
        pickle.dump(model, open('latest.model', 'wb'))
    """