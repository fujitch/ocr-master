# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import random
from chainer import optimizers, Variable
import pickle
import ocr_functionset
from chainer import cuda


## 初期値設定
original_image_size = 110 #画像サイズ
thresh = 127
batch_size = 3
dropout_ratio = 0.01
gpu_flag = -1
epochs = 1000

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

functions = ocr_functionset.functions()

## 画像回転用定数を外に出しておく
image_size = tuple([original_image_size, original_image_size])
image_center = tuple([int(image_size[0]/2), int(image_size[1]/2)])
image_scale = 1.0

## 学習データセット読み込み
"""
dir_path = 'JAPANESE_DATASET/*'
dir_list = glob.glob(dir_path) # 画像ディレクトリ名取得

code2num = {} # コードとNo変換
num2code = {} # Noとコード変換
dataset = {} # データセット

for dir_name in dir_list:
    code = dir_name[17:21]
    if code not in code2num:
        num = len(code2num)
        code2num[code] = num
        num2code[num] = code
        image_list = []
    else:
        image_list = dataset[code]
    image = cv2.imread(dir_name)
    image_list.append(image)
    dataset[code] = image_list
    
"""
code2num = pickle.load(open('code2num.pkl', 'rb'))
num2code = pickle.load(open('num2code.pkl', 'rb'))
dataset = {}

fname = "dataset" + str(1) + ".pkl"
dictionary = pickle.load(open(fname, "rb"))
dataset.update(dictionary)

dictionary = {}
num_of_data = len(code2num) # データセットの大きさ

print(num_of_data)
print(len(dataset))

# 学習パッチ作成
def make_batch(batch_size):
    batch = xp.zeros((batch_size, 1, 110, 110))
    batch = batch.astype(xp.float32)
    output = xp.zeros((batch_size))
    output = output.astype(xp.int32)
    for i in range(batch_size):
        now_code = '4ea' + str(i)
        image_list = dataset[now_code]
        batch[i, :, :, :] = functions.process_img(image_list[random.randint(0, len(image_list) - 1)], image_size, image_center, image_scale, thresh, original_image_size)
        output[i] = i
    return batch, output
    

# model作成
model = chainer.FunctionSet(conv1=L.Convolution2D(1, 8, 5),
                            conv2=L.Convolution2D(8, 16, 5),
                            conv3=L.Convolution2D(16, 32, 4),
                            conv4=L.Convolution2D(32, 64, 5),
                            conv5=L.Convolution2D(64, 128, 4),
                            conv6=L.Convolution2D(128, 256, 4),
                            conv7=L.Convolution2D(256, 512, 4),
                            l1 = L.Linear(12800, 16384),
                            l2 = L.Linear(16384, 3))

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

accMatrix = np.zeros((epochs))
lossMatrix = np.zeros((epochs))
for epoch in range(epochs):
    optimizer.zero_grads()
    batch, output = make_batch(batch_size)
    if gpu_flag >= 0:
        x, t = Variable(cuda.to_gpu(batch)), Variable(cuda.to_gpu(output))
    else:
        x, t = Variable(batch), Variable(output)
    x = F.relu(F.local_response_normalization(model.conv1(x)))
    x = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(x))), 2)
    x = F.relu(model.conv3(x))
    x = F.max_pooling_2d(F.relu(model.conv4(x)), 2)
    x = F.relu(model.conv5(x))
    x = F.max_pooling_2d(F.relu(model.conv6(x)), 2)
    x = F.relu(model.conv7(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=True)
    y = model.l2(x)
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    accMatrix[epoch] = acc.data
    lossMatrix[epoch] = loss.data
    loss.backward()
    optimizer.update()
    if epoch%1 == 0:
        print('epoch:' + str(epoch) + '  acc:' + str(acc.data) + '  loss:' + str(loss.data))