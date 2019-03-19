# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import random
from chainer import optimizers, Variable
import pickle
import ocr_functionset
from chainer import cuda
from chainer import serializers

## 初期値設定
original_image_size = 110 #画像サイズ
thresh = 127
batch_size = 100
dropout_ratio = 0.1
gpu_flag = 0
epochs = 10000000

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

functions = ocr_functionset.functions()

## 画像回転用定数を外に出しておく
image_size = tuple([original_image_size, original_image_size])
image_center = tuple([int(image_size[0]/2), int(image_size[1]/2)])
image_scale = 1.0

## 学習データセット読み込み

code2num = pickle.load(open('code2num_new.pkl', 'rb'))
num2code = pickle.load(open('num2code_new.pkl', 'rb'))
dataset = {}

for i in range(0,14):
    fname = "dataset" + str(i) + "_new.pkl"
    dictionary = pickle.load(open(fname, "rb"))
    dataset.update(dictionary)

dictionary = {}
num_of_data = len(code2num) # データセットの大きさ

print(num_of_data)
print(len(dataset))

# 学習パッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 1, 55, 55))
    batch = batch.astype(np.float32)
    output = np.zeros((batch_size))
    output = output.astype(np.int32)
    for i in range(batch_size):
        now_code = num2code[random.randint(0, num_of_data - 1)]
        image_list = dataset[now_code]
        batch[i, :, :, :] = functions.process_img(image_list[random.randint(0, len(image_list) - 1)], image_size, image_center, image_scale, thresh, original_image_size)
        output[i] = code2num[now_code]
    return batch, output

# オンライン学習のための1サイクル分のデータセット
def make_cycle(num_of_data):
    cycle = np.zeros((num_of_data, 1, 55, 55))
    cycle = cycle.astype(np.float32)
    output = np.zeros((num_of_data))
    output = output.astype(np.int32)
    for num in range(num_of_data):
        now_code = num2code[num]
        image_list = dataset[now_code]
        cycle[num, :, :, :] = functions.process_img(image_list[random.randint(0, len(image_list) - 1)], image_size, image_center, image_scale, thresh, original_image_size)
        output[num] = code2num[now_code]
    return cycle, output

# model作成
model = chainer.FunctionSet(conv1=L.Convolution2D(1, 64, 6),
                            conv2=L.Convolution2D(64, 256, 6),
                            conv3=L.Convolution2D(256, 512, 5),
                            l1 = L.Linear(18432, 8192),
                            l2 = L.Linear(8192, 6605))
serializers.load_npz("latestmodel20171115.npz", model)
if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

# optimizer
optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)

for epoch in range(epochs):
    print('start_epoch:' + str(epoch))
    cycle, output = make_cycle(num_of_data)
    cycle = cuda.to_gpu(cycle)
    output = cuda.to_gpu(output)
    for k in range(num_of_data/100 + 1):
        optimizer.zero_grads()
        if k == 68:
            x, t = Variable(cycle[100*k:100*k+6, :, :, :]), Variable(output[100*k:100*k+6])
        else:
            x, t = Variable(cycle[100*k:100*k+100, :, :, :]), Variable(output[100*k:100*k+100])
        x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
        x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
        x = F.relu(model.conv3(x))
        x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=True)
        y = model.l2(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
    cycle, output = make_batch(batch_size)
    x, t = Variable(cuda.to_gpu(cycle)), Variable(cuda.to_gpu(output))
    x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
    x = F.relu(model.conv3(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=False)
    y = model.l2(x)
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    print('epoch:' + str(epoch) + '  acc:' + str(acc.data) + '  loss:' + str(loss.data))
    
    if epoch%50 == 0:
        model.to_cpu()
        serializers.save_npz("latestmodel20171218.npz", model)
        model.to_gpu()