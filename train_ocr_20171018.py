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

## 初期値設定
original_image_size = 110 #画像サイズ
thresh = 127
batch_size = 100
dropout_ratio = 0.01
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
"""
dir_path = 'JAPANESE_DATASET_selected/*'
dir_list = glob.glob(dir_path) # 画像ディレクトリ名取得

code2num = {} # コードとNo変換
num2code = {} # Noとコード変換
dataset = {} # データセット

for dir_name in dir_list:
    code = dir_name[26:30]
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

for i in range(1,20):
    fname = "dataset" + str(i) + ".pkl"
    dictionary = pickle.load(open(fname, "rb"))
    dataset.update(dictionary)

dictionary = {}
num_of_data = len(code2num) # データセットの大きさ

print(num_of_data)
print(len(dataset))

# 学習パッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 1, 110, 110))
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
    cycle = np.zeros((num_of_data, 1, 110, 110))
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
model = chainer.FunctionSet(conv1=L.Convolution2D(1, 8, 5),
                            conv2=L.Convolution2D(8, 16, 5),
                            conv3=L.Convolution2D(16, 32, 4),
                            conv4=L.Convolution2D(32, 64, 5),
                            conv5=L.Convolution2D(64, 128, 4),
                            conv6=L.Convolution2D(128, 256, 4),
                            conv7=L.Convolution2D(256, 512, 4),
                            l1 = L.Linear(12800, 16384),
                            l2 = L.Linear(16384, 6870))

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


for epoch in range(epochs):
    print('start_epoch:' + str(epoch))
    cycle, output = make_cycle(num_of_data)
    cycle = cuda.to_gpu(cycle)
    output = cuda.to_gpu(output)
    for k in range(num_of_data/2):
        optimizer.zero_grads()
        x, t = Variable(cycle[2*k:2*k+1, :, :, :]), Variable(output[2*k:2*k+1])
        x = F.relu(model.conv1(x))
        x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
        x = F.relu(model.conv3(x))
        x = F.max_pooling_2d(F.relu(model.conv4(x)), 2)
        x = F.relu(model.conv5(x))
        x = F.max_pooling_2d(F.relu(model.conv6(x)), 2)
        x = F.relu(model.conv7(x))
        x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=True)
        y = model.l2(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
    cycle, output = make_batch(batch_size)
    x, t = Variable(cuda.to_gpu(cycle)), Variable(cuda.to_gpu(output))
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
    x = F.relu(model.conv3(x))
    x = F.max_pooling_2d(F.relu(model.conv4(x)), 2)
    x = F.relu(model.conv5(x))
    x = F.max_pooling_2d(F.relu(model.conv6(x)), 2)
    x = F.relu(model.conv7(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=True)
    y = model.l2(x)
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    print('epoch:' + str(epoch) + '  acc:' + str(acc.data) + '  loss:' + str(loss.data))
    """
    if epoch%100000 == 0:
        pickle.dump(model, open('latestmodel.pkl', 'wb'))
    """