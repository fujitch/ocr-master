# -*- coding: utf-8 -*-

import ocr_functionset
import pickle
import cv2
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from CharRNN import CharRNN

image_path = 'fortest_all6.png'
functions = ocr_functionset.functions()

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
original_list, processed_list = functions.convert_row(image)

num2code = pickle.load(open('num2code_new.pkl', 'rb'))
model = chainer.FunctionSet(conv1=L.Convolution2D(1, 64, 6),
                            conv2=L.Convolution2D(64, 256, 6),
                            conv3=L.Convolution2D(256, 512, 5),
                            l1 = L.Linear(18432, 8192),
                            l2 = L.Linear(8192, 6605))
serializers.load_npz("latestmodel20171218.npz", model)
# model = pickle.load(open('latestmodel.pkl', 'rb'))
count = 0
dropout_ratio = 0.1

checkMatrix = np.zeros((1000, 55, 55))
accMatrix = []
yMatrix = []
wordList = []
imgMatrix = []

def discrimination_one_image(img):
    if img.shape[1] < img.shape[0]:
        size = (55*int(img.shape[1])/int(img.shape[0]), 55)
    elif img.shape[1] > img.shape[0]:
        size = (55, 55*int(img.shape[0])/int(img.shape[1]))
    else:
        size = (55, 55)
    img = cv2.resize(img, size)
    dummyImg = np.zeros((55, 55, 3))
    dummyImg = dummyImg.astype(np.uint8)
    if img.shape[1] < img.shape[0]:
        matrix = np.ones((55, 55 - img.shape[1], 3))*255
        matrix = matrix.astype(np.uint8)
        dummyImg[:, :, 0] = np.c_[img[:, :, 0], matrix[:, :, 0]]
        dummyImg[:, :, 1] = np.c_[img[:, :, 1], matrix[:, :, 1]]
        dummyImg[:, :, 2] = np.c_[img[:, :, 2], matrix[:, :, 2]]
        img = dummyImg
    elif img.shape[0] < img.shape[1]:
        matrix = np.ones((55 - img.shape[0], 55, 3))*255
        matrix = matrix.astype(np.uint8)
        dummyImg[:, :, 0] = np.r_[img[:, :, 0], matrix[:, :, 0]]
        dummyImg[:, :, 1] = np.r_[img[:, :, 1], matrix[:, :, 1]]
        dummyImg[:, :, 2] = np.r_[img[:, :, 2], matrix[:, :, 2]]
        img = dummyImg
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = 255 - img
    img = img.astype(np.float32) / 255
    dummyZeros = np.zeros((55, 55))
    dummyZeros[10:45, 10:45] = cv2.resize(img, (35, 35))
    img = dummyZeros
    batch = np.zeros((1, 1, 55, 55))
    batch = batch.astype(np.float32)
    batch[0, 0, :, :] = img
    x = chainer.Variable(batch)
    x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
    x = F.relu(model.conv3(x))
    x = F.dropout(F.relu(model.l1(x)), ratio=dropout_ratio, train=False)
    y = model.l2(x)
    return F.softmax(y).data

def optimization_one_word(original, keyList, accuracyList, colList):
    if len(accuracyList) == 1:
        return keyList, accuracyList, colList
        
    for i in range(len(accuracyList)):
        if accuracyList[i] > 0.99:
            continue
        if i == 0:
            rightJoin = original[:, colList[i][0]:colList[i+1][1], :]
            y = discrimination_one_image(rightJoin)
            if max(max(y)) > 0.99:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k][0]
                        dummy[1] = colList[k+1][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(y))
                        newAccuracyList.append(max(max(y)))
                    elif k == i + 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
            elif max(max(y)) > 0.95 and max(max(y)) > accuracyList[i] and max(max(y)) > accuracyList[i+1]:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k][0]
                        dummy[1] = colList[k+1][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(y))
                        newAccuracyList.append(max(max(y)))
                    elif k == i + 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
                
        elif i == len(accuracyList) - 1:
            leftJoin = original[:, colList[i-1][0]:colList[i][1], :]
            y = discrimination_one_image(leftJoin)
            if max(max(y)) > 0.99:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k-1][0]
                        dummy[1] = colList[k][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(y))
                        newAccuracyList.append(max(max(y)))
                    elif k == i - 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
            elif max(max(y)) > 0.95 and max(max(y)) > accuracyList[i] and max(max(y)) > accuracyList[i-1]:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k-1][0]
                        dummy[1] = colList[k][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(y))
                        newAccuracyList.append(max(max(y)))
                    elif k == i - 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
                
        else:
            rightJoin = original[:, colList[i][0]:colList[i+1][1], :]
            yr = discrimination_one_image(rightJoin)
            if max(max(yr)) > 0.99:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k][0]
                        dummy[1] = colList[k+1][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(yr))
                        newAccuracyList.append(max(max(yr)))
                    elif k == i + 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
            leftJoin = original[:, colList[i-1][0]:colList[i][1], :]
            yl = discrimination_one_image(leftJoin)
            if max(max(yl)) > 0.99:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k-1][0]
                        dummy[1] = colList[k][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(yl))
                        newAccuracyList.append(max(max(yl)))
                    elif k == i - 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
            if max(max(yl)) > 0.95 and max(max(yl)) > accuracyList[i] and max(max(yl)) > max(max(yr)) and max(max(yl)) > accuracyList[i-1]:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k-1][0]
                        dummy[1] = colList[k][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(yl))
                        newAccuracyList.append(max(max(yl)))
                    elif k == i - 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
            if max(max(yr)) > 0.95 and max(max(yr)) > accuracyList[i] and max(max(yr)) > max(max(yl)) and max(max(yr)) > accuracyList[i+1]:
                newColList = []
                newKeyList = []
                newAccuracyList = []
                for k in range(len(colList)):
                    if k == i:
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colList[k][0]
                        dummy[1] = colList[k+1][1]
                        newColList.append(dummy)
                        newKeyList.append(np.argmax(yr))
                        newAccuracyList.append(max(max(yr)))
                    elif k == i + 1:
                        continue
                    else:
                        newColList.append(colList[k])
                        newKeyList.append(keyList[k])
                        newAccuracyList.append(accuracyList[k])
                return newKeyList, newAccuracyList, newColList
    return keyList, accuracyList, colList

for i in range(len(processed_list)):
    original = original_list[i]
    processed = processed_list[i]
    colListDefault = functions.simply_cut_col(processed_list[i])
    keyList = []
    accuracyList = []
    for k in range(len(colListDefault)):
        img = original[:, colListDefault[k][0]:colListDefault[k][1], :]
        y = discrimination_one_image(img)
        keyList.append(np.argmax(y))
        accuracyList.append(max(max(y)))
    
    
    wordsLength = len(keyList)
    while True:
        keyList, accuracyList, colListDefault = optimization_one_word(original, keyList, accuracyList, colListDefault)
        if len(keyList) != wordsLength:
            wordsLength = len(keyList)
        else:
            break
    
    for l in range(len(keyList)):
        key = keyList[l]
        img = original[:, colListDefault[l][0]:colListDefault[l][1], :]
        y = discrimination_one_image(img)
        word = unichr(int(num2code[key], 16))
        wordList.append(word)
        imgMatrix.append(img)
        yMatrix.append(max(y))
        accMatrix.append(max(max(y)))
        
## 文脈補正

word2num = pickle.load(open('word2num.pkl', 'rb'))
model = CharRNN(len(word2num)+1, 1024)
serializers.load_npz("latestmodel20171108.npz", model)
for i in range(len(wordList)-1):
    x_batch = np.array([word2num[wordList[i]]])
    y_batch = np.array([word2num[wordList[i+1]]])
    x = chainer.Variable(x_batch)
    t = chainer.Variable(y_batch)
    y = model(x, train=False)
    loss = F.softmax_cross_entropy(y, t).data
    if loss < 10:
        continue
    if accMatrix[i+1] > 0.99:
        continue
    if loss < 15 and accMatrix[i+1] > 0.9:
        continue
    yArgSort = np.argsort(yMatrix[i+1])[::-1]
    optWordNum = 0
    minLoss = 100000
    if accMatrix[i+1] > 0.8:
        roop = 3
    elif accMatrix[i+1] > 0.6:
        roop = 5
    elif accMatrix[i+1] > 0.3:
        roop = 15
    elif accMatrix[i+1] > 0.2:
        roop = 30
    elif accMatrix[i+1] > 0.1:
        roop = 40
    else:
        roop = 50
    for k in range(roop):
        sumLoss = 0
        tryModel = model
        y_batch = np.array([word2num[unichr(int(num2code[yArgSort[k]], 16))]])
        t = chainer.Variable(y_batch)
        sumLoss += F.softmax_cross_entropy(y, t).data
        if i == len(wordList)-2:
            if minLoss > sumLoss:
                optWordNum = k
                minLoss = sumLoss
            
        elif i > len(wordList) - 7:
            x_batch = np.array([word2num[unichr(int(num2code[yArgSort[k]], 16))]])
            y_batch = np.array([word2num[wordList[i+2]]])
            x = chainer.Variable(x_batch)
            t = chainer.Variable(y_batch)
            y = tryModel(x, train=False)
            sumLoss += F.softmax_cross_entropy(y, t).data
            for l in range(len(wordList) - 2 - i):
                x_batch = np.array([word2num[wordList[i+2+l]]])
                y_batch = np.array([word2num[wordList[i+3+l]]])
                x = chainer.Variable(x_batch)
                t = chainer.Variable(y_batch)
                y = tryModel(x, train=False)
                sumLoss += F.softmax_cross_entropy(y, t).data
            if minLoss > sumLoss:
                optWordNum = k
                minLoss = sumLoss
                
        else:
            x_batch = np.array([word2num[unichr(int(num2code[yArgSort[k]], 16))]])
            y_batch = np.array([word2num[wordList[i+2]]])
            x = chainer.Variable(x_batch)
            t = chainer.Variable(y_batch)
            y = tryModel(x, train=False)
            sumLoss += F.softmax_cross_entropy(y, t).data
            for l in range(4):
                x_batch = np.array([word2num[wordList[i+2+l]]])
                y_batch = np.array([word2num[wordList[i+3+l]]])
                x = chainer.Variable(x_batch)
                t = chainer.Variable(y_batch)
                y = tryModel(x, train=False)
                sumLoss += F.softmax_cross_entropy(y, t).data
            if minLoss > sumLoss:
                optWordNum = k
                minLoss = sumLoss
                
    wordList[i+1] = unichr(int(num2code[yArgSort[optWordNum]], 16))
    
for word in wordList:
    print word