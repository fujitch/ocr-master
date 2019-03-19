# -*- coding: utf-8 -*-

import ocr_functionset
import pickle
import cv2
import numpy as np
import chainer
import chainer.functions as F

image_path = 'fortest_hiragana.png'
functions = ocr_functionset.functions()

image_list = functions.convert(image_path)

dictionary = pickle.load(open('hiragana_dict.pkl', 'rb'))
model = pickle.load(open('latest.model', 'rb'))
count = 0
for img_row in image_list:
    for img in img_row:
        fname = 'each' + str(count) + '.jpg'
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(fname, img)
        count += 1
        if img.shape[1] < img.shape[0]:
            size = (48, 48*int(img.shape[1])/int(img.shape[0]))
        elif img.shape[1] > img.shape[0]:
            size = (48*int(img.shape[0])/int(img.shape[1]), 48)
        else:
            size = (48, 48)
        
        img = cv2.resize(img, size)
        
        dummyImg = np.zeros((48, 48, 3))
        dummyImg = dummyImg.astype(np.uint8)
        if img.shape[1] < img.shape[0]:
            matrix = np.ones((48, 48 - img.shape[1], 3))*255
            matrix = matrix.astype(np.uint8)
            dummyImg[:, :, 0] = np.c_[img[:, :, 0], matrix[:, :, 0]]
            dummyImg[:, :, 1] = np.c_[img[:, :, 1], matrix[:, :, 1]]
            dummyImg[:, :, 2] = np.c_[img[:, :, 2], matrix[:, :, 2]]
            img = dummyImg
        elif img.shape[0] < img.shape[1]:
            matrix = np.ones((48 - img.shape[0], 48, 3))*255
            matrix = matrix.astype(np.uint8)
            dummyImg[:, :, 0] = np.r_[img[:, :, 0], matrix[:, :, 0]]
            dummyImg[:, :, 1] = np.r_[img[:, :, 1], matrix[:, :, 1]]
            dummyImg[:, :, 2] = np.r_[img[:, :, 2], matrix[:, :, 2]]
            img = dummyImg
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        batch = np.zeros((1, 1, 48, 48))
        batch = batch.astype(np.float32)
        batch[0, 0, :, :] = img
        x = chainer.Variable(batch)
        x = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
        x = F.max_pooling_2d(F.relu(model.conv2(x)), 2)
        x = F.relu(model.conv3(x))
        x = F.dropout(F.relu(model.l1(x)), ratio=0.1, train=False)
        y = model.l2(x)
        y = F.softmax(y).data
        
        key = np.argmax(y)
        word = dictionary[key]
        print(word)