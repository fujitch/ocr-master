# -*- coding: utf-8 -*-

import ocr_functionset
import matplotlib.pyplot as plt
import numpy as np
import random

xp = np

functions = ocr_functionset.functions()

batch_size = 1
original_image_size = 110 #画像サイズ
thresh = 127

## 画像回転用定数を外に出しておく
image_size = tuple([original_image_size, original_image_size])
image_center = tuple([int(image_size[0]/2), int(image_size[1]/2)])
image_scale = 1.0

# 学習パッチ作成
def make_batch(batch_size):
    batch = xp.zeros((batch_size, 1, 55, 55))
    batch = batch.astype(xp.float32)
    output = xp.zeros((batch_size))
    output = output.astype(xp.int32)
    for i in range(batch_size):
        now_code = '4ea' + str(i)
        image_list = dataset[now_code]
        batch[i, :, :, :] = functions.process_img(image_list[random.randint(0, len(image_list) - 1)], image_size, image_center, image_scale, thresh, original_image_size)
        output[i] = i
    return batch

batch = make_batch(batch_size)

plt.imshow(batch[0, 0, :, :])