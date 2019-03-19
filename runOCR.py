# -*- coding: utf-8 -*-

from ocr_functionset import functions
import chainer
import chainer.links as L


class runOCR:
    def __init__(self, image):
        self.image = image
        self.CNNmodel = chainer.FunctionSet(conv1=L.Convolution2D(1, 64, 6),
                            conv2=L.Convolution2D(64, 256, 6),
                            conv3=L.Convolution2D(256, 512, 5),
                            l1 = L.Linear(18432, 8192),
                            l2 = L.Linear(8192, 6605))
        
        
    def run(self):
        original_list, processed_list = functions.convert_row(self.image)
    