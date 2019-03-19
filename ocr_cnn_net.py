# -*- coding: utf-8 -*-

from chainer import cuda, Variable, FunctionSet, optimizers, Chain
import chainer.links as L
import chainer.functions as F


class ocrCnn(Chain):
    def __init__(self, dropout_ratio):
        super(ocrCnn, self).__init__(
                conv1 = L.Convolution2D(1, 32, 9),
                conv2 = L.Convolution2D(32, 128, 6),
                conv3 = L.Convolution2D(128, 256, 4),
                conv4 = L.Convolution2D(256, 512, 3),
                l1 = L.Linear(32768, 16384),
                l2 = L.Linear(16384, 8196),
                l3 = L.Linear(8196, 6870)
                )
        self.train = True
        self.dropout_ratio = dropout_ratio
        
    def __call__(self, x, t):
        h = F.local_response_normalization(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2)
        h = F.local_response_normalization(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv4(h))
        h = F.dropout(F.relu(self.l1(h)), train=self.train, ratio=self.dropout_ratio)
        h = F.dropout(F.relu(self.l2(h)), train=self.train, ratio=self.dropout_ratio)
        y = self.l3(h)
        
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss, self.accuracy
    
    def get_score(self, x):
        h = F.local_response_normalization(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2)
        h = F.local_response_normalization(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv4(h))
        h = F.dropout(F.relu(self.l1(h)), train=False, ratio=self.dropout_ratio)
        h = F.dropout(F.relu(self.l2(h)), train=False, ratio=self.dropout_ratio)
        y = self.l3(h)
        
        return F.softmax(y)