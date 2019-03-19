# -*- coding: utf-8 -*-

import time
import sys
import argparse
import pickle
import copy
import codecs

import numpy as np
from chainer import cuda, Variable, optimizers
import chainer.functions as F
from CharRNN import CharRNN
from chainer import serializers

# input data

def load_data():
    word2num = pickle.load(open('word2num.pkl', 'rb'))
    num2word = pickle.load(open('num2word.pkl', 'rb'))
    words = codecs.open('jawiki1.txt', 'rb').read()
    words = words.replace('\n', '')
    words = words.replace(' ', '')
    words = words.replace('　', '')
    words = words.decode('utf-8')
    words = list(words)
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word in word2num:
            dataset[i] = word2num[word]
        else:
            dataset[i] = len(word2num)
    return dataset, word2num, num2word


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',                        type=int,   default=0)
parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
parser.add_argument('--dropout',                    type=float, default=0.0)
parser.add_argument('--init_from',                  type=str,   default='')

args = parser.parse_args()

n_epochs    = 1000
n_units     = 1024
batchsize   = 50
bprop_len   = 50
grad_clip   = 5

dataset, word2num, num2word = load_data()

if len(args.init_from) > 0:
    model = pickle.load(open(args.init_from, 'rb'))
else:
    model = CharRNN(len(word2num)+1, n_units)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
optimizer.setup(model)

whole_len    = dataset.shape[0]
jump         = whole_len / batchsize
epoch        = 0
start_at     = time.time()
cur_at       = start_at
if args.gpu >= 0:
    accum_loss   = Variable(cuda.zeros(()))
else:
    accum_loss   = Variable(np.zeros((), dtype=np.float32))
    
model.reset_state()

print('going to train {} iterations'.format(jump * n_epochs))
for i in range(int(jump)*n_epochs):
    x_batch = np.array([dataset[(jump * j + i) % whole_len]
                        for j in range(batchsize)])
    y_batch = np.array([dataset[(jump * j + i + 1) % whole_len]
                        for j in range(batchsize)])

    if args.gpu >=0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    x = Variable(x_batch)
    t = Variable(y_batch)
    y = model(x, dropout_ratio=args.dropout)
    loss_i = F.softmax_cross_entropy(y, t)
    accum_loss += loss_i

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        now = time.time()
        print('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))
        cur_at = now

        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        if args.gpu >= 0:
            accum_loss = Variable(cuda.zeros(()))
        else:
            accum_loss = Variable(np.zeros((), dtype=np.float32))

        optimizer.clip_grads(grad_clip)
        optimizer.update()

    if (i + 1) % 10000 == 0:
        pickle.dump(copy.deepcopy(model).to_cpu(), open('latest.chainermodel', 'wb'))
        model.to_cpu()
        serializers.save_npz("latestmodel20171108.npz", model)
        model.to_gpu()


    sys.stdout.flush()
