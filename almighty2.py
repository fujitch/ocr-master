# -*- coding: utf-8 -*-

from chainer import cuda, Variable, optimizers
import chainer.functions as F
from CharRNN import CharRNN
from chainer import serializers
model = CharRNN(len(word2num)+1, n_units)
serializers.load_npz("latestmodel20171108.npz", model)

sentence = []
sentence.append('す')
sentence.append('る')
sentence.append('検')
sentence.append('討')
sentence.append('会')
sentence.append('」')
sentence.append('は')
sentence.append('２')
sentence.append('日')
sentence.append('、')
sentence.append('孑')

loss = 0

for i in range(len(sentence)-1):
    sent = sentence[i]
    sentNext = sentence[i+1]
    x_batch = np.array([word2num[sent.decode('utf-8')]])
    y_batch = np.array([word2num[sentNext.decode('utf-8')]])
    x = Variable(x_batch)
    t = Variable(y_batch)
    y = model(x, train=False)
    loss += F.softmax_cross_entropy(y, t).data

"""
y = yMatrix[27]
ySort = np.sort(y)[::-1]
yArgSort = np.argsort(y)[::-1]

for x in range(40):
    key = yArgSort[x]
    word = unichr(int(num2code[key], 16))
    print word
"""

"""
import codecs

for i in range(4, 5):
    fname = 'data/jawiki' + str(i) + '.txt'
    words = codecs.open(fname, 'rb', 'utf-8').read()
    pre = 0
    while pre != len(words):
        print(pre)
        pre = len(words)
        startw = words.find('<doc')
        endw = words.find('">') + 2
        if startw == -1 or endw == 1:
            break
        if startw > endw:
            startw = endw - 2
        if endw - startw > 100000:
            endw = startw + 70
        if startw == 0:
            apart = ''
        else:
            apart = words[:startw-1]
        bpart = words[endw:]
        words = apart + bpart
    pre = 0
    while pre != len(words):
        print(pre)
        pre = len(words)
        anotherw = words.find('</doc>')
        if anotherw == -1:
            break
        apart = words[:anotherw-1]
        if anotherw+20 > len(words):
            bpart = ''
        else:
            bpart = words[anotherw+6:]
        words = apart + bpart
    pre = 0
    while pre != len(words):
        print(pre)
        pre = len(words)
        startw = words.find('<ref')
        endw = words.find('</ref>') + 2
        if startw == -1 or endw == 1:
            break
        if startw > endw:
            startw = endw - 6
        if endw - startw > 100000:
            endw = startw + 40
        if startw == 0:
            apart = ''
        else:
            apart = words[:startw-1]
        bpart = words[endw:]
        words = apart + bpart
        
    f = codecs.open(fname, 'wb', 'utf-8')
    f.write(words)
    f.close()
"""