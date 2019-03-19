import numpy as np
from chainer import FunctionSet
import chainer.functions as F
import chainer.links as L

class CharRNN(FunctionSet):

    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
            embed = L.EmbedID(n_vocab, n_units),
            l1 = L.LSTM(n_units, n_units),
            l2 = L.LSTM(n_units, n_units),
            l3   = L.Linear(n_units, n_vocab),
        )
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)
            
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x, dropout_ratio=0.1, train=True):

        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, ratio=dropout_ratio, train=train))
        h2 = self.l2(F.dropout(h1, ratio=dropout_ratio, train=train))
        y = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        
        return y