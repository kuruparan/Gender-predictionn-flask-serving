import os
import json
import numpy as np
import tensorflow as tf



#from keras.models import load_model
from tensorflow.keras.models import Sequential, load_model

import tensorflow as tf

class GenderAPI:
    def __init__(self):
        #self.graph = tf.compat.v1.get_default_graph() #tf.get_default_graph()
        #self.model=load_model('Filterd_model_BiLSTM_0.9210641')
        data={'max_len': 15, 'vocab_len': 54, 'char_index': {'H': 0, 'd': 1, 'a': 2, 'C': 3, 'K': 4, 's': 5, 'Z': 6, 'z': 7, 'P': 8, 'N': 9, 't': 10, 'R': 11, 'b': 12, 'L': 13, 'B': 14, 'n': 15, 'S': 16, 'x': 17, 'Y': 18, 'U': 19, 'END': 20, 'r': 21, 'F': 22, 'e': 23, 'm': 24, 'v': 25, 'W': 26, 'f': 27, 'o': 28, 'h': 29, 'T': 30, 'w': 31, 'j': 32, 'q': 33, 'y': 34, 'i': 35, 'u': 36, 'O': 37, 'c': 38, 'D': 39, 'J': 40, 'V': 41, ' ': 42, 'g': 43, 'I': 44, 'k': 45, 'G': 46, 'p': 47, 'X': 48, 'A': 49, 'Q': 50, 'E': 51, 'l': 52, 'M': 53}}
        self.model, data = self.read_model('serving')
        self.max_len = data['max_len']
        self.vocab_len = data['vocab_len']
        self.char_index = data['char_index']

    def vector(self, i, n):
        tmp = np.zeros(n)
        tmp[i] = 1
        return tmp

    def read_model(self, network_path):
        if not os.path.exists(network_path):
            raise ValueError('Path not found : {}'.format(network_path))
        dat = json.loads(open(os.path.join(network_path, 'data.json')).read())
        #dat=data
        print(dat)
        mod = load_model(os.path.join(network_path, 'server.model'))
        #mod=load_model('Filterd_model_BiLSTM_0.9210641')
        return mod,dat

    def predict(self, names, labelize=True):
        """
        Returns gender of given names

        Args:
            names:      list of strings
            labelize:   returns 'M' or 'F' labels if set to True,
                        returns list of porbabilities otherwise
        """
        #with self.graph.as_default():
            # format input
        #print(names)
        names = [s.lower() for s in names]
        #print("lower",names)

        names = [list(i)+['END']*(self.max_len-len(i)) for i in names]
        names = [[self.vector(self.char_index[j], self.vocab_len) for j in i] for i in names]
        names = np.asarray(names)

        # predict gender
        out = self.model.predict(names).tolist()
        print(out)
        return [('M' if p[0] > p[1] else 'F') for p in out] if labelize else out

