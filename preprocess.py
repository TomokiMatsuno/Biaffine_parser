import pandas as pd

import os

from collections import Counter

import config

def seq2ids(seq, indices):
    ret = []
    for elem, idx in zip(seq, indices):
        if idx == 1:
            ret.append([])
        ret[-1].append(elem)

    return ret


def files2DataFrame(files, delim):
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, delimiter=delim, header=None, engine='python'))
    ret = pd.concat(dfs)
    return ret


class Dictionary(object):

    def __init__(self, seq, initial_entries=None):
        self.cnt = Counter(seq)
        self.i2x = {}
        self.x2i = {}
        self.freezed = False
        self.initial_entries = initial_entries \
            if initial_entries is not None \
            else ['UNK']
        self.add_entries(self.initial_entries + seq)
        self.freezed = True

    def add_entries(self, seq):
        for elem in seq:
            self.add_entry(elem)

    def add_entry(self, elem):
        if elem not in self.x2i:
            if (self.cnt[elem] >= config.minimal_count and not self.freezed) or elem in self.initial_entries:
                self.x2i[elem] = len(self.x2i)
                self.i2x[len(self.i2x)] = elem
            else:
                self.x2i[elem] = self.x2i['UNK']

    def sent2ids(self, seq, indices):
        ret = []
        for elem, idx in zip(seq, indices):
            if idx == 1:
                ret.append([])
            ret[-1].append(self.x2i[elem])

        return ret

def make_dir(dir):
    dir_num = 0
    dir_valify = dir

    while os.path.exists(dir_valify):
        dir_valify = dir + str(dir_num)
        dir_num += 1

    os.makedirs(dir_valify)

    return dir_valify


