import pandas as pd


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
        dfs.append(pd.read_csv(file, delimiter=delim, header=None))
    ret = pd.concat(dfs)
    return ret


class Dictionary(object):

    def __init__(self, seq, initial_entries=None):
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
            if not self.freezed:
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
