import pandas as pd
import numpy as np

import os
import shutil
import glob
import sys

from collections import Counter
from gensim.models.keyedvectors import KeyedVectors

import config
import paths


def seq2ids(seq, indices):
    ret = []
    for elem, idx in zip(seq, indices):
        if idx == 1:
            ret.append([])
        ret[-1].append(elem)

    return ret


def files2sents(path, indices):
    files = glob.glob(path)
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, comment='#', delimiter='\t', header=None))
    df = pd.concat(dfs)

    ret = []
    for idx in indices:
        ret.append(df[idx].tolist())

    return ret


def sents2dicts(sents_train, sents_val, prets=None, initial_entries=None):
    dicts = []
    for sidx in range(len(sents_train)):
        if prets is None or prets[1] != sidx:
            dicts.append(Dict(sents_train[sidx], initial_entries))
            dicts[-1].add_entries(sents_val[sidx])
        else:
            dicts.append(Dictionary(sents_train[sidx], pret_file=prets[0]))
            dicts[-1].add_entries(sents_val[sidx])

    return dicts


def sents2ids(sents, dicts, indices, no_dict=None):
    ids = []
    for didx in range(len(dicts)):
        if no_dict is not None and didx in no_dict:
            ids.append(seq2ids(sents[didx], indices))
        else:
            ids.append(dicts[didx].sent2ids(sents[didx], indices))

    return ids



# def files2DataFrame(files, delim):
#     dfs = []
#     for file in files:
#         dfs.append(pd.read_csv(file, delimiter=delim, header=None, engine='python'))
#     ret = pd.concat(dfs)
#     return ret
#
#
# def files2sents(files):
#     if not config.small_data:
#         # df_train = preprocess.files2DataFrame(files_train, '\t')
#         # df_dev = preprocess.files2DataFrame(files_dev, '\t')
#         df = files2DataFrame(files, '\t')
#
#         indices, words, tags, heads, rels = \
#             df[0].tolist(), \
#             [w.lower() for w in df[1].tolist()], \
#             df[3].tolist(), \
#             df[6].tolist(), \
#             df[7].tolist()
#
#     if config.small_data:
#         df = files2DataFrame(files[:1], '\t')
#
#         indices, words, tags, heads, rels = \
#             df[0].tolist(), \
#             [w.lower() for w in df[1].tolist()], \
#             df[3].tolist(), \
#             df[6].tolist(), \
#             df[7].tolist()
#
#     return indices, words, tags, heads, rels

class Dictionary(object):

    def __init__(self, seq, pret_file=None):
        self.freezed = False
        self.cnt = Counter(seq)
        self.words_in_train = set(seq)
        self.i2x = ['UNK']
        self.add_entries()

        if pret_file:
            self._pret_file = pret_file
            self._add_pret_words()

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.x2i = reverse(self.i2x)

        self.freezed = True

    def add_entries(self, seq=None):
        if not self.freezed:
            for elem in self.words_in_train:
                if self.cnt[elem] >= config.minimal_count:
                    self.i2x.append(elem)
            # self.words_in_train = set(self.i2x)
        else:
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['UNK']

    def add_entry(self, elem):
        if elem not in self.x2i:
            if (self.cnt[elem] >= config.minimal_count and not self.freezed) or elem in self.initial_entries:
                self.x2i[elem] = len(self.x2i)
                self.i2x.append(elem)
            else:
                self.x2i[elem] = self.x2i['UNK']

    def sent2ids(self, seq, indices):
        ret = []
        for elem, idx in zip(seq, indices):
            if idx == 1:
                ret.append([])
            ret[-1].append(self.x2i[elem] if elem in self.x2i else self.x2i['UNK'])

        return ret

    # _add_pret_words and get_pret_embs are from jcyk

    def _add_pret_words(self):
        # self._words_in_train_data = len(self.i2x)
        # print
        # '#words in training set:', self._words_in_train_data
        # words_in_train_data = set(self.i2x)
        with open(self._pret_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line:
                    word = line[0]
                    if self.cnt[word] < config.minimal_count:  # add words in pret_file which do not occur in train_file to id2word
                        self.i2x.append(word)

    # print 'Total words:', len(self._id2word)

    def get_pret_embs(self):
        # assert (self._pret_file is not None), "No pretrained file provided."
        # embs = [[]] * len(self.i2x)  # make list of empty list of length of id2word
        # with open(self._pret_file) as f:
        #     for line in f.readlines():
        #         line = line.strip().split()
        #         if line:
        #             word, data = line[0], [float(l) for l in line[1:]]
        #             embs[self.x2i[word]] = data  #
        # emb_size = len(data)
        # for idx, emb in enumerate(embs):  # assign zero vector if a word is not provided with pretrained embedding
        #     if not emb:
        #         embs[idx] = np.zeros(emb_size)
        # # pret_embs = np.array(embs, dtype=np.float32)
        # pret_embs = np.array(embs)
        # return pret_embs / np.std(
        #     pret_embs)  # return an array of pretrained embeddings normalized by standard deviation
        word_vectors = KeyedVectors.load_word2vec_format(paths.pret_file, binary=False)

        return word_vectors



class Dict:
    def __init__(self, seq, initial_entries=None):
        self.i2x = {}
        self.x2i = {}
        self.appeared_x2i = {}
        self.appeared_i2x = {}
        self.freezed = False
        self.initial_entries = initial_entries

        if initial_entries is not None:
            for ent in initial_entries:
                self.add_entry(ent)

        for ent in seq:
            self.add_entry(ent)
        self.freezed = True

    def add_entry(self, ent):
        if ent not in self.x2i:
            if not self.freezed:
                self.i2x[len(self.i2x)] = ent
                self.x2i[ent] = len(self.x2i)
            else:
                self.x2i[ent] = self.x2i["UNK"]

    def add_entries(self, seq=None):
        if not self.freezed:
            for elem in self.words_in_train:
                if self.cnt[elem] >= config.minimal_count:
                    self.i2x.append(elem)
            # self.words_in_train = set(self.i2x)
        else:
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['UNK']

    def freeze(self):
        self.freezed = True

    def sent2ids(self, seq, indices):
        ret = []
        for elem, idx in zip(seq, indices):
            if idx == 1:
                ret.append([])
            ret[-1].append(self.x2i[elem] if elem in self.x2i else self.x2i['UNK'])

        return ret




def make_dir(dir):
    dir_num = 0
    dir_valify = dir

    while os.path.exists(dir_valify):
        dir_valify = dir + str(dir_num)
        dir_num += 1

    os.makedirs(dir_valify)

    return dir_valify


def save_codes(directory):
    python_codes = glob.glob('./*.py')
    for pycode in python_codes:
        shutil.copy2(pycode, directory)

def print2filecons(string):
    print(string)
    out2file(string)

def out2file(string):
    orig_out = sys.stdout
    filename = 'dev_result.txt' if not config.isTest else 'test_result.txt'
    f = open(filename, 'a')
    sys.stdout = f
    print(string)
    sys.stdout = orig_out
