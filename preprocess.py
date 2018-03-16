import pandas as pd
import numpy as np

import os
import shutil
import glob
import sys

from collections import Counter

import config
import paths

def files2sents_ch(files):
    words = []
    poses = []
    heads = []
    rels = []
    indices = []

    for file in files:
        tmp_indices = []
        widx = 1
        data = pd.read_csv(file, engine='python', encoding='GB2312', delimiter='\t', header=None,
                           skip_blank_lines=False)
        for idx in range(data.shape[0]):
            if data.iloc[idx, 1] is not None:
                tmp_indices.append(widx)
                widx += 1
            else:
                widx = 1
        data_out = pd.read_csv(file, engine='python', encoding='GB2312', delimiter='\t', header=None,
                           skip_blank_lines=True)
        # data_out['indices'] = tmp_indices
        indices.extend(tmp_indices)
        words.extend(data_out[0])
        poses.extend(data_out[1])
        heads.extend(data_out[2])
        rels.extend(data_out[3])

        if len(tmp_indices) != len(data_out[0]):
            # print(data_out)
            print('mismatch!')


    return indices, words, poses, heads, rels


def seq2ids(seq, indices, bi=False):
    ret = []
    for elem, idx in zip(seq, indices):
        if idx == 1:
            ret.append([])
            BI = 'B_'
        if bi:
            ret[-1].append(BI + elem)
        else:
            ret[-1].append(elem)
        BI = 'I_'

    return ret

def tochar(seq_word, seq_tag, indices_word, isTag=False):
    seq_char = []
    indices_char = []

    idx_char = 1
    idx_tag = 1
    for word, tag, idx_word in zip(seq_word, seq_tag, indices_word):
        if idx_word == 1:
            # if len(seq_char) > 0:
            #     seq_char[-1].append(2)  #index of EOS
            seq_char.append([])
            # seq_char[-1].append(1)      #index of BOS
            idx_char = 1
            idx_tag = 1
        if not isTag:
            for c in word:
                seq_char[-1].append(c)
                indices_char.append(idx_char)
                idx_char += 1
        else:
            # seq_char[-1].extend([tag] * len(word))
            seq_char[-1].extend([tag] + ['JNT'] * (len(word) - 1))
            indices_char.extend([i for i in range(idx_tag, idx_tag + len(word))])
            idx_tag += len(word)

    return seq_char, indices_char


def files2DataFrame(files, delim):
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, delimiter=delim, header=None, engine='python'))
    ret = pd.concat(dfs)
    return ret


class Dictionary(object):

    def __init__(self, seq, pret_file=None):
        self.freezed = False
        self.cnt = Counter(seq)
        self.words_in_train = set()
        self.i2x = ['UNK', 'BOS', 'EOS', 'JNT']
        self.add_entries()

        if pret_file:
            self._pret_file = pret_file
            self._add_pret_words()

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.x2i = reverse(self.i2x)

        self.freezed = True

    def add_entries(self, seq=None):
        if not self.freezed:
            for elem in self.cnt:
                if self.cnt[elem] >= config.minimal_count:
                    self.i2x.append(elem)
            self.words_in_train = set(self.i2x)
        else:
            unk_count = 0
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['UNK']
                    unk_count += 1
            ahaha = 0

    def add_entry(self, elem):
        if elem not in self.x2i:
            if (self.cnt[elem] >= config.minimal_count and not self.freezed) or elem in self.initial_entries:
                self.x2i[elem] = len(self.x2i)
                self.i2x.append(elem)
            else:
                self.x2i[elem] = self.x2i['UNK']

    def sent2ids(self, seq, indices, BOS_EOS=False):
        ret = []
        for elem, idx in zip(seq, indices):
            if idx == 1:
                if BOS_EOS and len(ret) > 0:
                    ret[-1].append(self.x2i['EOS'])
                ret.append([])
                if BOS_EOS:
                    ret[-1].append(self.x2i['BOS'])
            if type(elem) is list:
                for c in elem:
                    ret[-1].append(self.x2i[c] if c in self.x2i else self.x2i['UNK'])
            else:
                ret[-1].append(self.x2i[elem] if elem in self.x2i else self.x2i['UNK'])

        if BOS_EOS:
            ret[-1].append(self.x2i['EOS'])

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
        assert (self._pret_file is not None), "No pretrained file provided."
        embs = [[]] * len(self.i2x)  # make list of empty list of length of id2word
        with open(self._pret_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line:
                    word, data = line[0], line[1:]
                    embs[self.x2i[word]] = data  #
        emb_size = len(data)
        for idx, emb in enumerate(embs):  # assign zero vector if a word is not provided with pretrained embedding
            if not emb:
                embs[idx] = np.zeros(emb_size)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(
            pret_embs)  # return an array of pretrained embeddings normalized by standard deviation


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


def out2file():
    orig_out = sys.stdout
    filename = 'dev_result.txt' if not config.isTest else 'test_result.txt'
    f = open(filename, 'w')
    sys.stdout = f


def align_word_char(bi_word):
    c2w, w2c = [0], [0]
    widx = 1
    for idx in range(len(bi_word)):
        if bi_word[idx] == 0:
            cidx = idx + 1
            w2c.append(cidx)
            if idx != 0:
                widx += 1
        c2w.append(widx)

    return c2w, w2c


def align_deps(heads, bi_word):
    tmp = []
    heads_char = []

    c2w, w2c = align_word_char(bi_word)

    for h in heads:
        tmp.append(w2c[h])

    widx = 0

    for idx in range(len(bi_word)):
        if bi_word[idx] == 0:
            heads_char.append(tmp[widx])
            widx += 1
            cidx = idx
        else:
            heads_char.append(cidx + 1)

    return heads_char
