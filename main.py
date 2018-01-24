#library with acronym
import pandas as pd
import numpy as np
import dynet as dy

#public library
import glob

#my library
import paths
import preprocess
import parser
import config

files_train = glob.glob(paths.path2WSJ + '00/*')
files_dev = glob.glob(paths.path2WSJ + '23/*')

df_train = preprocess.files2DataFrame(files_train[:90], '\t')
# df_dev = preprocess.files2DataFrame(files_dev, '\t')
df_dev = preprocess.files2DataFrame(files_train[90:], '\t')

indices, words, tags, heads, rels = \
    df_train[0].tolist(), \
    df_train[1].tolist(), \
    df_train[3].tolist(), \
    df_train[6].tolist(), \
    df_train[7].tolist()

indices_dev, words_dev, tags_dev, heads_dev, rels_dev = \
    df_dev[0].tolist(), \
    df_dev[1].tolist(), \
    df_dev[3].tolist(), \
    df_dev[6].tolist(), \
    df_dev[7].tolist()

wd, td, rd = \
    preprocess.Dictionary(words), \
    preprocess.Dictionary(tags), \
    preprocess.Dictionary(rels)

wd.add_entries(words_dev)
td.add_entries(tags_dev)
rd.add_entries(rels_dev)

word_ids, tag_ids, rel_ids = \
    wd.sent2ids(words, indices), \
    td.sent2ids(tags, indices), \
    rd.sent2ids(rels, indices)

word_ids_dev, tag_ids_dev, rel_ids_dev = \
    wd.sent2ids(words_dev, indices_dev), \
    td.sent2ids(tags_dev, indices_dev), \
    rd.sent2ids(rels_dev, indices_dev)

head_ids = preprocess.seq2ids(heads, indices)
head_ids_dev = preprocess.seq2ids(heads_dev, indices_dev)

parser = parser.Parser(
    len(wd.i2x),
    len(td.i2x),
    config.input_dim,
    config.hidden_dim,
    config.pdrop,
    config.layers,
    config.mlp_dim,
    config.arc_dim,
    config.biaffine_bias_x_arc,
    config.biaffine_bias_y_arc,
    config.biaffine_bias_x_rel,
    config.biaffine_bias_y_rel
)


def train_dev(word_ids, tag_ids, head_ids, rel_ids, indices, isTrain):
    losses_arc = []
    tot_arc = 0
    tot_cor_arc = 0
    step = 0
    parser.embd_mask_generator(config.pdrop, indices)

    sent_ids = [i for i in range(len(word_ids))]
    if isTrain:
        np.random.shuffle(sent_ids)

    # for seq_w, seq_t, seq_h, seq_r in zip(word_ids, tag_ids, head_ids, rel_ids):
    for sent_id in sent_ids:
        seq_w, seq_t, seq_h, seq_r, masks_w, masks_t = word_ids[sent_id], tag_ids[sent_id], \
                                                       head_ids[sent_id], rel_ids[sent_id], \
                                                       parser._masks_w[sent_id], parser._masks_t[sent_id]

        if step % config.batch_size == 0 or not isTrain:
            dy.renew_cg()
            losses_arc = []

        loss_arc, preds_arc, num_cor_arc = parser.run(seq_w, seq_t, seq_h, seq_r, masks_w, masks_t, isTrain)
        losses_arc.append(dy.sum_batches(loss_arc))
        tot_arc += len(seq_w)
        tot_cor_arc += num_cor_arc
        step += 1

        if step % config.batch_size == 0 and isTrain:
            losses_arc = dy.esum(losses_arc)
            losses_value_arc = losses_arc.value()
            losses_arc.backward()
            parser._trainer.update()
            print(step)
            print(losses_value_arc)
            losses_arc = []

        if (not isTrain):
            if step % config.show_acc == 0:
                print(tot_cor_arc / tot_arc)

for e in range(config.epoc):
    isTrain = True
    train_dev(word_ids, tag_ids, head_ids, rel_ids, indices, isTrain)
    isTrain = False
    train_dev(word_ids_dev, tag_ids_dev, head_ids_dev, rel_ids_dev, indices_dev, isTrain)



print("succeed")
