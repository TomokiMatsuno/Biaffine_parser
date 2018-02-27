# library with acronym
import pandas as pd
import numpy as np
import dynet as dy
from collections import Counter

# public library
import glob

# my library
import paths
import preprocess
import preprocess as pre
import parser
import config
import timer

timer = timer.Timer()

file_train = paths.train_file
file_val = paths.test_file if config.isTest else paths.dev_file
# file_val = paths.dev_file if config.isTest else paths.dev_file

col_indices = [0, 1, 3, 6, 7, 10] if config.japanese else [0, 1, 3, 6, 7]

sents_train, sents_val = pre.files2sents(file_train, col_indices), pre.files2sents(file_val, col_indices)
if config.num_sents > 0:
    sents_train, sents_val = pre.files2sents(file_train, col_indices), pre.files2sents(file_train, col_indices)
# dicts = pre.sents2dicts(sents_train, sents_val, prets=(paths.pret_file, 1))
dicts = pre.sents2dicts(sents_train, sents_val, initial_entries=config.initial_entries)
indices = [sents_train[0], sents_val[0]]
ids_train, ids_val = pre.sents2ids(sents_train, dicts, indices[0], no_dict=[0, 3, 5]), pre.sents2ids(sents_val, dicts, indices[1], no_dict=[0, 3, 5])

wd, td, rd, bid = dicts[1], dicts[2], dicts[4], dicts[5]
word_ids, tag_ids, head_ids, rel_ids, bi_ids = [[], []], [[], []], [[], []], [[], []], [[], []]
word_ids[0], tag_ids[0], head_ids[0], rel_ids[0] = ids_train[1], ids_train[2], ids_train[3], ids_train[4]
word_ids[1], tag_ids[1], head_ids[1], rel_ids[1] = ids_val[1], ids_val[2], ids_val[3], ids_val[4]

bi_ids[0], bi_ids[1] = [[0 if bi == 'B' else 1 for bi in bi_sent] for bi_sent in ids_train[5]], [[0 if bi == 'B' else 1 for bi in bi_sent] for bi_sent in ids_val[5]]

# embs_word = wd.get_pret_embs()
embs_word = None

parser = parser.Parser(
    len(wd.i2x),
    len(td.i2x),
    len(rd.i2x),
    config.input_dim,
    config.hidden_dim,
    config.pdrop,
    config.pdrop_embs,
    config.pdrop_lstm,
    config.layers,
    config.mlp_dim,
    config.arc_dim,
    config.biaffine_bias_x_arc,
    config.biaffine_bias_y_arc,
    config.biaffine_bias_x_rel,
    config.biaffine_bias_y_rel,
    embs_word
)

parser._B_tag_id = 0

parser._punct_id = rd.x2i['punct']

# losses_batch = []
def train_dev(word_ids, tag_ids, head_ids, rel_ids, bi_ids, indices, isTrain):
    losses_batch = []
    tot_tokens = 0
    tot_arc_intra = 0
    tot_cor_arc = 0
    tot_cor_arc_intra = 0
    tot_cor_arc_inter = 0
    tot_cor_rel = 0
    cnt_cor_rel = Counter()
    cnt_gold_rel = Counter()
    cnt_preds_rel = Counter()

    step = 0
    parser._pdrop = config.pdrop * isTrain
    parser._pdrop_embs = config.pdrop_embs * isTrain
    parser.embd_mask_generator(parser._pdrop_embs, indices)

    sent_ids = [i for i in range(len(word_ids))]
    if isTrain and config.random_pickup and not config.small_data:
        np.random.shuffle(sent_ids)

    # for seq_w, seq_t, seq_h, seq_r in zip(word_ids, tag_ids, head_ids, rel_ids):
    for sent_id in sent_ids:

        seq_w, seq_t, seq_h, seq_r, seq_bi, masks_w, masks_t = word_ids[sent_id], tag_ids[sent_id], \
                                                       head_ids[sent_id], rel_ids[sent_id], bi_ids[sent_id], \
                                                       parser._masks_w[sent_id], parser._masks_t[sent_id]
        seq_bi = [parser._B_tag_id] + seq_bi
        if not isTrain:
            dy.renew_cg()

        loss, num_cor_arc, num_cor_arc_intra, num_arc_intra, num_cor_rel, cor_rels, gold_rels, preds_rels\
            = parser.run(seq_w, seq_t, seq_h, seq_r, seq_bi, masks_w, masks_t, isTrain)
        losses_batch.append(dy.sum_batches(loss))

        # punct_count = 0
        #
        # for r in seq_r:
        #     if r == parser._punct_id:
        #         punct_count += 1
        #
        # tot_tokens += len(seq_w) - punct_count
        tot_tokens += len(seq_w)
        tot_arc_intra += num_arc_intra
        tot_cor_arc += num_cor_arc
        tot_cor_arc_intra += num_cor_arc_intra
        tot_cor_arc_inter += num_cor_arc - num_cor_arc_intra
        tot_cor_rel += num_cor_rel

        if not isTrain:
            for elem in cor_rels:
                cnt_cor_rel[elem] += 1
            for elem in gold_rels:
                cnt_gold_rel[elem] += 1
            for elem in preds_rels:
                cnt_preds_rel[elem] += 1
        step += 1

        if (step % config.batch_size == 0 or step == len(word_ids) - 1) and isTrain:
            # print(step, "\t/\t", len(sent_ids), flush=True)
            losses = dy.esum(losses_batch)
            losses_value_arc = losses.value()
            losses.backward()
            parser.update_parameters()
            if step == len(word_ids) - 1:
                print(losses_value_arc)
            losses_batch = []

            dy.renew_cg()
            parser._global_step += 1

        if (not isTrain) and step == len(word_ids):
            assert tot_cor_arc <= tot_tokens, "tot_cor_arc > tot_tokens"
            score = (tot_cor_arc / tot_tokens)
            assert tot_cor_rel <= tot_tokens, "tot_cor_rel > tot_tokens"
            score_label = (tot_cor_rel / tot_tokens)

            score_inter = (tot_cor_arc_inter / (tot_tokens - tot_arc_intra))
            score_intra = (tot_cor_arc_intra / tot_arc_intra)
            # preprocess.print2filecons(str(score))
            # preprocess.print2filecons(str(score_label))
            print('UAS:\t', score, end='\t')
            print('LAS:\t', score_label, end='\t')
            print('inter:\t', score_inter, end='\t')
            print('intra:\t', score_intra)
            if score > parser._best_score:
                parser._update = True
                parser._early_stop_count = 0
                parser._best_score = score

            if score_label > parser._best_score_las:
                parser._best_score_las = score_label

            preprocess.print2filecons(str(parser._best_score))
            preprocess.print2filecons(str(parser._best_score_las))

            for ri in range(len(rd.i2x)):
                print(rd.i2x[ri], end='\t')
                print('recall', cnt_cor_rel[ri] / (cnt_gold_rel[ri] if cnt_gold_rel[ri] != 0 else 1.), end='\t')
                print('precision', cnt_cor_rel[ri] / (cnt_preds_rel[ri] if cnt_preds_rel[ri] != 0 else 1.), end='\t')
                print('f1', (2 * cnt_cor_rel[ri] / (cnt_gold_rel[ri] + cnt_preds_rel[ri])) if (cnt_gold_rel[ri] + cnt_preds_rel[ri]) else 0.)



    return

timer.from_prev()

for e in range(config.epoc):
    preprocess.print2filecons("epoc: " + str(e))

    parser._update = False
    if config.isTest:
        parser._pc.populate(paths.save_file_directory + config.load_file + str(e))
        preprocess.print2filecons("populated from:\t" + paths.save_file_directory + config.load_file + str(e))
    else:
        isTrain = True
        train_dev(word_ids[0], tag_ids[0], head_ids[0], rel_ids[0], bi_ids[0], indices[0], isTrain)
        timer.from_prev()

    isTrain = False
    if not config.small_data:
        train_dev(word_ids[1], tag_ids[1], head_ids[1], rel_ids[1], bi_ids[1], indices[1], isTrain)
    else:
        train_dev(word_ids[0], tag_ids[0], head_ids[0], rel_ids[0], bi_ids[0], indices[0], isTrain)
    timer.from_prev()

    if config.save and not config.isTest:
        if e == 0:
            dir_save = preprocess.make_dir(paths.save_file_directory)
            preprocess.save_codes(dir_save)
        parser._pc.save(dir_save + "/" + config.save_file + str(parser._early_stop_count))
        preprocess.print2filecons("saved into: " + dir_save + "/" + config.save_file + str(parser._early_stop_count))

    parser._early_stop_count += 1

    if parser._early_stop_count > config.early_stop:
        break

print("succeed")
