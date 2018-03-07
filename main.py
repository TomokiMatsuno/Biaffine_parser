# library with acronym
import pandas as pd
import numpy as np
import dynet as dy
from collections import Counter

# public library
import glob

# my library
import paths
import utils
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
# col_indices = [0, 1, 3, 6, 7]

sents_train, sents_val = pre.files2sents(file_train, col_indices), pre.files2sents(file_val, col_indices)
if config.num_sents > 0:
    sents_train, sents_val = pre.files2sents(file_train, col_indices), pre.files2sents(file_train, col_indices)
# dicts = pre.sents2dicts(sents_train, sents_val, prets=(paths.pret_file, 1))
dicts = pre.sents2dicts(sents_train, sents_val, initial_entries=config.initial_entries)
indices = [sents_train[0], sents_val[0]]
ids_train, ids_val = pre.sents2ids(sents_train, dicts, indices[0], no_dict=[0, 3, 5]), pre.sents2ids(sents_val, dicts, indices[1], no_dict=[0, 3, 5])

wd, td, rd, bid = dicts[1], dicts[2], dicts[4], dicts[5]
# wd, td, rd = dicts[1], dicts[2], dicts[4]
word_ids, tag_ids, head_ids, rel_ids, bi_ids = [[], []], [[], []], [[], []], [[], []], [[], []]
word_ids[0], tag_ids[0], head_ids[0], rel_ids[0] = ids_train[1], ids_train[2], ids_train[3], ids_train[4]
word_ids[1], tag_ids[1], head_ids[1], rel_ids[1] = ids_val[1], ids_val[2], ids_val[3], ids_val[4]

# bi_ids[0], bi_ids[1] = [[0 if bi == 'B' else 1 for bi in bi_sent] for bi_sent in ids_train[5]], [[0 if bi == 'B' else 1 for bi in bi_sent] for bi_sent in ids_val[5]]
bi_ids = [[], []]

func = [rd.x2i[r] for r in ['aux', 'case', 'auxpass', 'punct', 'cop', 'mark', 'neg', 'mwe', 'cc']]
func_begin = [wd.x2i[w] for w in ['「', '（', '“', '『']]
func_end = [wd.x2i[w] for w in ['」', '）', '”', '』']]
indp = [rd.x2i[r] for r in ['advcl', 'advmod']]
prefix = [rd.x2i[r] for r in ['compound', 'nummod']]
subob =  [rd.x2i[r] for r in ['nsubj', 'iobj', 'dobj', 'nsubjpass']]
posfunc =  [td.x2i[t] for t in ['ADV', 'ADP', 'AUX']]
poscont =  [td.x2i[t] for t in ['NOUN', 'PROPN']]
for step in range(len(rel_ids)):
    for i in range(len(rel_ids[step])):
        bi_ids[step].append(utils.chunk_tags(rel_ids[step][i], func, word_ids[step][i], func_begin, func_end, indp, prefix, subob, posfunc, poscont, tag_ids[step][i], td, rd))

for step in range(len(bi_ids)):
    for i in range(len(bi_ids[step])):
        bi_ids[step][i] = utils.re_chunk(head_ids[step][i], bi_ids[step][i])

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
parser._mwe_id = rd.x2i['mwe']
parser._acl_id = rd.x2i['acl']
parser._conj_id = rd.x2i['conj']
parser._case_id = rd.x2i['case']
parser._nmod_id = rd.x2i['nmod']

num_multi_head_chunk = 0
num_conj = 0
num_mwe = 0
num_punct = 0

# for step in range(len(word_ids)):
#     word_ids[step], tag_ids[step], head_ids[step], rel_ids[step], bi_ids[step] = utils.omit_invalid_sents(word_ids[step], tag_ids[step], head_ids[step], rel_ids[step], bi_ids[step],
#                                                                             B_tag_id=0, punct_id=rd.x2i['punct'])
bi_ids_new = [[], []]
difficult_sents = [347] # [347, 370]
for step in range(len(bi_ids)):
    for id in range(len(bi_ids[step])):
        seq_bi, seq_h, seq_r = bi_ids[step][id], head_ids[step][id], rel_ids[step][id]
        w2ch = utils.align_word_chunk(seq_bi, parser._B_tag_id, with_root=False)

        heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(seq_h, [0] + seq_bi, parser._B_tag_id, seq_r,
                                                                      parser._punct_id)
        back = utils.word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0]+seq_bi,
                              B_tag_idx=parser._B_tag_id)

        chunk_ranges = utils.ranges([0] + seq_bi, parser._B_tag_id)
        seq_h = [0] + seq_h
        seq_r = [0] + seq_r

        for r in chunk_ranges[1:]:
            start = r[0]
            end = r[1]
            set_head_rel = set()
            set_head_par = set()
            list_rel = []
            for i in range(start, end):
                if seq_h[i] < start or end <= seq_h[i]:
                    if w2ch[seq_h[i]] not in set_head_par:
                        set_head_par.add(w2ch[seq_h[i]])
                        list_rel.append(seq_r[i])
            if len(set_head_par) > 1 and parser._punct_id not in list_rel and id not in difficult_sents:
            # if len(set_head_par) > 1:
                print(list_rel)
                for rel in list_rel:
                    print(rd.i2x[rel], end='\t')
                print('')
                num_multi_head_chunk += 1

                if parser._punct_id not in list_rel:
                    num_punct += 1

                if rd.x2i['conj'] in list_rel:
                    num_conj += 1
                if rd.x2i['mwe'] in list_rel:
                    num_mwe += 1


print('num_multi_head_chunk ', num_multi_head_chunk)
print('num_conj', num_conj)
print('num_mwe', num_mwe)
print('num_punct', num_punct)

print('')


# bi_ids[0] = [utils.convertChunks(bi_id, head_id, parser._B_tag_id) for bi_id, head_id in zip(bi_ids[0], head_ids[0])]
# bi_ids[1] = [utils.convertChunks(bi_id, head_id, parser._B_tag_id) for bi_id, head_id in zip(bi_ids[1], head_ids[1])]
# tot_failed_before = 0
# tot_failed_after = 0
# new_bi = []
# for bi_idx in range(len(bi_ids[0])):
#     seq_bi, seq_h, seq_r, seq_t = bi_ids[0][bi_idx], head_ids[0][bi_idx], rel_ids[0][bi_idx], tag_ids[0][bi_idx]
#     punct_mask = np.array([1 if rel != parser._punct_id else 0 for rel in seq_r])
#     num_punct = np.sum(np.equal(punct_mask, np.zeros(len(punct_mask))))
#
#     heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(seq_h, [0] + seq_bi, parser._B_tag_id, seq_r,
#                                                                   parser._punct_id)
#     back = utils.word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0]+seq_bi,
#                           B_tag_idx=parser._B_tag_id)
#
#     if len(seq_h) - num_punct != np.sum(np.multiply(np.equal(back, seq_h), punct_mask)):
#         tot_failed_before += 1
#         # continue
#         new_bi.append(utils.convertChunks(bi_ids[0][bi_idx], head_ids[0][bi_idx], seq_r, seq_t, parser, rd, td))
#     else:
#         new_bi.append(seq_bi)
#
#     heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(seq_h, [0] + new_bi[-1], parser._B_tag_id, seq_r,
#                                                                   parser._punct_id)
#     back = utils.word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0] + new_bi[-1],
#                           B_tag_idx=parser._B_tag_id)
#
#
#     # if len(back) != len(seq_h):
#     #     print(bi_idx)
#     #     tot_failed_after += 1
#     #     continue
#
#     if len(seq_h) - num_punct != np.sum(np.multiply(np.equal(back, seq_h), punct_mask)):
#         tot_failed_after += 1
#         # continue



# del bi_ids[0][:]
# bi_ids[0] = new_bi.copy()
#
# tot_failed_before = 0
# tot_failed_after = 0
#
# new_bi2 = []
#
# for bi_idx in range(len(bi_ids[1])):
#     seq_bi, seq_h, seq_r, seq_t = bi_ids[1][bi_idx], head_ids[1][bi_idx], rel_ids[1][bi_idx], tag_ids[1][bi_idx]
#     heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(seq_h, [0] + seq_bi, parser._B_tag_id, seq_r,
#                                                                   parser._punct_id)
#     punct_mask = np.array([1 if rel != parser._punct_id else 0 for rel in seq_r])
#     num_punct = np.sum(np.equal(punct_mask, np.zeros(len(punct_mask))))
#
#     back = utils.word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0]+seq_bi,
#                           B_tag_idx=parser._B_tag_id)
#
#     if len(seq_h) - num_punct != np.sum(np.multiply(np.equal(back, seq_h), punct_mask)):
#         tot_failed_before += 1
#         new_bi2.append(utils.convertChunks(seq_bi, seq_h, seq_r, seq_t, parser, rd, td))
#         # continue
#
#
#     # new_bi2.append(utils.convertChunks(bi_ids[1][bi_idx], head_ids[1][bi_idx], seq_r, parser, rd))
#
#
#     heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(seq_h, [0] + new_bi2[-1], parser._B_tag_id, seq_r,
#                                                                   parser._punct_id)
#     back = utils.word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0] + new_bi2[-1],
#                           B_tag_idx=parser._B_tag_id)
#
#     if len(seq_h) - num_punct != np.sum(np.multiply(np.equal(back, seq_h), punct_mask)):
#         tot_failed_after += 1
#         # continue



# del bi_ids[1][:]
# bi_ids[1] = new_bi2.copy()

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

    tot_cor_bi = 0
    tot_cor_inter = 0
    tot_inter = 0
    tot_num_suc = 0
    tot_failed = 0

    step = 0
    parser._pdrop = config.pdrop * isTrain
    parser._pdrop_embs = config.pdrop_embs * isTrain
    # parser.embd_mask_generator(parser._pdrop_embs, indices)

    sent_ids = [i for i in range(len(word_ids))]

    if config.num_sents > 0:
        sent_ids = sent_ids[:config.num_sents]

    if isTrain and config.random_pickup and not config.small_data:
        np.random.shuffle(sent_ids)

    # for seq_w, seq_t, seq_h, seq_r in zip(word_ids, tag_ids, head_ids, rel_ids):
    for sent_id in sent_ids:

        seq_w, seq_t, seq_h, seq_r, seq_bi = word_ids[sent_id], tag_ids[sent_id], \
                                                               head_ids[sent_id], rel_ids[sent_id], bi_ids[sent_id]

        seq_bi = [parser._B_tag_id] + seq_bi

        punct_count = 0

        for r in seq_r:
            if r == parser._punct_id:
                punct_count += 1

        if not isTrain:
            dy.renew_cg()

        loss, num_cor_arc, num_cor_arc_intra, num_arc_intra, num_cor_rel, cor_rels, gold_rels, preds_rels, cor_bi, cor_inter, num_inter, num_suc = parser.run(seq_w, seq_t, seq_h, seq_r, seq_bi, isTrain)
        losses_batch.append(dy.sum_batches(loss))

        # tot_tokens += len(seq_w) - punct_count
        tot_tokens += len(seq_w)
        tot_arc_intra += num_arc_intra
        tot_cor_arc += num_cor_arc
        tot_cor_arc_intra += num_cor_arc_intra
        tot_cor_arc_inter += num_cor_arc - num_cor_arc_intra
        tot_cor_rel += num_cor_rel
        tot_cor_bi += cor_bi - 1
        tot_cor_inter += cor_inter
        tot_inter += num_inter
        tot_num_suc += num_suc

        if not isTrain:
            for elem in cor_rels:
                cnt_cor_rel[elem] += 1
            for elem in gold_rels:
                cnt_gold_rel[elem] += 1
            for elem in preds_rels:
                cnt_preds_rel[elem] += 1

        step += 1

        if (step % config.batch_size == 0 or step == len(word_ids) - 1) and isTrain:
            losses = dy.esum(losses_batch)
            losses_value_arc = losses.value()
            losses.backward()
            parser.update_parameters()
            if step == len(word_ids) - 1:
                print(losses_value_arc)
            losses_batch = []

            dy.renew_cg()
            parser._global_step += 1


    if (not isTrain):
        assert tot_cor_arc <= tot_tokens, "tot_cor_arc > tot_tokens"
        score = (tot_cor_arc / tot_tokens)
        assert tot_cor_rel <= tot_tokens, "tot_cor_rel > tot_tokens"
        score_label = (tot_cor_rel / tot_tokens)

        # score_inter = (tot_cor_arc_inter / (tot_tokens - tot_arc_intra))
        # score_intra = (tot_cor_arc_intra / tot_arc_intra)
        # preprocess.print2filecons(str(score))
        # preprocess.print2filecons(str(score_label))
        print('UAS:\t', score, end='\t')
        print('LAS:\t', score_label, end='\t')
        # print('inter:\t', score_inter, end='\t')
        # print('intra:\t', score_intra)
        print('chunk dep:\t', tot_cor_inter / tot_inter)
        if score > parser._best_score:
            parser._update = True
            parser._early_stop_count = 0
            parser._best_score = score

        if score_label > parser._best_score_las:
            parser._best_score_las = score_label

        preprocess.print2filecons(str(parser._best_score))
        preprocess.print2filecons(str(parser._best_score_las))

        # if config.las:
        #     for ri in range(len(rd.i2x)):
        #         print(rd.i2x[ri], end='\t')
        #         print('recall', cnt_cor_rel[ri] / (cnt_gold_rel[ri] if cnt_gold_rel[ri] != 0 else 1.), end='\t')
        #         print('precision', cnt_cor_rel[ri] / (cnt_preds_rel[ri] if cnt_preds_rel[ri] != 0 else 1.), end='\t')
        #         print('f1', (2 * cnt_cor_rel[ri] / (cnt_gold_rel[ri] + cnt_preds_rel[ri])) if (cnt_gold_rel[ri] + cnt_preds_rel[ri]) else 0.)

        print('cor bi', tot_cor_bi / tot_tokens)
        # print('suc rate', tot_num_suc / tot_tokens)

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
