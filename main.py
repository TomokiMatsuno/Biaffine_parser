# library with acronym
import pandas as pd
import numpy as np
import dynet as dy

# public library
import glob

# my library
import paths
import preprocess
import parser
import config
import timer

timer = timer.Timer()
# preprocess.out2file()

# files_train = glob.glob(paths.path2WSJ + '00/*')
if config.chinese:
    files_train = glob.glob(paths.path2CTB + 'train/*.tab')
    if not config.isTest:
        files_dev = glob.glob(paths.path2CTB + 'dev/*.tab')
        # files_dev = glob.glob(paths.path2CTB + 'train/*.tab')
    else:
        files_dev = glob.glob(paths.path2CTB + 'test/*.tab')

    indices, words, tags, heads, rels = preprocess.files2sents_ch(files_train)
    indices_dev, words_dev, tags_dev, heads_dev, rels_dev = preprocess.files2sents_ch(files_dev)
else:
    files_train = [f for f in glob.glob(paths.path2WSJ + '*/*')
                   if not ("/01" in f or "/22" in f or "/23" in f or "/24" in f or "section" in f or "predPOS" in f)]
    # files_train = [f for f in glob.glob(paths.path2WSJ + '0[2-9]/*')]
    if not config.isTest:
        files_dev = glob.glob(paths.path2WSJ + 'wsj_predPOS/23*/*')
    else:
        files_dev = [f for f in glob.glob(paths.path2WSJ + 'wsj_predPOS/*/*') if not "/23" in f]

    # df_train = preprocess.files2DataFrame(files_train, '\t')
    # df_dev = preprocess.files2DataFrame(files_dev, '\t')
    df_train = preprocess.files2DataFrame(files_train, '\t')
    df_dev = preprocess.files2DataFrame(files_dev, '\t')

    indices, words, tags, heads, rels = \
        df_train[0].tolist(), \
        [w.lower() for w in df_train[1].tolist()], \
        df_train[3].tolist(), \
        df_train[6].tolist(), \
        df_train[7].tolist()

    indices_dev, words_dev, tags_dev, heads_dev, rels_dev = \
        df_dev[0].tolist(), \
        [w.lower() for w in df_dev[1].tolist()], \
        df_dev[10].tolist(), \
        df_dev[6].tolist(), \
        df_dev[7].tolist()


# wd, td, rd = \
#     preprocess.Dictionary(words, paths.pret_file), \
#     preprocess.Dictionary(tags), \
#     preprocess.Dictionary(rels)

if config.chinese:
    wd, td, rd = \
        preprocess.Dictionary(words), \
        preprocess.Dictionary(tags), \
        preprocess.Dictionary(rels)
    embs_word = None
else:
    wd, td, rd = \
        preprocess.Dictionary(words, paths.pret_file), \
        preprocess.Dictionary(tags), \
        preprocess.Dictionary(rels)
    embs_word = wd.get_pret_embs()

chars, indices_char = preprocess.tochar(words, tags, indices)
chars_dev, indices_char_dev = preprocess.tochar(words_dev, tags_dev, indices_dev)

char_seq = []
for sent in chars:
    char_seq.extend(sent)

cd = preprocess.Dictionary(char_seq)

char_seq_dev = []
for sent in chars_dev:
    char_seq_dev.extend(sent)

cd.add_entries(char_seq_dev)
char_ids, char_ids_dev = cd.sent2ids(char_seq, indices_char), cd.sent2ids(char_seq_dev, indices_char_dev)


tags_char, _ = preprocess.tochar(words, tags, indices, isTag=True)
tags_char_dev, _ = preprocess.tochar(words_dev, tags_dev, indices_dev, isTag=True)

tags_char_seq = []
for sent in tags_char:
    tags_char_seq.extend(sent)
tags_char_seq_dev = []
for sent in tags_char_dev:
    tags_char_seq_dev.extend(sent)

tag_ids_char, tag_ids_char_dev = td.sent2ids(tags_char_seq, indices_char), td.sent2ids(tags_char_seq_dev, indices_char_dev)


bis_word = []
bis_word_dev = []
for w in words:
    bis_word.extend([0] + [1] * (len(w) - 1))
# bis_word.append(1)
for w in words_dev:
    bis_word_dev.extend([0] + [1] * (len(w) - 1))


# bis_word_dev = [bis_word_dev.extend([0] + [1] * (len(w) - 1)) for w in words_dev]

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
bis_word = preprocess.seq2ids(bis_word, indices_char)
bis_word_dev = preprocess.seq2ids(bis_word_dev, indices_char_dev)


head_ids_char = []
for head_id, bi_word in zip(head_ids, bis_word):
    head_ids_char.append(preprocess.align_deps(head_id, bi_word))

head_ids_char_dev = []
for head_id, bi_word in zip(head_ids_dev, bis_word_dev):
    head_ids_char_dev.append(preprocess.align_deps(head_id, bi_word))



parser = parser.Parser(
    len(cd.i2x),
    len(wd.i2x),
    len(td.i2x),
    len(rd.i2x),
    config.input_dim,
    config.hidden_dim,
    config.pdrop,
    config.pdrop_embs,
    config.layers,
    config.mlp_dim,
    config.arc_dim,
    config.biaffine_bias_x_arc,
    config.biaffine_bias_y_arc,
    config.biaffine_bias_x_rel,
    config.biaffine_bias_y_rel,
    embs_word
)

if config.chinese:
    punct = 'P'
else:
    punct = 'punct'

parser._punct_id = rd.x2i[punct]

punct_count = 0
for r in rels:
    if r == punct:
        punct_count += 1

print(punct_count / len(rels))


def train_dev(char_ids, tag_ids_char, word_ids, tag_ids, bis_word, head_ids, rel_ids, indices, isTrain):
    losses = []
    tot_tokens = 0
    tot_cor_arc = 0
    tot_cor_arc_align = 0
    tot_cor_rel = 0

    step = 0
    parser._pdrop = config.pdrop * isTrain
    parser._pdrop_embs = config.pdrop_embs * isTrain
    parser.embd_mask_generator(parser._pdrop_embs, indices)

    sent_ids = [i for i in range(len(word_ids))]
    if isTrain:
        np.random.shuffle(sent_ids)

    # for seq_w, seq_t, seq_h, seq_r in zip(word_ids, tag_ids, head_ids, rel_ids):
    for sent_id in sent_ids:
        seq_c, seq_t_char, seq_w, seq_t, bi_word, seq_h, seq_r, masks_w, masks_t = char_ids[sent_id], tag_ids_char[sent_id],\
                                                       word_ids[sent_id], tag_ids[sent_id], bis_word[sent_id], \
                                                       head_ids[sent_id], rel_ids[sent_id], \
                                                       parser._masks_w[sent_id], parser._masks_t[sent_id]
        # # add BOS and EOS
        # bi_word = [0] + bi_word + [0]
        # seq_c = [1] + seq_c + [2]
        # seq_t_char = [1] + seq_t_char + [2]

        if not isTrain:
            dy.renew_cg()

        # loss, num_cor_arc, num_cor_arc_align, num_cor_rel = parser.run(seq_c, seq_t_char, seq_w, seq_t, bi_word, seq_h, seq_r, masks_w, masks_t, isTrain)
        loss, num_cor_arc, num_cor_rel = parser.run(seq_c, seq_t_char, seq_w, seq_t, bi_word, seq_h, seq_r, masks_w, masks_t, isTrain)

        if isTrain:
            losses.append(dy.sum_batches(loss))

        # punct_count = 0
        #
        # for r in seq_r:
        #     if r == parser._punct_id:
        #         punct_count += 1
        #
        # tot_tokens += len(seq_w) - punct_count
        tot_tokens += len(seq_c)
        tot_cor_arc += num_cor_arc
        # tot_cor_arc_align += num_cor_arc_align
        tot_cor_rel += num_cor_rel

        step += 1

        if (step % config.batch_size == 0 or step == len(word_ids) - 1) and isTrain:
            # print(step, "\t/\t", len(sent_ids), flush=True)
            losses = dy.esum(losses)
            losses_value = losses.value()
            losses.backward()
            # parser._trainer.update()
            parser.update_parameters()
            if step == len(word_ids) - 1:
                print(losses_value)
            losses = []
            dy.renew_cg()
            parser._global_step += 1

        if (not isTrain) and step == len(word_ids) - 1:
            score = (tot_cor_arc / tot_tokens)
            # score_align = (tot_cor_arc_align / tot_tokens)
            score_label = (tot_cor_rel / tot_tokens)
            print(score)
            # print(score_align)
            print(score_label)
            if score > parser._best_score:
                parser._update = True
                parser._early_stop_count = 0
                parser._best_score = score

            if score_label > parser._best_score_las:
                parser._best_score_las = score_label

            print(parser._best_score)
            print(parser._best_score_las)

timer.from_prev()

for e in range(config.epoc):
    print("epoc: ", e)

    parser._update = False
    if config.isTest:
        parser._pc.populate(paths.save_file_directory + config.load_file + str(e))
        print("populated from:\t", paths.save_file_directory + config.load_file + str(e))
    else:
        parser.isTrain = True
        train_dev(char_ids, tag_ids_char, word_ids, tag_ids,
                  bis_word, head_ids_char, rel_ids, indices, parser.isTrain)
        timer.from_prev()

    parser.isTrain = False
    train_dev(char_ids_dev, tag_ids_char_dev, word_ids_dev, tag_ids_dev,
              bis_word_dev, head_ids_char_dev, rel_ids_dev, indices_dev, parser.isTrain)
    timer.from_prev()

    if config.save and not config.isTest:
        if e == 0:
            dir_save = preprocess.make_dir(paths.save_file_directory)
            preprocess.save_codes(dir_save)
        parser._pc.save(dir_save + "/" + config.save_file + str(parser._early_stop_count))
        print("saved into: ", dir_save + "/" + config.save_file + str(parser._early_stop_count))

    parser._early_stop_count += 1

    if parser._early_stop_count > config.early_stop:
        break

print("succeed")
