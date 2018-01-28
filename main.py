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


wd, td, rd = \
    preprocess.Dictionary(words, paths.pret_file), \
    preprocess.Dictionary(tags), \
    preprocess.Dictionary(rels)

embs_word = wd.get_pret_embs()

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

parser._punct_id = rd.x2i['punct']

punct_count = 0
for r in rels:
    if r == 'punct':
        punct_count += 1

print(punct_count / len(rels))


def train_dev(word_ids, tag_ids, head_ids, rel_ids, indices, isTrain):
    losses = []
    tot_tokens = 0
    tot_cor_arc = 0
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

        seq_w, seq_t, seq_h, seq_r, masks_w, masks_t = word_ids[sent_id], tag_ids[sent_id], \
                                                       head_ids[sent_id], rel_ids[sent_id], \
                                                       parser._masks_w[sent_id], parser._masks_t[sent_id]

        # if step % config.batch_size == 0 or not isTrain:
        if not isTrain:
            dy.renew_cg()

        loss, num_cor_arc, num_cor_rel = parser.run(seq_w, seq_t, seq_h, seq_r, masks_w, masks_t, isTrain)
        losses.append(dy.sum_batches(loss))

        punct_count = 0

        for r in seq_r:
            if r == parser._punct_id:
                punct_count += 1

        tot_tokens += len(seq_w) - punct_count
        tot_cor_arc += num_cor_arc
        tot_cor_rel += num_cor_rel

        step += 1

        if (step % config.batch_size == 0 or step == len(word_ids) - 1) and isTrain:
            # print(step, "\t/\t", len(sent_ids), flush=True)
            losses = dy.esum(losses)
            losses_value_arc = losses.value()
            losses.backward()
            # parser._trainer.update()
            parser.update_parameters()
            if step == len(word_ids) - 1:
                print(losses_value_arc)
            losses = []
            dy.renew_cg()
            parser._global_step += 1

        if (not isTrain) and step == len(word_ids) - 1:
            score = (tot_cor_arc / tot_tokens)
            score_label = (tot_cor_rel / tot_tokens)
            print(score)
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
        isTrain = True
        train_dev(word_ids, tag_ids, head_ids, rel_ids, indices, isTrain)
        timer.from_prev()

    isTrain = False
    train_dev(word_ids_dev, tag_ids_dev, head_ids_dev, rel_ids_dev, indices_dev, isTrain)
    timer.from_prev()

    if not config.isTest:
        if e == 0:
            dir_save = preprocess.make_dir(paths.save_file_directory)
            preprocess.save_codes(dir_save)
        parser._pc.save(dir_save + "/" + config.save_file + str(parser._early_stop_count))
        print("saved into: ", dir_save + "/" + config.save_file + str(parser._early_stop_count))

    parser._early_stop_count += 1

    if parser._early_stop_count > config.early_stop:
        break

print("succeed")
