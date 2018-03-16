import dynet as dy
import numpy as np

import utils
import config
import preprocess
#TODO: Biaffine word segmentation and POS tagging
#if a segmented span matches a word in vocabulary, the system gets reward?


class Parser(object):
    def __init__(self,
                 char_size,
                 word_size,
                 tag_size,
                 rel_size,
                 input_dim,
                 lstm_dim,
                 pdrop,
                 pdrop_embs,
                 layers,
                 mlp_dim,
                 arc_dim,
                 biaffine_bias_x_arc,
                 biaffine_bias_y_arc,
                 biaffine_bias_x_rel,
                 biaffine_bias_y_rel,
                 embs_word=None
                 ):

        self.isTrain = False

        self._global_step = 0
        self._early_stop_count = 0
        self._update = False
        self._best_score = 0.
        self._best_score_las = 0.

        self._punct_id = 0

        self._masks_w = []
        self._masks_t = []

        self._vocab_size_w = word_size
        self._vocab_size_t = tag_size
        # self._vocab_size_r = rel_size
        self._vocab_size_r = tag_size

        self._mlp_dim = mlp_dim
        self._arc_dim = arc_dim
        # self._rel_dim = mlp_dim - arc_dim
        self._rel_dim = mlp_dim
        self.biaffine_bias_x_arc = biaffine_bias_x_arc
        self.biaffine_bias_y_arc = biaffine_bias_y_arc
        self.biaffine_bias_x_rel = biaffine_bias_x_rel
        self.biaffine_bias_y_rel = biaffine_bias_y_rel

        self._pc = dy.ParameterCollection()

        if config.use_annealing:
            self._trainer = dy.AdamTrainer(self._pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
        else:
            self._trainer = dy.AdadeltaTrainer(self._pc)

        self.params = dict()
        if embs_word is None:
            # self.lp_w = self._pc.add_lookup_parameters((word_size, input_dim), init=dy.ConstInitializer(0.))
            self.lp_w = self._pc.add_lookup_parameters((word_size, input_dim))
        else:
            self.lp_w = self._pc.lookup_parameters_from_numpy(embs_word)
        # self.lp_t = self._pc.add_lookup_parameters((tag_size, input_dim), init=dy.ConstInitializer(0.))
        # self.emb_root = self._pc.add_lookup_parameters((2, input_dim), init=dy.ConstInitializer(0.))
        self.lp_c = self._pc.add_lookup_parameters((char_size, input_dim))
        self.lp_t = self._pc.add_lookup_parameters((tag_size, input_dim))
        self.emb_root = self._pc.add_lookup_parameters((2, lstm_dim * 2))

        self._pdrop = pdrop
        self._pdrop_embs = pdrop_embs

        # self.LSTM_builders_pos = self.LSTM_builders(config.layers_pos, input_dim, lstm_dim)
        # self.LSTM_builders_bi = self.LSTM_builders(config.layers_bi, input_dim * 2 + lstm_dim * 2, lstm_dim)
        # self.LSTM_builders_dep = self.LSTM_builders(config.layers_dep, lstm_dim * 2, lstm_dim)
        self.LSTM_builders_dep = self.LSTM_builders(config.layers_dep, input_dim, lstm_dim)

        W = utils.orthonormal_initializer(mlp_dim, 2 * lstm_dim)
        self.mlp_dep = self._pc.parameters_from_numpy(W)
        self.mlp_head = self._pc.parameters_from_numpy(W)
        # self.mlp_dep_bias = self._pc.add_parameters((mlp_dim,), init=dy.ConstInitializer(0.))
        # self.mlp_head_bias = self._pc.add_parameters((mlp_dim,), init=dy.ConstInitializer(0.))
        self.mlp_dep_bias = self._pc.add_parameters((mlp_dim,))
        self.mlp_head_bias = self._pc.add_parameters((mlp_dim,))

        # self.W_arc = self._pc.add_parameters((self._arc_dim, self._arc_dim + 1),
        #                                      init=dy.ConstInitializer(0.))
        # self.W_rel = self._pc.add_parameters((self._vocab_size_r * (self._rel_dim + 1), self._rel_dim + 1),
        #                                      init=dy.ConstInitializer(0.))
        self.W_arc = self._pc.add_parameters((self._arc_dim, self._arc_dim + 1),
                                             )
        # self.W_rel = self._pc.add_parameters((self._vocab_size_r * (self._rel_dim + 1), self._rel_dim + 1),
        #                                      )
        self.W_rel = self._pc.add_parameters((self._vocab_size_r * (self._mlp_dim + 1),
                                              self._mlp_dim + 1),
                                             )

        self._W_pos_char = self._pc.add_parameters((tag_size, lstm_dim * 2))
        self._bias_pos_char = self._pc.add_parameters((tag_size))
        self._W_pos_word = self._pc.add_parameters((tag_size, lstm_dim * 2))
        self._bias_pos_word = self._pc.add_parameters((tag_size))
        self._W_bi = self._pc.add_parameters((2, lstm_dim * 2))
        self._bias_bi = self._pc.add_parameters((2))

        self.W_seg = self._pc.add_parameters((self._arc_dim + 1, self._arc_dim + 1))
        self.W_tag = self._pc.add_parameters((self._rel_dim + 1, self._rel_dim + 1))

        return

    def LSTM_builders(self, layers, input_dim, lstm_dim):
        LSTM_builders = []
        f = utils.orthonormal_VanillaLSTMBuilder(1, input_dim, lstm_dim, self._pc, isTest=not config.orthonormal)
        b = utils.orthonormal_VanillaLSTMBuilder(1, input_dim, lstm_dim, self._pc, isTest=not config.orthonormal)
        LSTM_builders.append((f, b))
        for i in range(layers - 1):
            f = utils.orthonormal_VanillaLSTMBuilder(1, 2 * lstm_dim, lstm_dim, self._pc, isTest=not config.orthonormal)
            b = utils.orthonormal_VanillaLSTMBuilder(1, 2 * lstm_dim, lstm_dim, self._pc, isTest=not config.orthonormal)
            LSTM_builders.append((f, b))

        return LSTM_builders

    def uniLSTM_builders(self, layers, input_dim, lstm_dim):
        LSTM_builders = []
        f = utils.orthonormal_VanillaLSTMBuilder(1, input_dim, lstm_dim, self._pc, isTest=config.isTest)
        LSTM_builders.append(f)
        for i in range(layers - 1):
            f = utils.orthonormal_VanillaLSTMBuilder(1, lstm_dim, lstm_dim, self._pc, isTest=config.isTest)
            LSTM_builders.append(f)

        return LSTM_builders

    def align(self, pred_bi, gold_bi):
        pred_chunks = utils.ranges(pred_bi)
        gold_chunks = utils.ranges(gold_bi)
        alignment_gold2pred = [0]
        alignment_pred2gold = [0] + [-1] * len(pred_chunks) #root is aligned with another root

        for ridx, r in enumerate(gold_chunks):
            if r in pred_chunks:
                pidx = pred_chunks.index(r)
                alignment_gold2pred.append(pidx)
                alignment_pred2gold[pidx] = ridx
            else:
                alignment_gold2pred.append(-1)

        return alignment_gold2pred, alignment_pred2gold


    def eval_dep(self, pred_bi, pred_deps, gold_bi, gold_deps, punct_mask, offset=0):
        if offset > 0:
            pred_bi = pred_bi[offset:-offset]
            gold_bi = gold_bi[offset:-offset]

        cor = 0
        succeed_chunk = [1] + [0] * len(gold_deps)

        g2p, p2g = self.align(pred_bi, gold_bi)
        for pidx, pred_head in enumerate(pred_deps):
            idx_dep = p2g[pidx]
            idx_head = p2g[pred_head]

            if idx_dep == -1 or idx_head == -1:
                continue
            else:
                if gold_deps[idx_dep] == idx_head and punct_mask[idx_dep] == 1:
                    cor += 1
                    succeed_chunk[idx_dep] = 1

        return cor, succeed_chunk

    def eval_rel(self, pred_bi, pred_rels, gold_bi, gold_rels, punct_mask, succeed_chunk, offset=0):
        if offset > 0:
            pred_bi = pred_bi[1:-1]
            gold_bi = gold_bi[1:-1]

        cor = 0

        g2p, p2g = self.align(pred_bi, gold_bi)
        for pidx, pred_rel in enumerate(pred_rels):
            gold_idx = p2g[pidx]
            if gold_rels[gold_idx] == pred_rel \
                    and punct_mask[gold_idx] \
                    and succeed_chunk[gold_idx]:
                cor += 1

            # if idx_dep == -1 or idx_head == -1:
            #     continue
            # else:
            #     if gold_deps[idx_dep] == idx_head and punct_mask[idx_dep] == 1:
            #         cor += 1
        return cor



    def embd_mask_generator(self, pdrop, indices):
        masks_w = np.random.binomial(1, 1 - pdrop, len(indices))
        masks_t = np.random.binomial(1, 1 - pdrop, len(indices))
        scales = [3. / (2. * mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
        masks_w = [mask_w * scale for mask_w, scale in zip(masks_w, scales)]
        masks_t = [mask_t * scale for mask_t, scale in zip(masks_t, scales)]
        self._masks_w = preprocess.seq2ids(masks_w, indices)
        self._masks_t = preprocess.seq2ids(masks_t, indices)

    def update_parameters(self):
        if config.use_annealing:
            self._trainer.learning_rate = config.learning_rate * config.decay ** (self._global_step / config.decay_steps)
        self._trainer.update()

    def seg_tag(self, embs, bi_word, tags):
        W_seg = dy.parameter(self.W_seg)
        W_tag = dy.parameter(self.W_tag)

        gold = []
        word_head = 0

        for idx, bi in enumerate(bi_word):
            if bi == 0:
                word_head = idx

            gold.append(word_head)

        logit_seg = utils.bilinear(embs, W_seg, embs, self._arc_dim, len(embs), 1, 1, True, True)
        loss = dy.pickneglogsoftmax_batch(logit_seg, gold)

        preds_seg = logit_seg.npvalue().argmax(0)

        return loss, preds_seg

    def seg(self, embs, bi_word):
        W_seg = dy.parameter(self.W_seg)
        loss = dy.scalarInput(0)
        preds_seg = []
        preds_seg_tuple = []
        gold_seg_tuple = []

        gold = []
        word_head = 0
        cor_seg = 0

        for idx, bi in enumerate(bi_word):
            if bi == 0:
                if idx > 0:
                    gold_seg_tuple.append((word_head, idx))
                word_head = idx
            gold.append(word_head)
        logit_seg = utils.bilinear(embs, W_seg, embs, self._arc_dim, len(embs), 1, 1, True, True)
        if self.isTrain:
            loss = dy.pickneglogsoftmax_batch(logit_seg, gold)
        else:
            for idx, bi in enumerate(bi_word):
                if bi == 0:
                    if idx > 0:
                        gold_seg_tuple.append((word_head, idx))
                    word_head = idx
                gold.append(word_head)

            preds_seg = logit_seg.npvalue().argmax(0)
            cur_head = 0

            for idx, head in enumerate(preds_seg):
                if cur_head != head:
                    if idx > 0:
                        preds_seg_tuple.append((cur_head, idx - 1))
                    cur_head = head

            for gold, pred in zip(gold_seg_tuple, preds_seg_tuple):
                if gold[0] == pred[0] and gold[1] == pred[1]:
                    cor_seg += 1

        return loss, cor_seg

    def pred_pos_char(self, tags, embs, isTrain):
        W = dy.parameter(self._W_pos_char)
        bias = dy.parameter(self._bias_pos_char)

        preds = []
        loss = dy.scalarInput(0)

        X = dy.concatenate_cols(embs)
        Y = dy.affine_transform([bias, dy.dropout(W, self._pdrop), utils.leaky_relu(X)])

        Y = dy.reshape(Y, (W.dim()[0][0], ), len(embs))

        if isTrain:
            loss = dy.pickneglogsoftmax_batch(Y, tags)
        else:
            preds = Y.npvalue().argmax(0)

        return dy.sum_batches(loss), preds

    def pred_pos_word(self, tags, embs, isTrain):
        W = dy.parameter(self._W_pos_word)
        bias = dy.parameter(self._bias_pos_word)

        preds = []
        loss = dy.scalarInput(0)

        X = dy.concatenate_cols(embs)
        Y = dy.affine_transform([bias, dy.dropout(W, self._pdrop), utils.leaky_relu(X)])

        Y = dy.reshape(Y, (W.dim()[0][0], ), len(embs))

        if isTrain:
            loss = dy.pickneglogsoftmax_batch(Y, tags)
        else:
            preds = Y.npvalue().argmax(0)

        return dy.sum_batches(loss), preds

    def pred_bi_word(self, bi_word, embs, isTrain):
        W = dy.parameter(self._W_bi)
        bias = dy.parameter(self._bias_bi)

        preds = []
        loss = dy.scalarInput(0)

        X = dy.concatenate_cols(embs)
        Y = dy.affine_transform([bias, dy.dropout(W, self._pdrop), utils.leaky_relu(X)])

        Y = dy.reshape(Y, (W.dim()[0][0], ), len(embs))

        if isTrain:
            loss = dy.pickneglogsoftmax_batch(Y, bi_word)
        else:
            preds = Y.npvalue().argmax(0)

        return dy.sum_batches(loss), preds

    def gold_tensor(self, gold_deps, gold_tags):
        # tensor = np.zeros((len(gold_deps) + 1, len(gold_tags) + 1, len(gold_deps) + 1))
        ans = []

        for idx in range(len(gold_deps)):
            ans.append(gold_tags[idx] + gold_deps[idx] * self._vocab_size_r)
            # ans.append(gold_deps[idx] + gold_tags[idx] * len(gold_deps))

        # tensor = np.reshape(tensor, (len(gold_deps), len(gold_deps) * len(gold_tags)))

        return ans

    def eval_dep_rel(self, gold_deps, gold_rels, preds):
        cor_dep = 0
        cor_rel = 0

        for idx in range(len(gold_deps)):
            if preds[idx] // self._vocab_size_r == gold_deps[idx]:
                cor_dep += 1
            if preds[idx] % self._vocab_size_r == gold_rels[idx]:
                cor_rel += 1

        return cor_dep, cor_rel

    def run(self, chars, tags_char, words, tags_word, bi_word, heads, rels, masks_w, masks_t, isTrain):
        rels = tags_char

        mlp_dep_bias = dy.parameter(self.mlp_dep_bias)
        mlp_dep = dy.parameter(self.mlp_dep)
        mlp_head_bias = dy.parameter(self.mlp_head_bias)
        mlp_head = dy.parameter(self.mlp_head)
        W_arc = dy.parameter(self.W_arc)
        W_rel = dy.parameter(self.W_rel)


        #tokens in the sentence and root
        seq_len = len(words) + 1

        punct_mask = np.array([1 if rel != self._punct_id else 0 for rel in rels])

        preds_arc = []
        preds_rel = []

        loss_arc = 0
        loss_rel = 0

        num_cor_arc = 0
        num_cor_arc_align = 0
        num_cor_rel = 0

        embs_c = [self.lp_c[c] for c in chars]
        # if self._pdrop_embs != .0:
        #     embs_c = [dy.dropout(c, self._pdrop_embs) for c in embs_c]

        # bidir_pos, l2rs_pos, r2ls_pos = utils.biLSTM(self.LSTM_builders_pos, embs_c, 1, self._pdrop, self._pdrop)
        #
        #
        # if isTrain:
        #     bidir_pos = [dy.dropout(lstm_out, self._pdrop) for lstm_out in bidir_pos]
        #
        # loss_pos, preds_pos = self.pred_pos_char(tags_char, bidir_pos, isTrain)
        #
        # embs_t = [self.lp_t[t] for t in (tags_char if isTrain else preds_pos)]
        #
        # feats = [dy.concatenate([emb_c, emb_t]) for emb_c, emb_t in zip(embs_c, embs_t)]
        #
        # lstm_ins_bi = [dy.concatenate([lstm_out, feat])
        #             for lstm_out, feat in zip(bidir_pos, feats)]
        #
        # bidir_bi, l2rs_bi, r2ls_bi = utils.biLSTM(self.LSTM_builders_bi, lstm_ins_bi, 1, self._pdrop, self._pdrop)
        #
        # loss_bi, preds_bi = self.pred_bi_word(bi_word, bidir_bi, isTrain)
        # if not isTrain:
        #     preds_bi[0], preds_bi[1], preds_bi[-1] = 0, 0, 0
        #
        # # masks_chunk = self.chunk_masks(preds_bi, bi_word, heads, offset=1)
        #
        # l2rs_bi, r2ls_bi = utils.residual_connection(l2rs_bi, [l2rs_pos], config.residual_connection_rate), \
        #                    utils.residual_connection(r2ls_bi, [r2ls_pos], config.residual_connection_rate)
        # word_range = utils.ranges(bi_word if isTrain else preds_bi)
        # bidir_subword, l2rs_subword, r2ls_subword = utils.segment_embds(l2rs_bi, r2ls_bi, word_range, offset=1)
        #
        # lstm_ins_dep = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(l2rs_subword, r2ls_subword)]

        lstm_ins_dep = embs_c
        bidir_dep, l2rs_dep, r2ls_dep = utils.biLSTM(self.LSTM_builders_dep, lstm_ins_dep, 1, self._pdrop, self._pdrop)

        bidir_dep = [self.emb_root[0]] + bidir_dep
        seq_len = len(bidir_dep)
        bidir_dep = dy.concatenate_cols(bidir_dep)

        embs_dep, embs_head = \
            utils.leaky_relu(dy.affine_transform([mlp_dep_bias, mlp_dep, bidir_dep])), \
            utils.leaky_relu(dy.affine_transform([mlp_head_bias, mlp_head, bidir_dep]))

        if isTrain:
            embs_dep, embs_head = dy.dropout(embs_dep, self._pdrop), dy.dropout(embs_head, self._pdrop)

        # dep_arc, dep_rel = embs_dep[:self._arc_dim], embs_dep[self._arc_dim:]
        # head_arc, head_rel = embs_head[:self._arc_dim], embs_head[self._arc_dim:]

        # logits_arc = utils.bilinear(dep_arc, W_arc, head_arc,
        #                             self._arc_dim, seq_len, config.batch_size, 1,
        #                             self.biaffine_bias_x_arc, self.biaffine_bias_y_arc)
        # flat_logits_arc = dy.reshape(logits_arc, (seq_len, ), seq_len)
        #
        # if isTrain:
        #     loss_arc = dy.pickneglogsoftmax_batch(flat_logits_arc, [0] + heads)
        # else:
        #     loss_arc = dy.scalarInput(0)
        #
        # # loss = loss_pos + loss_bi + dy.sum_batches(loss_arc)
        # loss = dy.sum_batches(loss_arc)
        #
        # if not isTrain:
        #     preds_arc = logits_arc.npvalue().argmax(0)
        #     num_cor_arc = np.sum(np.equal(np.equal(preds_arc[1:], heads), punct_mask))
        # #     num_cor_arc_align, succeed_deps = self.eval_dep(preds_bi, preds_arc[1:], bi_word, heads, punct_mask, offset=1)
        #
        # if not config.las:
        #     return loss, num_cor_arc, num_cor_rel
            # return loss, num_cor_arc_align, num_cor_rel

        logits_rel = utils.bilinear(embs_dep, W_rel, embs_head,
                                    self._rel_dim, seq_len, 1, self._vocab_size_r,
                                    self.biaffine_bias_x_rel, self.biaffine_bias_y_rel)

        # flat_logits_rel = dy.reshape(logits_rel, (seq_len, self._vocab_size_r), seq_len)
        flat_logits_rel = dy.reshape(logits_rel, (seq_len * self._vocab_size_r, ), seq_len)

        # partial_rel_logits = dy.pick_batch(flat_logits_rel, [0] + heads if isTrain else [0] + preds_arc)
        partial_rel_logits = flat_logits_rel
        ans = self.gold_tensor(heads, rels)

        if isTrain:
            loss_rel = dy.sum_batches(dy.pickneglogsoftmax_batch(partial_rel_logits, [0] + ans))
            num_cor_rel_align = 0
        else:
            preds_rel = partial_rel_logits.npvalue().argmax(0)
            # num_cor_rel = np.sum(np.equal(np.equal(preds_rel[1:], rels), punct_mask))
            # num_cor_rel = np.sum(np.equal(preds_rel[1:], ans))
            num_cor_arc, num_cor_rel = self.eval_dep_rel(heads, rels, preds_rel[1:])

            # num_cor_rel_align = self.eval_rel(preds_bi, preds_rel[1:], bi_word, rels, punct_mask, succeed_deps, offset=1)
        return loss_rel, num_cor_arc, num_cor_rel
        # return loss + loss_rel, num_cor_arc_align, num_cor_rel_align
