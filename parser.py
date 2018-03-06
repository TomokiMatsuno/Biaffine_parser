import dynet as dy
import numpy as np

import utils
import config
import preprocess


class Parser(object):
    def __init__(self,
                 word_size,
                 tag_size,
                 rel_size,
                 input_dim,
                 hidden_dim,
                 pdrop,
                 pdrop_embs,
                 pdrop_lstm,
                 layers,
                 mlp_dim,
                 arc_dim,
                 biaffine_bias_x_arc,
                 biaffine_bias_y_arc,
                 biaffine_bias_x_rel,
                 biaffine_bias_y_rel,
                 embs_word=None
                 ):

        self.not_matched = 0

        self._global_step = 0
        self._early_stop_count = 0
        self._update = False
        self._best_score = 0.
        self._best_score_las = 0.

        self._punct_id = 0
        self._B_tag_id = 1

        self._masks_w = []
        self._masks_t = []

        self._vocab_size_w = word_size
        self._vocab_size_t = tag_size
        self._vocab_size_r = rel_size

        self._mlp_dim = mlp_dim
        self._arc_dim = arc_dim
        self._rel_dim = mlp_dim - arc_dim
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
            self.lp_w = self._pc.add_lookup_parameters((word_size, input_dim), init=(dy.ConstInitializer(0.) if config.const_init else None))
        else:
            self.lp_w = self._pc.lookup_parameters_from_numpy(embs_word)
        self.lp_t = self._pc.add_lookup_parameters((tag_size, input_dim), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.emb_root = self._pc.add_lookup_parameters((2, input_dim), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.emb_root_intra = self._pc.add_lookup_parameters((2, hidden_dim * 2), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.emb_root_intra_chunk = self._pc.add_lookup_parameters((2, hidden_dim * 2), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.emb_BOS = self._pc.add_lookup_parameters((2, input_dim), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.emb_EOS = self._pc.add_lookup_parameters((2, input_dim), init=(dy.ConstInitializer(0.) if config.const_init else None))


        # if config.isTest:
        #     self.l2r_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        #     self.r2l_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        # else:
        #     self.l2r_lstm = utils.orthonormal_VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        #     self.r2l_lstm = utils.orthonormal_VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)

        self._pdrop = pdrop
        self._pdrop_embs = pdrop_embs
        self._pdrop_lstm = pdrop_lstm

        self.LSTM_Builders_word = self.init_LSTMBuilders(config.layers_word, input_dim, hidden_dim)
        self.LSTM_Builders_word_2 = self.init_LSTMBuilders(1, hidden_dim, hidden_dim)
        self.LSTM_Builders_chunk = self.init_LSTMBuilders(config.layers_chunk, hidden_dim, hidden_dim)

        # self.dropout_lstm_input = dropout_lstm_input
        # self.dropout_lstm_hidden = dropout_lstm_hidden

        # mlp_size = mlp_arc_size + mlp_rel_size
        W = utils.orthonormal_initializer(mlp_dim, 2 * hidden_dim)
        self.mlp_dep = self._pc.parameters_from_numpy(W)
        self.mlp_head = self._pc.parameters_from_numpy(W)
        self.mlp_dep_bias = self._pc.add_parameters((mlp_dim,), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.mlp_head_bias = self._pc.add_parameters((mlp_dim,), init=(dy.ConstInitializer(0.) if config.const_init else None))

        self.mlp_intra_dep = self._pc.parameters_from_numpy(W)
        self.mlp_intra_head = self._pc.parameters_from_numpy(W)
        self.mlp_intra_dep_bias = self._pc.add_parameters((mlp_dim,), init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.mlp_intra_head_bias = self._pc.add_parameters((mlp_dim,), init=(dy.ConstInitializer(0.) if config.const_init else None))

        self.mlp_intra_dep_arc = self._pc.add_parameters((arc_dim, 2 * hidden_dim))
        self.mlp_intra_head_arc = self._pc.add_parameters((arc_dim, 2 * hidden_dim))
        self.mlp_intra_dep_arc_bias = self._pc.add_parameters(arc_dim)
        self.mlp_intra_head_arc_bias = self._pc.add_parameters(arc_dim)

        W = utils.orthonormal_initializer(2, 2 * hidden_dim)
        self.W_chunking = self._pc.parameters_from_numpy(W)
        self.bias_chunking = self._pc.add_parameters((2,), init=(dy.ConstInitializer(0.) if config.const_init else None))



        # self.mlp_arc_size = mlp_arc_size
        # self.mlp_rel_size = mlp_rel_size
        # self.dropout_mlp = dropout_mlp

        self.R_arc = self._pc.add_parameters((hidden_dim * 2, hidden_dim * 4),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.R_arc_chunk = self._pc.add_parameters((hidden_dim * 2, hidden_dim * 4),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))
        # self.R_head_arc = self._pc.add_parameters((self._arc_dim, self._arc_dim * 2),
        #                                      init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.R_dep_rel = self._pc.add_parameters((self._rel_dim, self._rel_dim * 2),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.R_head_rel = self._pc.add_parameters((self._rel_dim, self._rel_dim * 2),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))

        self.W_arc = self._pc.add_parameters((self._arc_dim, self._arc_dim + 1),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.W_arc_intra = self._pc.add_parameters((self._arc_dim, self._arc_dim + 1),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))
        self.W_rel = self._pc.add_parameters((self._vocab_size_r * (self._rel_dim + 1), self._rel_dim + 1),
                                             init=(dy.ConstInitializer(0.) if config.const_init else None))

        return

    def embd_mask_generator(self, pdrop, slen=None):
        masks_w = np.random.binomial(1, 1 - pdrop, slen)
        masks_t = np.random.binomial(1, 1 - pdrop, slen)
        scales = [3. / (2. * mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
        masks_w = [mask_w * scale for mask_w, scale in zip(masks_w, scales)]
        masks_t = [mask_t * scale for mask_t, scale in zip(masks_t, scales)]
        return masks_w, masks_t

    def init_LSTMBuilders(self, layers, input_dim, hidden_dim, bidir_input=True):
        LSTM_builders = []

        if not config.isTest:
            # f = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * (1 + bidir_input), hidden_dim, self._pc)
            # b = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * (1 + bidir_input), hidden_dim, self._pc)
            f = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)
            b = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)
        else:
            f = dy.VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)
            b = dy.VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)

        LSTM_builders.append((f, b))
        for i in range(layers - 1):
            if not config.isTest:
                f = utils.orthonormal_VanillaLSTMBuilder(1, hidden_dim * (1 + bidir_input), hidden_dim, self._pc)
                b = utils.orthonormal_VanillaLSTMBuilder(1, hidden_dim * (1 + bidir_input), hidden_dim, self._pc)
            else:
                f = dy.VanillaLSTMBuilder(1, hidden_dim * (1 + bidir_input), hidden_dim, self._pc)
                b = dy.VanillaLSTMBuilder(1, hidden_dim * (1 + bidir_input), hidden_dim, self._pc)

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

    def update_parameters(self):
        if config.use_annealing:
            self._trainer.learning_rate = config.learning_rate * config.decay ** (self._global_step / config.decay_steps)
        self._trainer.update()

    def chunk_words(self, bi_chunk, X, isTrain):
        W = dy.parameter(self.W_chunking)
        bias = dy.parameter(self.bias_chunking)

        preds = []
        loss = dy.scalarInput(0)

        Y = dy.affine_transform([bias, dy.dropout(W, self._pdrop), utils.leaky_relu(X)])

        Y = dy.reshape(Y, (W.dim()[0][0], ), len(bi_chunk))

        if isTrain:
            loss = dy.pickneglogsoftmax_batch(Y, bi_chunk)
        else:
            preds = Y.npvalue().argmax(0)

        return dy.sum_batches(loss), preds


    def run(self, words, tags, heads, rels, bi_chunk, isTrain):

        mlp_dep_bias = dy.parameter(self.mlp_dep_bias)
        mlp_dep = dy.parameter(self.mlp_dep)
        mlp_head_bias = dy.parameter(self.mlp_head_bias)
        mlp_head = dy.parameter(self.mlp_head)

        W_arc = dy.parameter(self.W_arc)
        W_rel = dy.parameter(self.W_rel)

        if config.japanese:
            mlp_intra_dep_bias = dy.parameter(self.mlp_intra_dep_bias)
            mlp_intra_dep = dy.parameter(self.mlp_intra_dep)
            mlp_intra_head_bias = dy.parameter(self.mlp_intra_head_bias)
            mlp_intra_head = dy.parameter(self.mlp_intra_head)
            W_arc_intra = dy.parameter(self.W_arc_intra)

            mlp_intra_dep_arc =dy.parameter(self.mlp_intra_dep_arc)
            mlp_intra_head_arc =dy.parameter(self.mlp_intra_head_arc)
            mlp_intra_dep_arc_bias =dy.parameter(self.mlp_intra_dep_arc_bias)
            mlp_intra_head_arc_bias =dy.parameter(self.mlp_intra_head_arc_bias)

        # tokens in the sentence and root
        seq_len = len(words) + 1

        punct_mask = np.array([1 if rel != self._punct_id else 0 for rel in rels])

        preds_arc = []
        preds_arc_intra = []
        preds_rel = []

        loss_arc = dy.scalarInput(0)
        loss_rel = 0

        tot_arc_intra = 0
        num_cor_arc = 0
        num_cor_arc_intra = 0
        num_cor_rel = 0
        type_cor_rel = []
        type_preds_rel = []
        cor_rel = []
        gold_rels = []
        system_rels = []
        preds_rel = []

        cor_bi = 0
        cor_inter = 0
        num_suc = 0

        masks_w, masks_t = self.embd_mask_generator(self._pdrop_embs, slen=len(words))

        if isTrain:
            embs_w = [self.lp_w[w] * mask_w for w, mask_w in zip(words, masks_w)]
            embs_t = [self.lp_t[t] * mask_t for t, mask_t in zip(tags, masks_t)]
            embs_w = [self.emb_BOS[0], self.emb_root[0] * masks_t[-1]] + embs_w + [self.emb_EOS[0]]
            embs_t = [self.emb_BOS[1], self.emb_root[1] * masks_w[-1]] + embs_t + [self.emb_EOS[1]]

        else:
            embs_w = [self.lp_w[w] for w in words]
            embs_t = [self.lp_t[t] for t in tags]
            embs_w = [self.emb_BOS[0], self.emb_root[0]] + embs_w + [self.emb_EOS[0]]
            embs_t = [self.emb_BOS[1], self.emb_root[1]] + embs_t + [self.emb_EOS[1]]

        lstm_ins = [dy.concatenate([emb_w, emb_t]) for emb_w, emb_t in zip(embs_w, embs_t)]
        bidirouts_word, l2routs_word, r2louts_word = utils.biLSTM(self.LSTM_Builders_word, lstm_ins, None, self._pdrop_lstm, self._pdrop_lstm)
        # _, l2routs_word, r2louts_word = utils.biLSTM(self.LSTM_Builders_word_2, lstm_ins, None, self._pdrop_lstm, self._pdrop_lstm)
        lstm_outs_word = dy.concatenate_cols(bidirouts_word[1:-1])

        if isTrain:
            lstm_outs_word = dy.dropout(lstm_outs_word, self._pdrop)

        if len(bidirouts_word) > lstm_outs_word.dim()[0][1] + 2:
            print('select col out of bounds')

        loss_bi, preds_chunk = self.chunk_words(bi_chunk, lstm_outs_word, isTrain)

        if not isTrain:
            cor_bi = np.sum(np.equal(preds_chunk, bi_chunk))

        if not isTrain:
            preds_chunk[0], preds_chunk[1] = self._B_tag_id, self._B_tag_id
        word_ranges = utils.ranges(bi_chunk if isTrain else preds_chunk, self._B_tag_id)
        heads_inter, heads_intra, chunk_heads = utils.inter_intra_dep(heads, bi_chunk if isTrain else preds_chunk, self._B_tag_id, rels, self._punct_id)

        lstm_ins_chunk = utils.segment_embds(l2routs_word, r2louts_word, ranges=word_ranges, offset=1)
        bidirouts_chunk, _, _ = utils.biLSTM(self.LSTM_Builders_chunk, lstm_ins_chunk, None, self._pdrop_lstm, self._pdrop_lstm)
        lstm_outs_chunk = dy.concatenate_cols(bidirouts_chunk)

        embs_dep, embs_head = \
            utils.leaky_relu(dy.affine_transform([mlp_dep_bias, mlp_dep, lstm_outs_chunk])), \
            utils.leaky_relu(dy.affine_transform([mlp_head_bias, mlp_head, lstm_outs_chunk]))

        if isTrain:
            embs_dep, embs_head = dy.dropout(embs_dep, self._pdrop), dy.dropout(embs_head, self._pdrop)

        dep_arc_chunk, dep_rel_chunk = embs_dep[:self._arc_dim], embs_dep[self._arc_dim:]
        head_arc_chunk, head_rel_chunk = embs_head[:self._arc_dim], embs_head[self._arc_dim:]


        dep_arc_inter = dep_arc_chunk
        head_arc_inter = head_arc_chunk

        len_inter = len(bidirouts_chunk)

        logits_arc = utils.bilinear(dep_arc_inter, W_arc, head_arc_inter,
                                    self._arc_dim, len_inter, config.batch_size, 1,
                                    self.biaffine_bias_x_arc, self.biaffine_bias_y_arc)
        flat_logits_arc = dy.reshape(logits_arc, (len_inter, ), len_inter)

        if isTrain:
            if len_inter != len(heads_inter) + 1:
                print('error!')

            if max(heads_inter) >= flat_logits_arc.dim()[0][0]:
                return loss_arc, num_cor_arc, num_cor_rel

            loss_arc += dy.pickneglogsoftmax_batch(flat_logits_arc, [0] + heads_inter)
        else:
            msk = [1] * len_inter
            arc_probs = dy.softmax(logits_arc).npvalue()
            arc_probs = np.transpose(arc_probs)
            arc_preds = utils.arc_argmax(arc_probs, len_inter, msk)
            preds_arc_chunk = arc_preds[1:]

            cor_inter += np.sum(np.equal(preds_arc_chunk, heads_inter))

            # preds_arc = logits_arc.npvalue().argmax(0)
            # num_cor_arc = np.sum(np.equal(preds_arc[1:], heads_inter))
            # res_arc = utils.word_dep(preds_arc[1:], preds_arc_intra, preds_chunk, chunk_heads, self._B_tag_id)
            # if len(res_arc) != len(heads):
            #     print('error!')
            #     return loss_arc, num_cor_arc, num_cor_rel
            # cor_arc_mask = np.equal(res_arc, heads)
            # num_cor_arc = np.sum(cor_arc_mask)
            # num_cor_arc = np.sum(np.multiply(np.equal(res_arc, heads), punct_mask))

        embs_chunk = []
        tmp_chunk = []
        idx_chunk = 0
        R_arc = dy.parameter(self.R_arc)

        for idx, bi in enumerate(bi_chunk):
            tmp_chunk.append(bidirouts_chunk[idx_chunk])

            if bi == 0 and idx_chunk < len(bidirouts_chunk) - 1:
                idx_chunk += 1

        embs_chunk = dy.concatenate_cols(tmp_chunk)

        # lstm_outs_word = R_arc * dy.concatenate([lstm_outs_word, embs_chunk])

        embs_intra_dep, embs_intra_head = \
            utils.leaky_relu(dy.affine_transform([mlp_intra_dep_bias, mlp_intra_dep, lstm_outs_word])), \
            utils.leaky_relu(dy.affine_transform([mlp_intra_head_bias, mlp_intra_head, lstm_outs_word]))
        if len(bidirouts_word) > lstm_outs_word.dim()[0][1] + 2:
            print('select col out of bounds')
        if isTrain:
            embs_intra_dep, embs_intra_head = dy.dropout(embs_intra_dep, self._pdrop), dy.dropout(embs_intra_head, self._pdrop)
        if len(bidirouts_word) > lstm_outs_word.dim()[0][1] + 2:
            print('select col out of bounds')

        # embs_intra_dep, embs_intra_head = lstm_outs_word, lstm_outs_word

        dep_arc_word, dep_rel_word = embs_intra_dep[:self._arc_dim], embs_intra_dep[self._arc_dim:]
        head_arc_word, head_rel_word = embs_intra_head[:self._arc_dim], embs_intra_head[self._arc_dim:]

        preds_arc_child = []
        preds_arc_parent = []
        gold_arc_child = []
        gold_arc_parent = []
        if not isTrain:
            heads_inter = preds_arc_chunk

        R_arc_chunk = dy.parameter(self.R_arc_chunk)

        for idx, head_intra in enumerate(heads_intra):
            chunk_child = word_ranges[idx + 1]
            chunk_parent = word_ranges[heads_inter[idx]]

            col_range_child = [i + 1 for i in range(chunk_child[0], chunk_child[1])]
            col_range_parent = [i + 1 for i in range(chunk_parent[0], chunk_parent[1])]

            len_chunk = chunk_child[1] - chunk_child[0] + chunk_parent[1] - chunk_parent[0] + 1

            embs_intra_arc = dy.concatenate_cols([self.emb_root_intra[0], lstm_outs_word])
            embs_intra_arc = [dy.select_cols(embs_intra_arc, [idx]) for idx in [0] + col_range_child + col_range_parent]
            embs_intra_arc, _, _ = utils.biLSTM(self.LSTM_Builders_word_2, embs_intra_arc, 1, self._pdrop_lstm, self._pdrop_lstm)
            embs_intra_arc = dy.concatenate_cols(embs_intra_arc)
            chunk_embs = dy.concatenate_cols([self.emb_root_intra_chunk[0]] + [bidirouts_chunk[idx + 1]] * len(col_range_child) + [bidirouts_chunk[heads_inter[idx]]] * len(col_range_parent))
            embs_intra_arc = R_arc_chunk * dy.concatenate([embs_intra_arc, chunk_embs])

            dep_arc_intra, head_arc_intra = \
                utils.leaky_relu(dy.affine_transform([mlp_intra_dep_arc_bias, mlp_intra_dep_arc, embs_intra_arc])), \
                utils.leaky_relu(dy.affine_transform([mlp_intra_head_arc_bias, mlp_intra_head_arc, embs_intra_arc]))

            if isTrain:
                dep_arc_intra, head_arc_intra= dy.dropout(dep_arc_intra, self._pdrop), dy.dropout(head_arc_intra,
                                                                                                      self._pdrop)

            logits_arc = utils.bilinear(dep_arc_intra, W_arc_intra, head_arc_intra,
                                        self._arc_dim, len_chunk, config.batch_size, 1,
                                        self.biaffine_bias_x_arc, self.biaffine_bias_y_arc)
            flat_logits_arc = dy.reshape(logits_arc, (len_chunk, ), len_chunk)

            # if len(head_intra) != len(col_range_child + col_range_parent):
            #     print('error')
            #
            # if len(head_intra) + 1 <= max(head_intra):
            #     print('error')


            if isTrain:
                loss_arc += dy.sum_batches(dy.pickneglogsoftmax_batch(flat_logits_arc, [0] + head_intra))
                # loss_arc += dy.sum_batches(dy.pickneglogsoftmax_batch(flat_logits_arc, head_intra))
                # preds_arc = logits_arc.npvalue().argmax(0)
                preds_arc_child.append(head_intra[len(col_range_child):])
                # preds_arc_parent.append([arc - len(col_range_child) + 1 for arc in head_intra[:len(col_range_child)]])
                preds_arc_parent.append([arc for arc in head_intra[:len(col_range_child)]])
                preds_chunk = bi_chunk

                preds_arc_chunk = heads_inter
            else:
                preds_arc = logits_arc.npvalue().argmax(0)
                preds_arc = preds_arc[1:]
                # preds_arc_intra.append(preds_arc[1 + len(col_range_child):])
                preds_arc_child.append(preds_arc[len(col_range_child):])
                preds_arc_parent.append(preds_arc[:len(col_range_child)])
                gold_arc_child.append(head_intra[len(col_range_child):])
                gold_arc_parent.append(head_intra[:len(col_range_child)])
                # head_mask = np.not_equal(head_intra, np.zeros(len(head_intra)))
                # num_cor_arc_intra += np.sum(np.multiply(np.equal(preds_arc[1:], head_intra), head_mask))
                # tot_arc_intra += np.sum(head_mask)

        if not isTrain:
            res_arc = utils.complete_sents(preds_arc_parent, preds_arc_child, preds_arc_chunk, preds_chunk)
            gold_arc = utils.complete_sents(gold_arc_parent, gold_arc_child, preds_arc_chunk, preds_chunk)

            if config.exclude_puncts:
                cor_arc_mask = np.multiply(np.equal(res_arc, heads))
            else:
                cor_arc_mask = np.equal(res_arc, heads)
            num_cor_arc = sum(cor_arc_mask)

        if not config.las:
            return loss_arc + loss_bi + loss_rel, num_cor_arc, num_cor_arc_intra, tot_arc_intra, num_cor_rel, \
                   cor_rel, gold_rels, system_rels, cor_bi, cor_inter, len_inter - 1, num_suc

        # embs_dep_chunk = []
        # embs_head_chunk = []
        # idx_chunk = -1
        # for bi in bi_chunk if isTrain else preds_chunk:
        #     if bi == 0:
        #         idx_chunk += 1
        #     embs_dep_chunk.append(dy.select_cols(dep_rel_chunk, [idx_chunk]))
        #     embs_head_chunk.append(dy.select_cols(head_rel_chunk, [idx_chunk]))
        #
        # R_dep_rel = dy.parameter(self.R_dep_rel)
        # R_head_rel = dy.parameter(self.R_head_rel)
        #
        # dep_rel_word = R_dep_rel * dy.concatenate([dep_rel_word, dy.concatenate_cols(embs_dep_chunk)])
        # head_rel_word = R_head_rel * dy.concatenate([head_rel_word, dy.concatenate_cols(embs_head_chunk)])
        #
        # dep_rel_word, head_rel_word = utils.leaky_relu(dep_rel_word), utils.leaky_relu(head_rel_word)
        # if isTrain:
        #     dep_rel_word, head_rel_word = dy.dropout(dep_rel_word, self._pdrop), dy.dropout(head_rel_word, self._pdrop)


        logits_rel = utils.bilinear(dep_rel_word, W_rel, head_rel_word,
                                    self._rel_dim, seq_len, 1, self._vocab_size_r,
                                    self.biaffine_bias_x_rel, self.biaffine_bias_y_rel)

        flat_logits_rel = dy.reshape(logits_rel, (seq_len, self._vocab_size_r), seq_len)

        partial_rel_logits = dy.pick_batch(flat_logits_rel, [0] + heads if isTrain else [0] + res_arc)

        if isTrain:
            loss_rel = dy.sum_batches(dy.pickneglogsoftmax_batch(partial_rel_logits, [0] + rels))
        else:
            if max(res_arc) > len(res_arc):
                return loss_arc + loss_bi + loss_rel, num_cor_arc, num_cor_arc_intra, tot_arc_intra, num_cor_rel, \
                       cor_rel, gold_rels, system_rels, cor_bi, cor_inter, len_inter - 1, num_suc
            preds_rel = partial_rel_logits.npvalue().argmax(0)
            num_cor_rel = np.sum(np.multiply(np.equal(preds_rel[1:], rels), cor_arc_mask))
            # num_cor_rel = np.sum(np.multiply(np.multiply(np.equal(preds_rel[1:], rels), cor_arc_mask), punct_mask))
            cor_rel = np.multiply(rels, np.multiply(np.equal(preds_rel[1:], rels), cor_arc_mask))
            #gold_rels = np.multiply(rels, cor_arc_mask)
            gold_rels = rels
            # type_cor_rel = np.multiply(rels, np.equal(preds_rel[1:], rels))
            #system_rels = np.multiply(preds_rel[1:], cor_arc_mask)
            system_rels = preds_rel[1:]

        return loss_arc + loss_bi + loss_rel, num_cor_arc, num_cor_arc_intra, tot_arc_intra, num_cor_rel,\
               cor_rel, gold_rels, system_rels, cor_bi, cor_inter, len_inter - 1, num_suc
