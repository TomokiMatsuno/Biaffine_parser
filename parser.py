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
                 pdrop_embs,
                 pdrop_lstm,
                 pdrop_mlp,
                 layers,
                 mlp_dim,
                 arc_dim,
                 biaffine_bias_x_arc,
                 biaffine_bias_y_arc,
                 biaffine_bias_x_rel,
                 biaffine_bias_y_rel,
                 embs_word=None
                 ):

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

        # self._trainer.set_clip_threshold(1.0)

        self.params = dict()
        if embs_word is None:
            self.lp_w = self._pc.add_lookup_parameters((word_size, input_dim), init=dy.ConstInitializer(0.))
        else:
            self.lp_w = self._pc.lookup_parameters_from_numpy(embs_word)
        self.lp_t = self._pc.add_lookup_parameters((tag_size, input_dim), init=dy.ConstInitializer(0.))
        self.emb_root = self._pc.add_lookup_parameters((2, input_dim), init=dy.ConstInitializer(0.))


        # if config.isTest:
        #     self.l2r_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        #     self.r2l_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        # else:
        #     self.l2r_lstm = utils.orthonormal_VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        #     self.r2l_lstm = utils.orthonormal_VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        self._pdrop_embs = pdrop_embs
        self._pdrop_lstm = pdrop_lstm
        self._pdrop_mlp = pdrop_mlp

        # self.mlp_dep = self._pc.add_parameters((mlp_dim, hidden_dim * 2))
        # self.mlp_head = self._pc.add_parameters((mlp_dim, hidden_dim * 2))
        # self.mlp_dep_bias = self._pc.add_parameters(mlp_dim)
        # self.mlp_head_bias = self._pc.add_parameters(mlp_dim)
        #
        # self.W_arc = self._pc.add_parameters((self._arc_dim + biaffine_bias_y_arc, self._arc_dim + biaffine_bias_x_arc))
        # self.W_rel = self._pc.add_parameters(((self._rel_dim + biaffine_bias_y_rel) * self._vocab_size_r, self._rel_dim + biaffine_bias_x_rel))
        if config.isTest:
            self.LSTM_builders = []
            f = dy.VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)
            b = dy.VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)

            self.LSTM_builders.append((f, b))
            for i in range(layers - 1):
                f = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
                b = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
                self.LSTM_builders.append((f, b))
        else:
            self.LSTM_builders = []
            f = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)
            b = utils.orthonormal_VanillaLSTMBuilder(1, input_dim * 2, hidden_dim, self._pc)

            self.LSTM_builders.append((f, b))
            for i in range(layers - 1):
                f = utils.orthonormal_VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
                b = utils.orthonormal_VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
                self.LSTM_builders.append((f, b))
        # self.dropout_lstm_input = dropout_lstm_input
        # self.dropout_lstm_hidden = dropout_lstm_hidden

        # mlp_size = mlp_arc_size + mlp_rel_size
        if config.biaffine:
            W = utils.orthonormal_initializer(mlp_dim, 2 * hidden_dim)
            self.mlp_dep = self._pc.parameters_from_numpy(W)
            self.mlp_head = self._pc.parameters_from_numpy(W)
            self.mlp_dep_bias = self._pc.add_parameters((mlp_dim,), init=dy.ConstInitializer(0.))
            self.mlp_head_bias = self._pc.add_parameters((mlp_dim,), init=dy.ConstInitializer(0.))
        else:
            W = utils.orthonormal_initializer(mlp_dim * 2, 2 * hidden_dim)
            self.mlp = self._pc.parameters_from_numpy(W)
            self.mlp_bias = self._pc.add_parameters((mlp_dim * 2,), init=dy.ConstInitializer(0.))

        # self.mlp_arc_size = mlp_arc_size
        # self.mlp_rel_size = mlp_rel_size
        # self.dropout_mlp = dropout_mlp
        if config.biaffine:
            self.W_arc = self._pc.add_parameters((self._arc_dim, self._arc_dim + 1),
                                                 init=dy.ConstInitializer(0.))
            self.W_rel = self._pc.add_parameters((self._vocab_size_r * (self._rel_dim + 1), self._rel_dim + 1),
                                                 init=dy.ConstInitializer(0.))
        else:
            self.V_r_arc = self._pc.add_parameters((self._arc_dim))
            self.V_i_arc = self._pc.add_parameters((self._arc_dim))
            self.bias_arc = self._pc.add_parameters((self._arc_dim * 2))
            self.V_r_rel = self._pc.add_parameters((self._rel_dim * self._vocab_size_r))
            self.V_i_rel = self._pc.add_parameters((self._rel_dim * self._vocab_size_r))
            self.bias_rel = self._pc.add_parameters((self._rel_dim * self._vocab_size_r * 2))

        return

    def embd_mask_generator(self, pdrop, indices):
        masks_w = np.random.binomial(1, 1 - pdrop, len(indices))
        masks_t = np.random.binomial(1, 1 - pdrop, len(indices))
        # scales = [3. / (2. * mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
        scales = [2. / (mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
        # masks_w = [1e-12 + mask_w * scale for mask_w, scale in zip(masks_w, scales)]
        # masks_t = [1e-12 + mask_t * scale for mask_t, scale in zip(masks_t, scales)]
        masks_w = [mask_w * scale for mask_w, scale in zip(masks_w, scales)]
        masks_t = [mask_t * scale for mask_t, scale in zip(masks_t, scales)]
        self._masks_w = preprocess.seq2ids(masks_w, indices)
        self._masks_t = preprocess.seq2ids(masks_t, indices)

    def update_parameters(self):
        if config.use_annealing:
            self._trainer.learning_rate = config.learning_rate * config.decay ** (self._global_step / config.decay_steps)
        self._trainer.update()

    def run(self, words, tags, heads, rels, masks_w, masks_t, isTrain):
        if config.biaffine:
            mlp_dep_bias = dy.parameter(self.mlp_dep_bias)
            mlp_dep = dy.parameter(self.mlp_dep)
            mlp_head_bias = dy.parameter(self.mlp_head_bias)
            mlp_head = dy.parameter(self.mlp_head)
            W_arc = dy.parameter(self.W_arc)
            W_rel = dy.parameter(self.W_rel)


        #tokens in the sentence and root
        seq_len = len(words) + 1

        punct_mask = np.array([1 if rel != self._punct_id else 0 for rel in rels], dtype=np.uint32)

        preds_arc = []
        preds_rel = []

        loss_arc = 0
        loss_rel = 0

        num_cor_arc = 0
        num_cor_rel = 0

        if isTrain:
            # embs_w = [self.lp_w[w if w < self._vocab_size_w else 0] * mask_w for w, mask_w in zip(words, masks_w)]
            # embs_t = [self.lp_t[t if t < self._vocab_size_t else 0] * mask_t for t, mask_t in zip(tags, masks_t)]
            embs_w = [self.lp_w[w] * mask_w for w, mask_w in zip(words, masks_w)]
            embs_t = [self.lp_t[t] * mask_t for t, mask_t in zip(tags, masks_t)]
            embs_w = [self.emb_root[0] * masks_t[-1]] + embs_w
            embs_t = [self.emb_root[1] * masks_w[-1]] + embs_t

        else:
            # embs_w = [self.lp_w[w if w < self._vocab_size_w else 0] for w in words]
            # embs_t = [self.lp_t[t if t < self._vocab_size_t else 0] for t in tags]
            embs_w = [self.lp_w[w] for w in words]
            embs_t = [self.lp_t[t] for t in tags]
            embs_w = [self.emb_root[0]] + embs_w
            embs_t = [self.emb_root[1]] + embs_t

        lstm_ins = [dy.concatenate([emb_w, emb_t]) for emb_w, emb_t in zip(embs_w, embs_t)]
        # lstm_outs = dy.concatenate_cols([self.emb_root[0]] + utils.bilstm(self.l2r_lstm, self.r2l_lstm, lstm_ins, self._pdrop))
        # lstm_outs = dy.concatenate_cols(utils.bilstm(self.l2r_lstm, self.r2l_lstm, lstm_ins, self._pdrop))
        lstm_outs = dy.concatenate_cols(utils.biLSTM(self.LSTM_builders, lstm_ins, None, self._pdrop_lstm, self._pdrop_lstm))

        # if isTrain:
        #     lstm_outs = dy.dropout(lstm_outs, self._pdrop)

        if config.biaffine:
            embs_dep, embs_head = \
                utils.leaky_relu(dy.affine_transform([mlp_dep_bias, mlp_dep, lstm_outs])), \
                utils.leaky_relu(dy.affine_transform([mlp_head_bias, mlp_head, lstm_outs]))

            if isTrain:
                embs_dep, embs_head = dy.dropout(embs_dep, self._pdrop_mlp), dy.dropout(embs_head, self._pdrop_mlp)

            dep_arc, dep_rel = embs_dep[:self._arc_dim], embs_dep[self._arc_dim:]
            head_arc, head_rel = embs_head[:self._arc_dim], embs_head[self._arc_dim:]

            logits_arc = utils.bilinear(dep_arc, W_arc, head_arc,
                                       self._arc_dim, seq_len, config.batch_size, 1,
                                       self.biaffine_bias_x_arc, self.biaffine_bias_y_arc)
        else:
            mlp = dy.parameter(self.mlp)
            mlp_bias = dy.parameter(self.mlp_bias)

            embs = \
                utils.leaky_relu(dy.affine_transform([mlp_bias, mlp, lstm_outs]))
            if isTrain:
                embs = dy.dropout(embs, self._pdrop_mlp)

            embs_arc, embs_rel = embs[:self._arc_dim * 2], embs[self._arc_dim * 2:]

            W_r_arc = dy.parameter(self.V_r_arc)
            W_i_arc = dy.parameter(self.V_i_arc)
            bias_arc = dy.parameter(self.bias_arc)

            logits_arc = utils.biED(embs_arc, W_r_arc, W_i_arc, embs_arc, seq_len, 1, bias=bias_arc)

        flat_logits_arc = dy.reshape(logits_arc, (seq_len, ), seq_len)
        flat_logits_arc = dy.pick_batch_elems(flat_logits_arc, [e for e in range(1, seq_len)])

        loss_arc = dy.pickneglogsoftmax_batch(flat_logits_arc, heads)

        if not isTrain:
            preds_arc = logits_arc.npvalue().argmax(0)
            cor_arcs = np.multiply(np.equal(preds_arc[1:], heads), punct_mask)
            num_cor_arc = np.sum(cor_arcs)

        if not config.las:
            return loss_arc, num_cor_arc, num_cor_rel

        if config.biaffine:
            logits_rel = utils.bilinear(dep_rel, W_rel, head_rel,
                                       self._rel_dim, seq_len, 1, self._vocab_size_r,
                                       self.biaffine_bias_x_rel, self.biaffine_bias_y_rel)
        else:
            V_r_rel = dy.parameter(self.V_r_rel)
            V_i_rel = dy.parameter(self.V_i_rel)
            bias_rel = dy.parameter(self.bias_rel)

            logits_rel = utils.biED(embs_rel, V_r_rel, V_i_rel, embs_rel, seq_len, self._vocab_size_r, bias=bias_rel)

        flat_logits_rel = dy.reshape(logits_rel, (seq_len, self._vocab_size_r), seq_len)
        flat_logits_rel = dy.pick_batch_elems(flat_logits_rel, [e for e in range(1, seq_len)])

        partial_rel_logits = dy.pick_batch(flat_logits_rel, heads if isTrain else preds_arc[1:])

        if isTrain:
            loss_rel = dy.sum_batches(dy.pickneglogsoftmax_batch(partial_rel_logits, rels))
        else:
            preds_rel = partial_rel_logits.npvalue().argmax(0)
            num_cor_rel = np.sum(np.multiply(np.equal(preds_rel, rels), cor_arcs))
        return loss_arc + loss_rel, num_cor_arc, num_cor_rel
        # return loss_arc, num_cor_arc, num_cor_rel
