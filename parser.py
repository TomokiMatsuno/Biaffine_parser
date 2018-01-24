import dynet as dy
import numpy as np

import utils
import config
import preprocess

class Parser(object):
    def __init__(self,
                 word_size,
                 tag_size,
                 input_dim,
                 hidden_dim,
                 pdrop,
                 layers,
                 mlp_dim,
                 arc_dim,
                 biaffine_bias_x_arc,
                 biaffine_bias_y_arc,
                 biaffine_bias_x_rel,
                 biaffine_bias_y_rel
                 ):

        self._global_step = 0

        self._masks_w = []
        self._masks_t = []

        self._vocab_size_w = word_size
        self._vocab_size_t = tag_size

        rel_dim = mlp_dim - arc_dim
        self.mlp_dim = mlp_dim
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
        self.lp_w = self._pc.add_lookup_parameters((word_size, input_dim))
        self.lp_t = self._pc.add_lookup_parameters((tag_size, input_dim))
        self.emb_root = self._pc.add_lookup_parameters((1, hidden_dim * 2))

        self.l2r_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        self.r2l_lstm = dy.VanillaLSTMBuilder(layers, input_dim * 2, hidden_dim, self._pc)
        self._pdrop = pdrop

        self.mlp_dep = self._pc.add_parameters((mlp_dim, hidden_dim * 2))
        self.mlp_head = self._pc.add_parameters((mlp_dim, hidden_dim * 2))
        self.mlp_dep_bias = self._pc.add_parameters(mlp_dim)
        self.mlp_head_bias = self._pc.add_parameters(mlp_dim)

        self.W_arc = self._pc.add_parameters((arc_dim + biaffine_bias_y_arc, arc_dim + biaffine_bias_x_arc))
        self.W_rel = self._pc.add_parameters((rel_dim + biaffine_bias_y_rel, rel_dim + biaffine_bias_x_rel))

        return


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

    def run(self, words, tags, heads, rels, masks_w, masks_t, isTrain):
        mlp_dep_bias = dy.parameter(self.mlp_dep_bias)
        mlp_dep = dy.parameter(self.mlp_dep)
        mlp_head_bias = dy.parameter(self.mlp_head_bias)
        mlp_head = dy.parameter(self.mlp_head)
        W_arc = dy.parameter(self.W_arc)


        #tokens in the sentence and root
        seq_len = len(words) + 1

        preds_arc = []
        num_cor_arc = 0
        if isTrain:
            embs_w = [self.lp_w[w if w < self._vocab_size_w else 0] * mask_w for w, mask_w in zip(words, masks_w)]
            embs_t = [self.lp_t[t if t < self._vocab_size_t else 0] * mask_t for t, mask_t in zip(tags, masks_t)]
        else:
            embs_w = [self.lp_w[w if w < self._vocab_size_w else 0] for w in words]
            embs_t = [self.lp_t[t if t < self._vocab_size_t else 0] for t in tags]

        lstm_ins = [dy.concatenate([emb_w, emb_t]) for emb_w, emb_t in zip(embs_w, embs_t)]
        lstm_outs = dy.concatenate_cols([self.emb_root[0]] + utils.bilstm(self.l2r_lstm, self.r2l_lstm, lstm_ins, self._pdrop))

        embs_dep, embs_head = \
            dy.rectify(dy.affine_transform([mlp_dep_bias, mlp_dep, lstm_outs])), \
            dy.rectify(dy.affine_transform([mlp_head_bias, mlp_head, lstm_outs]))

        if isTrain:
            embs_dep, embs_head = dy.dropout(embs_dep, self._pdrop), dy.dropout(embs_head, self._pdrop)

        logits_arc = utils.bilinear(embs_dep, W_arc, embs_head,
                                    self.mlp_dim, seq_len, config.batch_size, 1,
                                    self.biaffine_bias_x_arc, self.biaffine_bias_y_arc)
        flat_logits_arc = dy.reshape(logits_arc, (seq_len, ), seq_len)

        loss_arc = dy.pickneglogsoftmax_batch(flat_logits_arc, [0] + heads)

        if not isTrain:
            preds_arc = logits_arc.npvalue().argmax(0)
            num_cor_arc = np.sum(np.equal(preds_arc[1:], heads))

        return loss_arc, preds_arc, num_cor_arc














