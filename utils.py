import dynet as dy
import numpy as np

# id of B tag in BI tag sequences

def ranges(bi_seq, B_tag_idx):
    ret = []
    start = 0

    for i in range(1, len(bi_seq)):
        if bi_seq[i] == B_tag_idx:
            end = i
            ret.append((start, end))
            start = i

    ret.append((start, len(bi_seq)))

    return ret


def inter_intra_dep(parents_word, bi_chunk, B_tag_idx):
    w2ch = align_word_chunk(bi_chunk, B_tag_idx)
    word_ranges = ranges(bi_chunk, B_tag_idx)
    chunk_heads = []
    intra_dep = []
    parents_word = [0] + parents_word

    for r in word_ranges[1:]:
        start, end = r[0], r[1]
        # start, end = r[0], r[1]

        head_found = False
        intra_dep.append([])
        for w in range(start, end):

            parent_id = parents_word[w]
            diff = parent_id - start + 1
            if (parent_id < start or end <= parent_id):
                if not head_found:
                    chunk_heads.append(w)
                    head_found = True
                intra_dep[-1].append(0)
            else:
                intra_dep[-1].append(diff)

    inter_dep = [w2ch[parents_word[ch_head]] for ch_head in chunk_heads]

    return inter_dep, intra_dep, [0] + chunk_heads


def word_dep(inter_dep, intra_dep, bi_chunk, chunk_heads, B_tag_idx):
    ret = []
    # inter_dep = [0] + inter_dep
    word_ranges = ranges(bi_chunk, B_tag_idx)

    for idx, chunk in enumerate(intra_dep):
        start = word_ranges[idx + 1][0]
        for w in chunk:
            if w == 0:
                ret.append(chunk_heads[inter_dep[idx]])
            else:
                ret.append(start + w - 1)

    return ret


def align_word_chunk(bi_chunk, B_tag_idx):
    word2chunk = [0]

    chunk_idx = 0
    for bi in bi_chunk[1:]:
        if bi == B_tag_idx:
            chunk_idx += 1
        word2chunk.append(chunk_idx)

    return word2chunk



def bilstm(l2rlstm, r2llstm, inputs, pdrop):
    l2rlstm.set_dropouts(pdrop)
    r2llstm.set_dropouts(pdrop)

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    l2r_outs = s_l2r.add_inputs(inputs)
    r2l_outs = s_r2l.add_inputs(reversed(inputs))

    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]
    # l2r_outs = [l2r_outs[i].output() for i in range(len(l2r_outs))]
    # r2l_outs = [r2l_outs[i].output() for i in range(len(r2l_outs))]

    return lstm_outs#, l2r_outs, r2l_outs


def inputs2singlelstmouts(lstm, inputs, pdrop):
    s_0 = lstm.initial_state()

    lstm.set_dropouts(pdrop, pdrop)

    s = s_0

    outs = s.add_inputs(inputs)

    lstm_outs = [outs[i].output() for i in range(len(outs))]

    return lstm_outs


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs = 1, bias_x = False, bias_y = False):
    # adopted from: https://github.com/jcyk/Dynet-Biaffine-dependency-parser/blob/master/lib/utils.py

    # x,y: (input_size x seq_len) x batch_size
    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs*seq_len), batch_size = batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size = batch_size)
    # seq_len_y x seq_len_x if output_size == 1
    # seq_len_y x num_outputs x seq_len_x else
    return blin


def leaky_relu(x):
    return dy.bmax(.1 * x, x)


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc):
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer >0 else input_dims)) #the first layer takes prev hidden and input vec
        W_h, W_x = W[:,:lstm_hiddens], W[:,lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x]*4, 0))
        params[1].set_value(np.concatenate([W_h]*4, 0))
        b = np.zeros(4*lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2*lstm_hiddens] = -1.0#fill second quarter of bias vec with -1.0
        params[2].set_value(b)
    return builder


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print (output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


# def biLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0.):
#     for fb, bb in builders:
#         f, b = fb.initial_state(), bb.initial_state()
#         fb.set_dropouts(dropout_x, dropout_h)
#         bb.set_dropouts(dropout_x, dropout_h)
#         if batch_size is not None:
#             fb.set_dropout_masks(batch_size)
#             bb.set_dropout_masks(batch_size)
#         fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
#         inputs = [dy.concatenate([f,b]) for f, b in zip(fs, reversed(bs))]
#     return inputs
#

def uniLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0., rev=False):
    for b in builders:
        b.set_dropouts(dropout_x, dropout_h)
        s_0 = b.initial_state()
        if batch_size is not None:
            b.set_dropout_masks(batch_size)
        s = s_0.transduce(inputs if not rev else reversed(inputs))
        inputs = s if not rev else reversed(s)
    return inputs


def biLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0., bidir_input=True, inputs_f=None, inputs_b=None):
    fs, bs = inputs, inputs

    for fb, bb in builders:
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        f, b = fb.initial_state(), bb.initial_state()
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs if bidir_input else fs), b.transduce(reversed(inputs if bidir_input else bs))
        inputs = [dy.concatenate([f,b]) for f, b in zip(fs, reversed(bs))]
        # inputs_f, inputs_b = [f for f in fs], [b for b in bs]
    return inputs, fs, bs


def segment_embds(l2r_outs, r2l_outs, ranges, offset=0, segment_concat=False):
    ret = []
    if offset == 0:
        st = 0
        en = len(ranges)
    elif offset == -1:
        st = 0
        en = len(ranges)
        offset = 0
    else:
        st = offset
        en = -offset

    # for r in ranges[st:en]:
    for r in ranges:
        # start = r[0] - offset
        # end = r[1] - offset
        start = r[0] + offset
        end = r[1] + offset

        if segment_concat:
            l2r = l2r_outs[end - 1] + l2r_outs[start]
            r2l = r2l_outs[start] + r2l_outs[end - 1]
            # l2r = l2r_outs[end - 1]
            # r2l = r2l_outs[start]
        else:
            l2r = l2r_outs[end - 1] - l2r_outs[start - 1]
            r2l = r2l_outs[start] - r2l_outs[end]

        ret.append(dy.concatenate([l2r, r2l]))

    return ret



def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))


        # block loops and pad heads
        parse_probs = parse_probs * tokens_to_keep * (1 - I)
        # arc_masks = left_arc_mask(length)
        # parse_probs = parse_probs * arc_masks
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length) #original
        # tokens = np.arange(length) #modified

        roots = np.where(parse_preds[tokens] == 0)[0] + 1 #original
        # roots = np.where(parse_preds[tokens] == 0)[0] #modified
        # ensure at least one root
        if len(roots) < 1:
            # global root_0
            # root_0 += 1

            # The current root probabilities
            root_probs = parse_probs[tokens, 0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
        # ensure at most one root
        elif len(roots) > 1:
            # global root_more_than_1
            # root_more_than_1 += 1

            # The probabilities of the current heads
            root_probs = parse_probs[roots, 0]
            # Set the probability of depending on the root zero
            parse_probs[roots, 0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1 # original line
            # new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) # modified line
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            # global circle_count
            # circle_count += 1

            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1 #original
                # new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) #modified
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds
