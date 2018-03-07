import dynet as dy
import numpy as np
from tarjan import Tarjan

# id of B tag in BI tag sequences

def ranges(bi_seq, B_tag_idx=0):
    ret = []
    start = 0

    for i in range(1, len(bi_seq)):
        if bi_seq[i] == B_tag_idx:
            end = i
            ret.append((start, end))
            start = i

    ret.append((start, len(bi_seq)))

    return ret

def validate_sent(seq_h, seq_bi, seq_r, B_tag_id, punct_id):
    # chunk_ranges = ranges([0] + seq_bi)

    w2ch = align_word_chunk(seq_bi, with_root=False)

    # heads_inter, heads_intra, chunk_heads = inter_intra_dep(seq_h, [0] + seq_bi, B_tag_id, seq_r,
    #                                                               punct_id)
    # # back = word_dep(heads_inter, heads_intra, chunk_heads=chunk_heads, bi_chunk=[0] + seq_bi,
    # #                       B_tag_idx=B_tag_id)

    chunk_ranges = ranges([0] + seq_bi, B_tag_id)
    seq_h = [0] + seq_h
    seq_r = [0] + seq_r

    for r in chunk_ranges[1:]:
        start = r[0]
        end = r[1]
        set_head_par = set()
        list_rel = []
        for i in range(start, end):
            if seq_h[i] < start or end <= seq_h[i]:
                if w2ch[seq_h[i]] not in set_head_par:
                    set_head_par.add(w2ch[seq_h[i]])
                    list_rel.append(seq_r[i])
        # if len(set_head_par) > 1 and punct_id not in list_rel:
        if len(set_head_par) > 1:
            return False

    return True


def omit_invalid_sents(word_ids, tag_ids, head_ids, rel_ids, bi_ids, B_tag_id, punct_id):
    valid_ids = [validate_sent(head_ids[i], bi_ids[i], rel_ids[i], B_tag_id, punct_id) for i in range(len(bi_ids))]
    word_ids = [word_ids[idx] for idx in range(len(word_ids)) if valid_ids[idx]]
    tag_ids = [tag_ids[idx] for idx in range(len(tag_ids)) if valid_ids[idx]]
    head_ids = [head_ids[idx] for idx in range(len(head_ids)) if valid_ids[idx]]
    rel_ids = [rel_ids[idx] for idx in range(len(rel_ids)) if valid_ids[idx]]
    bi_ids = [bi_ids[idx] for idx in range(len(bi_ids)) if valid_ids[idx]]


    return word_ids, tag_ids, head_ids, rel_ids, bi_ids






def inter_intra_dep(parents_word, bi_chunk, B_tag_idx, rels, punct_idx):
    w2ch = align_word_chunk(bi_chunk, B_tag_idx)
    chunk_ranges = ranges(bi_chunk, B_tag_idx)
    chunk_heads = []
    intra_dep = []
    parents_word = [0] + parents_word
    rels = [0] + rels
    par_outside = []

    for r in chunk_ranges[1:]:
        start, end = r[0], r[1]
        # start, end = r[0], r[1]

        head_found = False
        intra_dep.append([])
        par_outside.append([])
        for w in range(start, end):

            parent_id = parents_word[w]
            diff = parent_id - start + 1
            if (parent_id < start or end <= parent_id):
                if rels[w] != punct_idx and not head_found:
                    chunk_heads.append(w)
                    head_found = True
                intra_dep[-1].append(0)
                par_outside[-1].append(parent_id)
            else:
                intra_dep[-1].append(diff)

        if not head_found:
            chunk_heads.append(w)


    inter_dep = [w2ch[parents_word[ch_head]] for ch_head in chunk_heads]

    ret = []

    for idx, par_chunk in enumerate(inter_dep):
        # if inter_dep[-1] != 0:
        #     print('!')

        # if par_chunk == 0:
        #     continue
        child = []
        par_start, par_end = chunk_ranges[par_chunk]
        for elem in intra_dep[idx]:
            if elem == 0:
                if elem > par_outside[idx][0] - par_start + len(intra_dep[idx]):
                    child.append(0)
                else:
                    child.append(par_outside[idx].pop(0) - par_start + len(intra_dep[idx]))
            else:
                if elem > len(intra_dep[idx]) + len(intra_dep[inter_dep[idx] - 1]) + 3:
                    print('error')
                child.append(elem)
        if par_chunk != 0:
            tmp = child + [(elem + len(intra_dep[idx]) - 1) if elem != 0 else 0 for elem in intra_dep[inter_dep[idx] - 1]]
        else:
            tmp = child + [0]

        ret.append(tmp)

    # return inter_dep, intra_dep, [0] + chunk_heads
    return inter_dep, ret, [0] + chunk_heads



def re_chunk(heads, bi_chunk):

    bi_chunk = [0] + bi_chunk
    chunk_ranges = ranges(bi_chunk)
    w2ch = align_word_chunk(bi_chunk)
    heads = [0] + heads
    ret = []
    rechunk_flag = [0]

    set_heads = set()
    for idx in range(len(chunk_ranges[1:])):

        start, end = chunk_ranges[idx + 1]
        num_heads = 0
        for w in range(start, end):
            parent_id = heads[w]
            if parent_id < start or end <= parent_id:
                num_heads += 1
                # set_heads.add(parent_id)
                set_heads.add(w)
                if num_heads > 1:
                    rechunk_flag.append(1)

                    break
        if num_heads <= 1:
            rechunk_flag.append(0)

    for idx, h in enumerate(heads):
        if h not in set_heads:
            rechunk_flag[w2ch[h]] = 1


    for idx, r in enumerate(chunk_ranges):
        if rechunk_flag[idx] == 1:
            ret.extend([0] * (r[1] - r[0]))
        else:
            ret.extend(bi_chunk[r[0]:r[1]])

    return ret[1:]


def chunk_tags(rels, func, words, func_begin, func_end, indp, prefix, subob, posfunc, poscont, tags, td, rd):
    ret = []
    flag_cont = False
    r_prev = 0

    for idx in range(len(words)):
        w, r, t = words[idx], rels[idx], tags[idx]

        if ((r_prev in [rd.x2i['cc'], rd.x2i['punct']]) and (r == rd.x2i['conj'])) or\
            r == rd.x2i['dep'] and t in posfunc:
            # (rels[idx - 1] in [rd.x2i['cc']] and r in [rd.x2i['compound']] and rels[idx + 1] in [rd.x2i['conj']]) or \
            ret.append(1)
            # flag_cont = True
        elif r in indp or \
                (r_prev == rd.x2i['nmod'] and r not in func) or \
                (r == rd.x2i['cc'] and r_prev == 0) or \
                (r_prev == rd.x2i['conj'] and r == rd.x2i['nmod']) or \
                (r == rd.x2i['dep'] and t in poscont) or \
                (r_prev not in prefix and r in subob) or \
                (w in func_begin) or \
                (words[idx - 1] in func_end):
            ret.append(0)
            flag_cont = False
        elif ((w in func_begin or
              r not in func)) \
            and not flag_cont:
            ret.append(0)
            flag_cont = True
        else:
            ret.append(1)


        # if r in func and not (r == rd.x2i['dep'] and t == td.x2i['NOUN']):
        if r in func or (r == rd.x2i['dep'] and t == td.x2i['ADP']):
            flag_cont = False
        if r == rd.x2i['punct'] and t == td.x2i['SYM'] and rels[idx + 1] == rd.x2i['conj']:
            flag_cont = True

        r_prev = r

    return ret


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

def complete_sents(inters, intras, heads_chunk, bi_chunk):
    ret = []
    chunk_ranges = ranges(bi_chunk)

    for idx, head in enumerate(heads_chunk):
        start_chd, end_chd = chunk_ranges[idx + 1]
        start_par, end_par = chunk_ranges[head]
        intra = intras[idx]
        inter = inters[idx]

        tmp = [[] for i in range(len(inter))]
        # que = [i for i in intra if i > len(inter) - 1]

        for idx in range(len(inter)):
            if inter[idx] >= len(inter):
                tmp[idx] = inter[idx] - (len(inter)) + start_par
            else:
                tmp[idx] = inter[idx] + start_chd - 1

        ret.extend(tmp)

    return ret










def isAncestorOf(anc_idx, des_heads):
    set_des = set()
    set_des.add(anc_idx)
    set_des_size_prev = 0
    des_indices = [False] * len(des_heads)
    if anc_idx >= len(des_indices) or anc_idx < 0:
        print('error!')
    des_indices[anc_idx] = True

    while set_des_size_prev != len(set_des):
        for d_idx, dh in enumerate(des_heads):
            if dh in set_des:
                set_des.add(d_idx)
                des_indices[d_idx] = True

        set_des_size_prev = len(set_des)
    ret_idx = 0
    for di in des_indices:
        if not di:
            break
        ret_idx += 1

    return ret_idx


def ancestorArray(parents):
    parents = [0] + parents
    ret = [[] for p in parents]

    tpls = []
    que = list()

    for pidx, p in enumerate(parents):
        tpls.append((p, pidx))

    tpls.sort()

    que.append(tpls[1])
    prev = []

    while len(que) > 0:
        tmp = que.pop(0)
        ret[tmp[1]].extend([tmp[0]] + ret[tmp[0]])
        que.extend([t for t in tpls if t[0] == tmp[1]])

    # return ret[1:]
    return ret


def convertChunks(bi_chunk, parents, rels, tags, parser, rd, td):
    B_tag_idx = parser._B_tag_id
    # bi_chunk = [0] + bi_chunk
    parents = [0] + parents
    rels = [0] + rels
    tags = [0] + tags

    chunk_ranges = ranges(bi_chunk, B_tag_idx)
    ret_bi = [0] + bi_chunk.copy()
    dep_tuples = []
    set_heads = set([0])
    anc_array = ancestorArray(parents[1:])
    w2ch = align_word_chunk(bi_chunk, B_tag_idx)

    for ridx, r in enumerate(chunk_ranges[1:]):
        start = r[0]
        end = r[1]
        heads_in_chunk = []
        set_parents = set()

        for w in range(start, end):
            if (parents[w] < start or end <= parents[w]) and parents[w] not in set_parents:
                # set_heads.add(w)
                # dep_tuples[-1].append((w, heads[w]))
                heads_in_chunk.append(w)
                set_heads.add(w)
                set_parents.add(parents[w])
                dep_tuples.append((w, parents[w]))

            # if rels[w] == parser._mwe_id:
            #     ret_bi[w] = 1
            # if rels[w] == parser._acl_id:
            #     ret_bi[w] = 0
            # # if rels[w - 1] == parser._conj_id and rels[w] == parser._case_id:
            # #     ret_bi[w - 1] = 1
            # if rels[w - 1] == rd.x2i['compound'] and rels[w] == rd.x2i['nmod'] and rels[w + 1] == rd.x2i['case']:
            #     ret_bi[w] = 0
            # if rels[w - 1] == rd.x2i['cc'] and rels[w] == rd.x2i['compound'] and rels[w + 1] == rd.x2i['conj']:
            #     ret_bi[w] = 1
            # if rels[w - 1] == rd.x2i['cc'] and rels[w] == rd.x2i['conj'] and rels[w + 1] == rd.x2i['case']:
            #     ret_bi[w] = 1
            # if rels[w] == rd.x2i['advcl']:
            #     ret_bi[w] = 0
            # if rels[w - 1] == rd.x2i['nummod'] and rels[w] == rd.x2i['punct'] and rels[w + 1] == rd.x2i['dep']:
            #     ret_bi[w + 1] = 1
            # if rels[w - 1] == rd.x2i['punct'] and rels[w] == rd.x2i['nummod'] and rels[w + 1] == rd.x2i['compound']:
            #     ret_bi[w] = 1
            # if rels[w - 1] == rd.x2i['aux'] and rels[w] == rd.x2i['neg'] and rels[w + 1] == rd.x2i['nmod']:
            #     ret_bi[w + 1] = 0
            # if rels[w - 1] == rd.x2i['case'] and rels[w] == rd.x2i['dep'] and rels[w + 1] == rd.x2i['dep']:
            #     ret_bi[w] = 0
            # if rels[w - 1] == rd.x2i['case'] and rels[w] == rd.x2i['acl'] and rels[w + 1] == rd.x2i['nmod']:
            #     ret_bi[w + 1] = 0
            # if rels[w - 1] == rd.x2i['acl'] and rels[w] == rd.x2i['root']:
            #     ret_bi[w] = 0
            # if rels[w - 1] == rd.x2i['nmod'] and rels[w] == rd.x2i['case'] and rels[w + 1] == rd.x2i['case']:
            #     ret_bi[w - 1] = 0
            # if rels[w - 1] == rd.x2i['conj'] and rels[w] == rd.x2i['punct'] and rels[w + 1] == rd.x2i['conj']:
            #     ret_bi[w - 1] = 0
            #     ret_bi[w + 1] = 0
            # if rels[w] == rd.x2i['conj'] and rels[w + 1] == rd.x2i['case']:
            #     ret_bi[w] = 0

            # if tags[w - 1] == td.x2i['CONJ'] and tags[w] == td.x2i['NOUN']:
            #     ret_bi[w] = 1

        # dep_tuples.append([])

        while len(heads_in_chunk) > 1:
            # if heads[heads_in_chunk[0]] == heads_in_chunk[0] - 1:
            #     ret_bi[heads_in_chunk[0]] = 1
            #     heads_in_chunk.pop(0)
            #     continue
            # chunk_id = w2ch[heads_in_chunk[0]]
            head_idx = heads_in_chunk[0]

            for ridx in range(start, end):
                if head_idx not in anc_array[ridx]:
                    ret_bi[ridx] = 0
                    heads_in_chunk.pop(0)
                    break

            if len(heads_in_chunk) > 1:
                heads_in_chunk.pop(0)

            # new_start = isAncestorOf(heads_in_chunk[0] - start, parents[start:end])
            # if len(ret_bi) <= start + new_start:
            #     heads_in_chunk.pop(0)
            #     continue
            # ret_bi[start + new_start] = B_tag_idx
            # heads_in_chunk.pop(0)

    chunk_ranges = ranges(ret_bi, B_tag_idx)
    w2ch = align_word_chunk(ret_bi, B_tag_idx)

    # for ridx, r in enumerate(chunk_ranges[1:]):
    #     start = r[0]
    #     end = r[1]
    #     heads_in_chunk = []
    #
    #     for w in range(start, end):
    #         if parents[w] < start or end <= parents[w]:
    #             # set_heads.add(w)
    #             dep_tuples.append((w, parents[w]))
    #             # heads_in_chunk.append(w)

    for dt in dep_tuples:
        if not dt[1] in set_heads:
            r = chunk_ranges[w2ch[dt[1]]]
            start, end = r[0], r[1]
            head_idx = dt[1]

            for ridx in range(start, end):
                if head_idx not in anc_array[ridx]:
                    ret_bi[ridx] = 0
                    break

    return ret_bi[1:]


#
# def convertChunks(bi_chunk, parents, rels, tags, parser, rd, td):
#     # add root index into each feature array
#
#     B_tag_idx = parser._B_tag_id
#     bi_chunk = [0] + bi_chunk
#     parents = [0] + parents
#     rels = [0] + rels
#     tags = [0] + tags
#
#     chunk_ranges = ranges(bi_chunk, B_tag_idx)
#     ret_bi = [0] + bi_chunk.copy()
#
#     # tuple of (head in a chunk, its parent)
#     dep_tuples = []
#
#     # set of tokens which has at least one child
#     set_chunk_heads = set([0])
#     w2ch = align_word_chunk(bi_chunk, B_tag_idx)
#
#     for ridx, r in enumerate(chunk_ranges[1:]):
#         start = r[0]
#         end = r[1]
#         heads_in_chunk = []
#
#         for w in range(start, end):
#             if parents[w] < start or end <= parents[w]:
#                 # set_heads.add(w)
#                 # dep_tuples[-1].append((w, heads[w]))
#                 set_chunk_heads.add(w)
#                 heads_in_chunk.append(w)
#                 dep_tuples.append((w, parents[w]))
#             # if rels[w] == parser._mwe_id:
#             #     ret_bi[w] = 1
#             # if rels[w] == parser._acl_id:
#             #     ret_bi[w] = 0
#             # # if rels[w - 1] == parser._conj_id and rels[w] == parser._case_id:
#             # #     ret_bi[w - 1] = 1
#             # if rels[w - 1] == rd.x2i['compound'] and rels[w] == rd.x2i['nmod'] and rels[w + 1] == rd.x2i['case']:
#             #     ret_bi[w] = 0
#             # if rels[w - 1] == rd.x2i['cc'] and rels[w] == rd.x2i['compound'] and rels[w + 1] == rd.x2i['conj']:
#             #     ret_bi[w] = 1
#             # if rels[w - 1] == rd.x2i['cc'] and rels[w] == rd.x2i['conj'] and rels[w + 1] == rd.x2i['case']:
#             #     ret_bi[w] = 1
#             # if rels[w] == rd.x2i['advcl']:
#             #     ret_bi[w] = 0
#             # if rels[w - 1] == rd.x2i['nummod'] and rels[w] == rd.x2i['punct'] and rels[w + 1] == rd.x2i['dep']:
#             #     ret_bi[w + 1] = 1
#             # if rels[w - 1] == rd.x2i['punct'] and rels[w] == rd.x2i['nummod'] and rels[w + 1] == rd.x2i['compound']:
#             #     ret_bi[w] = 1
#             # if rels[w - 1] == rd.x2i['aux'] and rels[w] == rd.x2i['neg'] and rels[w + 1] == rd.x2i['nmod']:
#             #     ret_bi[w + 1] = 0
#             # if rels[w - 1] == rd.x2i['case'] and rels[w] == rd.x2i['dep'] and rels[w + 1] == rd.x2i['dep']:
#             #     ret_bi[w] = 0
#             # if rels[w - 1] == rd.x2i['case'] and rels[w] == rd.x2i['acl'] and rels[w + 1] == rd.x2i['nmod']:
#             #     ret_bi[w + 1] = 0
#             # if rels[w - 1] == rd.x2i['acl'] and rels[w] == rd.x2i['root']:
#             #     ret_bi[w] = 0
#             # if rels[w - 1] == rd.x2i['nmod'] and rels[w] == rd.x2i['case'] and rels[w + 1] == rd.x2i['case']:
#             #     ret_bi[w - 1] = 0
#             # if rels[w - 1] == rd.x2i['conj'] and rels[w] == rd.x2i['punct'] and rels[w + 1] == rd.x2i['conj']:
#             #     ret_bi[w - 1] = 0
#             #     ret_bi[w + 1] = 0
#             # if rels[w] == rd.x2i['conj'] and rels[w + 1] == rd.x2i['case']:
#             #     ret_bi[w] = 0
#
#             # if tags[w - 1] == td.x2i['CONJ'] and tags[w] == td.x2i['NOUN']:
#             #     ret_bi[w] = 1
#
#         # dep_tuples.append([])
#
#         while len(heads_in_chunk) > 1:
#             # if heads[heads_in_chunk[0]] == heads_in_chunk[0] - 1:
#             #     ret_bi[heads_in_chunk[0]] = 1
#             #     heads_in_chunk.pop(0)
#             #     continue
#             new_start = newChunk(heads_in_chunk[0], parents, start, end)
#             if len(ret_bi) <= start + new_start:
#                 heads_in_chunk.pop(0)
#                 continue
#             ret_bi[start + new_start] = B_tag_idx
#             heads_in_chunk.pop(0)
#
#     chunk_ranges = ranges(ret_bi, B_tag_idx)
#     w2ch = align_word_chunk(ret_bi, B_tag_idx)
#
#     for ridx, r in enumerate(chunk_ranges[1:]):
#         start = r[0]
#         end = r[1]
#         heads_in_chunk = []
#
#         for w in range(start, end):
#             if heads[w] < start or end < heads[w]:
#                 set_heads.add(w)
#                 dep_tuples.append((w, heads[w]))
#                 heads_in_chunk.append(w)
#
#     for didx, dt in enumerate(dep_tuples):
#         if not dt[1] in set_heads:
#             r = chunk_ranges[w2ch[dt[1]]]
#             start, end = r[0], r[1]
#
#             new_start = isAncestorOf(dt[1] - start, heads[start:end])
#             ret_bi[start + new_start] = B_tag_idx
#
#     return ret_bi[1:]
#
#
#
#
#



def align_word_chunk(bi_chunk, B_tag_idx=0, with_root=True):
    word2chunk = [0]

    if not with_root:
        bi_chunk = [0] + bi_chunk

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


def biLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0.):
    fs, bs = inputs, inputs

    for fb, bb in builders:
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        f, b = fb.initial_state(), bb.initial_state()
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        inputs = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]

    bs = [b for b in reversed(bs)]

    return inputs, fs, bs


def segment_embds(l2r_outs, r2l_outs, ranges, offset=0, segment_concat=False):
    ret = []

    for r in ranges:
        start = r[0] + offset
        end = r[1] + offset

        if segment_concat:
            l2r = l2r_outs[end - 1] + l2r_outs[start]
            r2l = r2l_outs[start] + r2l_outs[end - 1]
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
