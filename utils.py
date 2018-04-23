import dynet as dy
import numpy as np
import config

def bilstm(l2rlstm, r2llstm, inputs, pdrop):
    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    l2rlstm.set_dropouts(pdrop, pdrop)
    r2llstm.set_dropouts(pdrop, pdrop)

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


def biED(x, V_r, V_i, y, seq_len, num_outputs, bias):

    W_r = dy.concatenate_cols([V_r] * seq_len)
    W_i = dy.concatenate_cols([V_i] * seq_len)

    input_size = x.dim()[0][0]

    x_r, x_i = x[:- input_size // 2], x[input_size // 2:]
    y_r, y_i = y[:- input_size // 2], y[input_size // 2:]

    B = dy.inputTensor(np.zeros((seq_len * num_outputs, seq_len), dtype=np.float32))

    # if bias_r and bias_i:
    #     bias = dy.reshape(dy.concatenate_cols([dy.concatenate([bias_r, bias_i]) * seq_len]), (seq_len * num_outputs, input_size))
    #     B += bias * dy.concatenate([x_r, x_i])
    # if bias_i:
    #     bias = dy.reshape(dy.concatenate_cols([bias_i] * seq_len), (seq_len * num_outputs, input_size // 2))
    #     B += bias * x_i
    if bias and config.add_bias:
        bias_R = dy.transpose(dy.reshape(dy.concatenate([(dy.reshape(bias, (input_size, num_outputs)))] * seq_len), (input_size, num_outputs * seq_len)))
        # bias_R = dy.reshape(bias_T, (seq_len * num_outputs, input_size))
        # tmp = dy.reshape(dy.concatenate_cols([bias] * seq_len), (seq_len * num_outputs, input_size))
        # B += tmp * x
        B += bias_R * x
        # tmp = dy.reshape(dy.concatenate_cols([bias] * seq_len), (seq_len * num_outputs, input_size))
        B += dy.concatenate(y * [num_outputs]) * dy.concatenate_cols([bias] * seq_len)
        # B += dy.reshape(dy.transpose(bias_R * y), (seq_len * num_outputs, seq_len))
        # B += dy.reshape(tmp * y, (seq_len * num_outputs, seq_len))

    y_r = dy.concatenate([y_r] * num_outputs)
    y_i = dy.concatenate([y_i] * num_outputs)

    X = dy.concatenate([x_r, x_i, x_r, -x_i])
    Y = dy.concatenate([y_r, y_i, y_i, -y_r])
    W = dy.concatenate([W_r, W_r, W_i, -W_i])
    WY = dy.reshape(dy.cmult(W, Y), (input_size // 2 * 4, seq_len * num_outputs))
    blin = dy.transpose(X) * WY + dy.reshape(B, (seq_len, seq_len * num_outputs))

    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len))

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
    tries = 0 if config.orthonormal and not config.isTest else 10
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


def biLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0.):
    for fb, bb in builders:
        f, b = fb.initial_state(), bb.initial_state()
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        inputs = [dy.concatenate([f,b]) for f, b in zip(fs, reversed(bs))]
    return inputs
