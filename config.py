input_dim = 100
hidden_dim = 400
pdrop = 0.33
pdrop_embs = 0.33
pdrop_lstm = 0.33
layers = 1
mlp_dim = 600
arc_dim = 500
layers_word = 2
layers_chunk = 1

all_100 = not True
if all_100:
    input_dim = hidden_dim = mlp_dim = arc_dim = 100
    mlp_dim += 100
    layers_word = 1
    layers_chunk = 1

biaffine_bias_x_arc = True
biaffine_bias_y_arc = False
biaffine_bias_x_rel = True
biaffine_bias_y_rel = True

epoc = 1000
batch_size = 32

use_annealing = True
learning_rate = 0.002
decay = 0.75
# decay_steps = num_sent_in_iter // batch_size
decay_steps = 5000
beta_1 = .9
beta_2 = .9
epsilon = 1e-12

show_acc = 1000

minimal_count = 2
early_stop = 30
save_file = "parameter"
load_file_num = 34
load_file = str(load_file_num) + "/parameter"

isTest = not True

las = True

overfit = not True

small_data = not True
save = True
random_pickup = True
no_reg = not True

if overfit:
    small_data = True
    save = False
    random_pickup = False
    no_reg = True

if no_reg:
    pdrop = pdrop_lstm = pdrop_embs = 0.0

initial_entries = ['UNK']

const_init = not True
japanese = True
num_sents = 0
