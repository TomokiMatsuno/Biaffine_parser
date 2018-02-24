input_dim = 100
hidden_dim = 400
pdrop = 0.33
pdrop_embs = 0.33
pdrop_lstm = 0.33
layers = 1
mlp_dim = 600
arc_dim = 500

all_100 = True
if all_100:
    input_dim = hidden_dim = mlp_dim = arc_dim = 100
    mlp_dim += 100


biaffine_bias_x_arc = True
biaffine_bias_y_arc = False
biaffine_bias_x_rel = True
biaffine_bias_y_rel = True

epoc = 1000
batch_size = 32

use_annealing = True
learning_rate = 0.002
decay = 0.9
# decay_steps = num_sent_in_iter // batch_size
decay_steps = 5000
beta_1 = .9
beta_2 = .9
epsilon = 1e-12

show_acc = 1000

minimal_count = 2
early_stop = 30
save_file = "parameter"
load_file_num = 9
load_file = str(load_file_num) + "/parameter"

isTest = False

las = True

small_data = not True
save = False

const_init = not True
japanese = True
