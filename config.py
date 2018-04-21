input_dim = 100
hidden_dim = 400
pdrop = 0.1
pdrop_embs = 0.33
layers = 3
mlp_dim = 600
arc_dim = 500
orthonormal = False
pret_embs = False

small_model = True
if small_model:
    input_dim = hidden_dim = mlp_dim = arc_dim = 100
    mlp_dim += 100
    layers = 1


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
load_file_num = 14
load_file = str(load_file_num) + "/parameter"

isTest = False
if isTest:
    pdrop = 0.0
    pdrop_embs = 0.0

las = True
biaffine = False

