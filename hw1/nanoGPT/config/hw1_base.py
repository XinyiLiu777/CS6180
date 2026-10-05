# HW1 shared config: scaled-down train_shakespeare_char.py so all 6 runs fit on an 8GB M1 (MPS).
# Every experiment uses this file and only overrides the architecture flag(s) under study.
exec(open('config/train_shakespeare_char.py').read())

device = 'mps'
compile = False

batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
warmup_iters = 100

eval_interval = 250
eval_iters = 50
log_interval = 50
