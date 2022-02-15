import os
import math
import sys

debug   = int(not __debug__)
phase   = 'train'
# ------------------------ data args ---------------------------- 
train_data = '../CASIA-WebFace_112_align_v1_masked'
# ------------------------ model args ---------------------------- 
model_name        = 'FFR-Net'
optimizer         = 'Adam'
lr = 1e-1
beta1             = 0.9
momentum          = 0.9
weight_decay      = 0 
loss_weight = [1,1,1,1] # ss, triplet, id, cls
# Gnorm_type        = 'none'
# Dgroups           = 1 

# ------------------------ train args ---------------------------- 
gpus              = 4 if not debug else 1
batch_size        = 64 if not debug else 32
total_epochs      = 200 if not debug else 10000 
continue_train    = 0
which_file        = 'latest'
if debug: which_file = 'debug'
print_freq        = 100 if not debug else 1
save_freq         = 4000 if not debug else 100000
eval_freq         = 1000 if not debug else 2 
if phase == 'pretrain':
    other = 'pretrain_model'
elif debug:
    other = 'debug'
else:
    other = 'train'
    other = 'train-featmse'
    other = 'train-pixadv'
if debug:
    other = 'debug'

param = [
        '--phase {}'.format(phase),
        '--gpus {}'.format(gpus),
        '--debug {}'.format(debug),
        '--train_data {}'.format(train_data),
        '--batch_size {}'.format(batch_size),
        '--total_epochs {}'.format(total_epochs),
        '--model_name {}'.format(model_name),
        '--optimizer {}'.format(optimizer),
        '--lr {}'.format(lr),
        '--beta1 {}'.format(beta1),
        '--momentum {}'.format(momentum),
        '--weight_decay {}'.format(weight_decay),
        '--loss_weight {} {} {} {}'.format(*loss_weight),
        '--print_freq {}'.format(print_freq),
        '--save_freq {}'.format(save_freq),
        '--eval_freq {}'.format(eval_freq),
        '--continue_train {}'.format(continue_train),
        '--which_file {}'.format(which_file),
        '--other {}'.format(other),
        ]

os.system('python train.py {}'.format(" ".join(param)))

print('Train done.')
