import os
import math
import sys

debug   = int(not __debug__)
phase   = 'train'
#  phase   = 'pretrain'
#  phase   = 'test'
# ------------------------ data args ---------------------------- 
# train_data = 'vgg'
train_data = '/app/CASIA-WebFace_112_align_v1_masked'
#  train_data = 'CASIA'
# ------------------------ model args ---------------------------- 
#  model_name        = 'Ocl_Rec_GAN_resnet50_arcface'
#  model_name        = 'Ocl_Rec_GAN_resnet50_cosface'
#  model_name        = 'Ocl_Rec_GAN_sphere64_cosface'
# model_name        = 'senet50_Simplified_arcFace_noSCLoss_Sigmoid'
# model_name        = 'senet50_Simplified_arcFace_noSCLoss_Softmax'
# model_name        = 'senet50_Simplified_cosFace_noSCLoss_Sigmoid'
# model_name        = 'senet50_Simplified_cosFace_SpaceLoss_Softmax'
model_name        = 'model_without_ss_id'
#  model_name        = 'Ocl_Rec_GAN_resnet50_arcface-C080V4'
#  model_name        = 'Ocl_Rec_GAN_resnet50_arcface-C003V8'
#  model_name        = 'Ocl_Rec_GAN_resnet50_arcface-F000'
#  model_name        = 'Ocl_Rec_GAN_sphere50_arcface'
#  model_name        = 'Ocl_Rec_GAN'
Gnorm_type        = 'none'
Dgroups           = 1 

#  optimizer         = 'SGD'
#  optimizer         = 'RMSprop'
#  optimizer         = 'AdaBound'
optimizer         = 'Adam'
# lr                = 2e-4
# lr = 0.001
# optimizer         = 'SGD'
# lr                = 0.001 
lr = 1e-2
beta1             = 0.9
momentum          = 0.9
weight_decay      = 0 
#  loss_weight       = [1, 1]
#  use_masklabel     = 1
#  loss_weight       = [1, 0]
#  use_masklabel     = 0
# Loss weight: recognition loss, occluder prediction loss, gan loss, reconstruction loss, identity loss
# ------------ Baseline Loss Weight ---------------
# loss_weight = [1, 0, 0, 0, 0]
# ------------ DPG loss weight -------------
#  loss_weight       = [1, 1000, 1e-6, 1000, 1000] if pair_same else [1, 1000, 10, 1000, 1]
#  loss_weight       = [1, 1000, 1e-2, 1000, 1000] if pair_same else [1, 1000, 10, 1000, 1]
#  loss_weight       = [1, 100, 1e-1, 1000, 1000] if pair_same else [1, 1000, 10, 1000, 1]
# loss_weight = [1, 1, 1, 1, 1, 1] # non, triplet, ocl, space, channel, cosFace
loss_weight = [0,1,0,1] # ss, triplet, id, cls
# ------------- fine tune weight -----------
#  loss_weight       = [0, 0, 0, 0, 1000] 

# ------------------------ train args ---------------------------- 
gpus              = 4 if not debug else 1
batch_size        = 64 if not debug else 32
total_epochs      = 200 if not debug else 10000 
#  continue_train    = 1
continue_train    = 0
which_file        = 'latest'
# which_file        = '0032000'
# which_file = '0236000'
#  which_file = '0140000'
#  which_file = './weight/model_Ocl_Rec_GAN_sphere64_arcface-loss_sphere-pretrain_model/latest.pth.gzip'
#  which_file        = './weight/model_Ocl_Rec_GAN_sphere64_cosface-loss_sphere-pretrain_model/latest'
#  which_file = '0276000'
if debug: which_file = 'debug'
# 0 which_file = './pretrain_models/pretrain_wogan.pth.gzip'
#  which_file = './weight/model_partial_sphereface-loss_sphere-pretrain_model_v3/0080000'
#  which_file = './weight/model_partial_sphereface-loss_sphere-pretrain_model_v4_bn/0068000'
print_freq        = 100 if not debug else 1
save_freq         = 4000 if not debug else 100000
visual_freq       = 10 if not debug else 1
eval_freq         = 1000 if not debug else 2 
if phase == 'pretrain':
    other = 'pretrain_model'
elif debug:
    other = 'debug'
else:
    other = 'train'
    other = 'train-featmse'
    other = 'train-pixadv'
#  other             = 'train_Gnorm-{}_weight-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(Gnorm_type, *loss_weight)
#  other = 'finetune_block_{}'.format(bocc_sz)
#  other = 'AR_finetune'
#  other             = 'simple_GAN'
#  other             = 'pretrain_model'
#  other             = 'pretrain_model_v3'
#  other = 'baseline2'
if debug:
    other = 'debug'

param = [
        #  '--gpu_ids {}'.format(gpu_ids),
        '--phase {}'.format(phase),
        '--gpus {}'.format(gpus),
        '--debug {}'.format(debug),
        '--train_data {}'.format(train_data),
        '--batch_size {}'.format(batch_size),
        '--total_epochs {}'.format(total_epochs),
        '--model_name {}'.format(model_name),
        '--Gnorm_type {}'.format(Gnorm_type),
        '--Dgroups {}'.format(Dgroups),
        '--optimizer {}'.format(optimizer),
        '--lr {}'.format(lr),
        '--beta1 {}'.format(beta1),
        '--momentum {}'.format(momentum),
        '--weight_decay {}'.format(weight_decay),
        '--loss_weight {} {} {} {}'.format(*loss_weight),
        '--print_freq {}'.format(print_freq),
        '--save_freq {}'.format(save_freq),
        '--visual_freq {}'.format(visual_freq),
        '--eval_freq {}'.format(eval_freq),
        '--continue_train {}'.format(continue_train),
        '--which_file {}'.format(which_file),
        '--other {}'.format(other),
        ]

os.system('python train.py {}'.format(" ".join(param)))

print('Train done.')
