import argparse
import os
from . import utils
import torch
import numpy as np
import random

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--debug', type=int, default=0, help='debug mode')
        # -----------  Data args ----------------
        self.parser.add_argument('--dataset', type=str, default='CASIA', help='dataset to used for train')
        self.parser.add_argument('--train_data', type=str, default='', help="train data dir")
        self.parser.add_argument('--train_img_list', type=str, default='', help="train data dir")
        self.parser.add_argument('--val_data', type=str, default='', help="validation data path")
        self.parser.add_argument('--weight_root', type=str, default='./weight', help='Weight saving path')
        self.parser.add_argument('--rot_aug', type=str, default='small', help='Data augmentation range.')
        self.parser.add_argument('--in_res', type=int, default=16, help="input low resolution size")
        self.parser.add_argument('--scale', type=int, default=8, help="scale factor")
        self.parser.add_argument('--out_res', type=int, default=128, help="output high resolution size")
        self.parser.add_argument('--inresize', type=int, default=1, help='bicubic resize input or not.')
        self.parser.add_argument('--data_filter', type=str, default='none', help='filter data')
        self.parser.add_argument('--mask_percent', type=int, default=60, help='mask size percentage')
        self.parser.add_argument('--neg_num', type=int, default=1, help='number of negative images')

        # -----------  Model args ----------------
        self.parser.add_argument('--model_name', type=str, default='', help="name of the model")
        self.parser.add_argument('--max_ch', type=int, default=128, help="maximum channels")
        self.parser.add_argument('--use_mask', type=int, default=1, help="output high resolution size")
        self.parser.add_argument('--use_pmask', type=int, default=1, help="output high resolution size")
        self.parser.add_argument('--use_shortcut', type=int, default=1, help='use shortcut_func or not')
        self.parser.add_argument('--use_catt', type=int, default=0, help='use channel attention or not')
        self.parser.add_argument('--use_ms_pool', type=int, default=1, help='use multiscale weighted pool or not')


        # -----------  Generator args ----------------
        self.parser.add_argument('--Gnorm_type', type=str, default='bn', help="normal type of Generator")
        self.parser.add_argument('--Grelu_type', type=str, default='prelu', help="relu type of Generator")

        # -----------  Discriminator args ----------------
        self.parser.add_argument('--Dnorm_type', type=str, default='bn', help="normal type of Discriminator")
        self.parser.add_argument('--Drelu_type', type=str, default='LeakyReLU', help="relu type of Generator")
        self.parser.add_argument('--Dgroups', type=int, default=1, help='groups of discriminator')

        # -----------  Loss args ----------------
        self.parser.add_argument('--loss_weight', type=float, nargs=1, default=[1e0]*5, 
                help="weight of different losses")
        # self.parser.add_argument('--gan_loss', default='gan', type=str, help='loss type: lsgan/gan/dragan.')
        self.parser.add_argument('--loss', default='sphere', type=str, help='main loss type')
        self.parser.add_argument('--use_masklabel', default=0, type=int, help='use mask label or not')

        # -----------  Optimizer args ----------------
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
        self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        self.parser.add_argument('--nesterov', type=int, default=0, help='nesterov for SGD')

        # -----------  Train args ----------------
        self.parser.add_argument('--gpus', type=int, default=1, help='how many gpus to use')
        self.parser.add_argument('--seed', type=int, default=123, help='Random seed for training')
        self.parser.add_argument('--nThread', type=int, default=8, help='Threads used to load data.')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Train and test batch size')
        self.parser.add_argument('--total_epochs', type=int, default=10, help='Total train epochs')

        self.parser.add_argument('--continue_train', type=int, help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--save_freq', type=int, default=2000, help='iterations to save checkpoints')
        self.parser.add_argument('--visual_freq', type=int, default=100, help='epochs to save checkpoints')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--eval_freq', type=int, default=100, help='evaluation frequency')

        # -----------  Test and val args ----------------
        self.parser.add_argument('--test_result_dir', type=str, default='./test_result', help='save test result.')
        self.parser.add_argument('--which_file', type=str, default='latest', help='Test file name, use the latest model by default')
        self.parser.add_argument('--other', type=str, default='', help="Other information")

    def parse(self):
        self.opt = self.parser.parse_args()

        # Find avaliable GPUs automatically
        if self.opt.gpus > 0:
            self.opt.gpu_ids = utils.get_gpu_memory_map()[1][:self.opt.gpus]
            if not isinstance(self.opt.gpu_ids, list):
                self.opt.gpu_ids = [self.opt.gpu_ids]
            self.opt.gpu_ids = [2,3]
            print('Using GPUs: ', self.opt.gpu_ids)
        else:
            self.opt.gpu_ids = []

        # set gpu ids
        if self.opt.gpus > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.opt.device = torch.device('cuda')
        else:
            self.opt.device = torch.device('cpu')
        np.random.seed(self.opt.seed)
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)

        # set mile stones
        #  self.opt.mile_stones = [int(x) for x in self.opt.mile_stones.strip().split(',')]

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.save_weight_dir = 'model_{}-loss_{}-{}'.format(self.opt.model_name, self.opt.loss, self.opt.other)
        self.opt.ckpt_dir = os.path.join(self.opt.weight_root, self.opt.save_weight_dir)
        utils.mkdirs(self.opt.ckpt_dir)

        self.opt.visual_dir = os.path.join(self.opt.ckpt_dir, 'visuals')
        utils.mkdirs(self.opt.visual_dir)
        self.opt.test_dir = os.path.join(self.opt.ckpt_dir, 'test')
        utils.mkdirs(self.opt.test_dir)
        self.opt.log_dir = os.path.join(self.opt.weight_root, 'log_dir')
        utils.mkdirs(self.opt.log_dir)
        self.opt.val_result_dir = os.path.join(self.opt.ckpt_dir, 'val')
        utils.mkdirs(self.opt.val_result_dir)

        file_name = os.path.join(self.opt.ckpt_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt.config_string = '\n'.join([x for x in open(file_name).readlines()])
        return self.opt
