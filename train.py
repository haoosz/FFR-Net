import models.trainer as train_module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.options import Options
from utils.timer import Timer
from utils.logger import Logger
from utils import utils

from data.dataloader import create_dataloader
from lfw.lfw_eval import get_avg_accuracy

import numpy as np
import os
import copy
import collections
from skimage.io import imsave
import torch
import shutil

def train(opts):

    logger = Logger(opts)
    timer = Timer()
    #  data_loader = create_dataloader(opts, 0.5)
    data_loader = create_dataloader(opts)
    trainer = train_module.Trainer(opts)
    if hasattr(trainer, 'start_point'):
        start_epoch = trainer.start_point['epoch'] 
        start_iter = trainer.start_point['iter']
    else:
        start_epoch, start_iter = 0, 0

    cur_iters = start_iter
    total_iters = opts.total_epochs * len(data_loader) 
    eval_scores = []

    logger.record_text('model_config', opts.config_string)
    for epoch in range(start_epoch, opts.total_epochs):
        for i, data in enumerate(data_loader):
            if epoch == start_epoch and i < start_iter: continue
            cur_iters += 1

            logger.set_current_iter(cur_iters, epoch)
            # =================== load data ===============
            in_non = data['img1'].float().to(opts.device)
            in_ocl = data['img2'].float().to(opts.device)
            label = data['label'].to(opts.device)
            timer.update_time('DataTime')
            # =================== model train ===============
            trainer.set_input(in_non, in_ocl, label)
            trainer.forward(), timer.update_time('Forward')
            trainer.optimizer_parameters(cur_iters), timer.update_time('Backward')
            value = trainer.get_current_values()

            logger.record_scalar(value, 'train_values')
            logger.print_scalar(trainer.get_lr())
            # =================== save model and visualize ===============
            if cur_iters % opts.print_freq == 0:
                print('Model Configuration: {}'.format(opts.save_weight_dir))
                logger.printIterSummary(opts.in_res, total_iters, opts.total_epochs, timer)
    
            extra_info = {'epoch': epoch, 'iter': cur_iters}
            if cur_iters % opts.save_freq == 0:
                file_name = '{:07d}'.format(cur_iters) 
                if opts.debug: file_name = 'debug'
                trainer.save_model(file_name, extra_info)

            # Save latest checkpoint
            if cur_iters % (opts.save_freq // 10) == 0:
                trainer.save_model('latest', extra_info)
    
            if cur_iters % opts.eval_freq == 0:
                model = trainer.clone_model()
                # Test accuracy no mask
                acc_new, acc = eval_lfw(opts, model, 0, cur_iters)
                logger.record_scalar({'acc': acc}, 'test_acc/ocl0')
                logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl0')
                print('test result ocl0: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))

                if not opts.debug:
                    # Test accuracy with one mask
                    acc_new, acc = eval_lfw(opts, model, 1, cur_iters)
                    logger.record_scalar({'acc': acc}, 'test_acc/ocl1')
                    logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl1')
                    print('test result ocl1: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))

                    # Test accuracy with two masks
                    acc_new, acc = eval_lfw(opts, model, 2, cur_iters)
                    logger.record_scalar({'acc': acc}, 'test_acc/ocl2')
                    logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl2')
                    print('test result ocl2: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))


            if opts.debug: break
            trainer.update_learning_rate()
    logger.close()


def eval_lfw(opts, model, ocl_num, epoch):
    opts.phase = 'test'
    model['Senet'].eval()
    model['Recnet'].eval()

    encoder = lambda x: model['Senet'](x)
    recnet = lambda x: model['Recnet'](x)

    data_loader = create_dataloader(opts, test_ocl_num=ocl_num)
    acc_new, acc = get_avg_accuracy(encoder,recnet,data_loader)
    opts.phase = 'train'

    return acc_new, acc

def eval_func(model):
    mask_size = [(20, 30), (30, 40), (40, 60)]    
    avg_results = []
    for m in mask_size:
        acc = get_avg_accuracy(model, '/data_sr/LFW', m)
        avg_results.append(acc)
    return avg_results
  
def test(opts):
    if opts.which_file == 'latest':
        weights = sorted([x for x in os.listdir(opts.ckpt_dir) if x.endswith('.pth.gzip')])
        opts.which_file = weights[-1].split('.')[0]

    logger = Logger(opts)
    logger.set_current_iter(0, 0)
    trainer = train_module.Trainer(opts)

    model = trainer.clone_model()
    # Test accuracy no mask
    acc_new, acc = eval_lfw(opts, model, 0, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl0')
    logger.record_scalar({'acc_new': acc_new}, 'test_acc_new/ocl0')
    #  logger.record_single_image(visual_img, 'test/visual_img')
    # Test accuracy with one mask
    acc_new, acc = eval_lfw(opts, model, 1, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl1')
    logger.record_scalar({'acc_new': acc_new}, 'test_acc_new/ocl1')
    #  logger.record_single_image(visual_img, 'test/visual_img_mask')
    # Test accuracy with two masks
    acc_new, acc = eval_lfw(opts, model, 2, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl2')
    logger.record_images({'acc_new': acc_new}, 'test_image/ocl2')
    logger.close()
    
if __name__ == '__main__':
    opts = Options().parse() 
    if opts.phase == 'train':
        train(opts)
    else:
        test(opts)