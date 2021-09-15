import models.trainer as train_module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.options import Options
from utils.timer import Timer
from utils.logger import Logger
from utils import utils

from data.dataloader_new import create_dataloader
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
                print('pos_loss: {}'.format(trainer.get_pos()))
                print('neg_loss: {}'.format(trainer.get_neg()))
    
            extra_info = {'epoch': epoch, 'iter': cur_iters}
            if cur_iters % opts.save_freq == 0:
                file_name = '{:07d}'.format(cur_iters) 
                if opts.debug: file_name = 'debug'
                trainer.save_model(file_name, extra_info)

            # Save latest checkpoint
            if cur_iters % (opts.save_freq // 10) == 0:
                trainer.save_model('latest', extra_info)
    
            if cur_iters % opts.visual_freq == 0:
                visual_imgs = trainer.get_current_visuals()
                logger.record_images(visual_imgs, nrow=20)
    
            if cur_iters % opts.eval_freq == 0:
                model = trainer.clone_model()
                # Test accuracy no mask
                acc_new, acc = eval_lfw(opts, model, 0, cur_iters)
                logger.record_scalar({'acc': acc}, 'test_acc/ocl0')
                logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl0')
                print('test result ocl0: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))
                # logger.record_images(visual_imgs, 20, 'test_image/ocl0')
                #  logger.record_single_image(visual_img, 'test/visual_img')
                if not opts.debug:
                    # Test accuracy with one mask
                    acc_new, acc = eval_lfw(opts, model, 1, cur_iters)
                    logger.record_scalar({'acc': acc}, 'test_acc/ocl1')
                    logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl1')
                    print('test result ocl1: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))
                    # logger.record_images(visual_imgs, 20, 'test_image/ocl1')
                    #  logger.record_single_image(visual_img, 'test/visual_img_mask')
                    # Test accuracy with two masks
                    acc_new, acc = eval_lfw(opts, model, 2, cur_iters)
                    logger.record_scalar({'acc': acc}, 'test_acc/ocl2')
                    logger.record_scalar({'acc_new': acc_new}, 'test_acc/ocl2')
                    print('test result ocl2: acc_new {:.4f} acc {:.4f}'.format(acc_new, acc))
                    # logger.record_images(visual_imgs, 20, 'test_image/ocl2')
                    #  logger.record_single_image(visual_img, 'test/visual_img_half_mask')

            # if cur_iters % (opts.visual_freq) == 0:
            #     model = trainer.clone_model()
            #     accs, visual_imgs = eval_ar(model)
            #     logger.record_scalar(accs, 'test_ar')
            #     logger.record_images(visual_imgs, 20, 'test_image_ar')

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

    # model.feature = 2 
    # visual_imgs = []
    # data = next(iter(data_loader))
    # img1, img2 = data['img1'], data['img2']
    # with torch.no_grad():
    #     predicts = model(img1.cuda(), None, None)
    # visual_imgs.append(utils.tensor_to_numpy(img1))
    # visual_imgs.append(utils.tensor_to_numpy(img2))
    # visual_imgs.append(utils.tensor_to_numpy(predicts[0]))
    # visual_imgs.append(utils.tensor_to_numpy(predicts[1]))
    # visual_imgs.append(utils.tensor_to_numpy(predicts[2]))

    # imgs_show = [utils.batch_numpy_to_image(x) for x in visual_imgs[:2]]
    # imgs_show += [utils.batch_numpy_to_image(visual_imgs[2], size=(96, 112), v_range=[0, 1])]
    # imgs_show += [utils.batch_numpy_to_image(x) for x in visual_imgs[3:]]

    # return acc, imgs_show 
    return acc_new, acc

def eval_ar(model):
    from eval_ar import ARDataset, get_acc  
    dataset = ARDataset('../data_occlusion/ar_96_112')

    model.eval()
    model.feature = 1 
    acc_all, acc_sun, acc_sca, data_loader = get_acc(model, dataset)

    model.feature = 2 
    count = 0
    save_dir = './AR_deocc'
    os.makedirs(save_dir, exist_ok=True)
    for img, label, mask in data_loader:
        img, label, mask = next(iter(data_loader))
        visual_imgs = []
        mask = torch.nn.functional.interpolate(mask, scale_factor=1/8)
        with torch.no_grad():
            predicts = model(img.cuda(), None, None)
        visual_imgs.append(utils.tensor_to_numpy(img))
        visual_imgs.append(utils.tensor_to_numpy(mask))
        visual_imgs.append(utils.tensor_to_numpy(predicts[0]))
        visual_imgs.append(utils.tensor_to_numpy(predicts[1]))
        visual_imgs.append(utils.tensor_to_numpy(predicts[2]))

        imgs_show = [utils.batch_numpy_to_image(visual_imgs[0])]
        imgs_show += [utils.batch_numpy_to_image(visual_imgs[1], size=(96, 112), v_range=[0, 1])]
        imgs_show += [utils.batch_numpy_to_image(visual_imgs[2], size=(96, 112), v_range=[0, 1])]
        imgs_show += [utils.batch_numpy_to_image(x) for x in visual_imgs[3:]]

        vis = np.hstack(imgs_show)
        from scipy import misc
        for i in vis:
            save_path = os.path.join(save_dir, '{:05d}.jpg'.format(count))
            misc.imsave(save_path, i)
            count += 1

    accs = {'all': acc_all, 'sunglasses': acc_sun, 'scarf': acc_sca}

    return accs, imgs_show 


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
    acc, visual_imgs = eval_lfw(opts, model, 0, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl0')
    logger.record_images(visual_imgs, 20, 'test_image/ocl0')
    #  logger.record_single_image(visual_img, 'test/visual_img')
    # Test accuracy with one mask
    acc, visual_imgs = eval_lfw(opts, model, 1, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl1')
    logger.record_images(visual_imgs, 20, 'test_image/ocl1')
    #  logger.record_single_image(visual_img, 'test/visual_img_mask')
    # Test accuracy with two masks
    acc, visual_imgs = eval_lfw(opts, model, 2, 0)
    logger.record_scalar({'acc': acc}, 'test_acc/ocl2')
    logger.record_images(visual_imgs, 20, 'test_image/ocl2')
    logger.close()
 
    #  img_names = sorted([x for x in os.listdir(opts.val_data)])

    #  count = 0
    #  for i, imgs in enumerate(data_loader):
        #  hr_imgs = imgs['gt'].to(opts.device) 
        #  lr_imgs = imgs['input'].to(opts.device)
        #  hr_size = (hr_imgs.shape[3], hr_imgs.shape[2])
        #  with torch.no_grad():
            #  out = model(lr_imgs)
            #  if isinstance(out, tuple):
                #  att_maps, pred_imgs = out

        #  lr_pred_imgs = utils.batch_tensor_to_img(pred_imgs, hr_size)
        #  gt_imgs = utils.batch_tensor_to_img(hr_imgs, hr_size)

        #  for j in range(hr_imgs.shape[0]):
            #  imsave(os.path.join(save_dir, img_names[count]),
                #  lr_pred_imgs[j].astype(np.uint8))            
            #  count += 1
    
    #  if not 'wild' in opts.val_data:
        #  print(psnr_ssim.psnr_ssim_dir(opts.val_data, save_dir))
    
if __name__ == '__main__':
    opts = Options().parse() 
    if opts.phase == 'train':
        train(opts)
    else:
        test(opts)