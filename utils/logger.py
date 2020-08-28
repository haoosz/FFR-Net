import os
from collections import OrderedDict
import numpy as np
from .utils import mkdirs
from tensorboardX import SummaryWriter
from datetime import datetime
import socket

class Logger():
    def __init__(self, opts):
        self.opts = opts
        self.log_dir = os.path.join(opts.log_dir, opts.save_weight_dir)
        self.phase_keys = ['train', 'val', 'test']
        self.iter_log = []
        self.epoch_log = OrderedDict() 
        self.mk_log_file()
        self.set_mode(opts.phase)

        # Generate experiment records in format 'events_DATETIME_HOSTNAME' 
        save_events_dir = 'exp_{}_{}'.format(
                datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                socket.gethostname()
                )

        self.writer = SummaryWriter(os.path.join(self.log_dir, save_events_dir))

    def mk_log_file(self):
        mkdirs(self.log_dir)
        self.txt_files = OrderedDict()
        for i in self.phase_keys:
            self.txt_files[i] = os.path.join(self.log_dir, 'log_{}'.format(i))

    def set_mode(self, mode):
        self.mode = mode
        self.epoch_log[mode] = []

    def set_current_iter(self, cur_iter, cur_epoch=None):
        self.cur_iter = cur_iter
        self.cur_epoch = cur_epoch
        self.iter_log.append(OrderedDict())
        
    def record_scalar(self, items, tag):
        """
        iteration log: [iter]{key: value}
        """
        self.iter_log[-1].update(items)
        for k, v in items.items():
            self.writer.add_scalar('{}/{}'.format(tag, k), float(v), self.cur_iter)

    def print_scalar(self, items):
        self.iter_log[-1].update(items)
    
    def record_images(self, visuals, nrow=6, tag='ckpt_image'):
        imgs = []
        for i in range(nrow):
            tmp_imgs = [x[i] for x in visuals]
            imgs.append(np.hstack(tmp_imgs))
        imgs = np.vstack(imgs)
        self.writer.add_image(tag, imgs.astype(np.uint8), self.cur_iter,dataformats='HWC')

    def record_single_image(self, img, tag):
        self.writer.add_image(tag, img.astype(np.uint8), self.cur_iter,dataformats='HWC')

    def record_text(self, tag, text):
        self.writer.add_text(tag, text) 

    def printIterSummary(self, in_res, total_it, total_epoch, timer):
        msg = '{}\nInRes: {}x\tEpoch[Iter]: {:03d}/{:03d}[{:03d}/{:03d}]\t'.format(
                timer.to_string(total_it - self.cur_iter), in_res, 
                self.cur_epoch, total_epoch,
                self.cur_iter, total_it)
        for k, v in self.iter_log[-1].items():
            msg += '{}: {}\t'.format(k, v) 
        print(msg + '\n')
        with open(self.txt_files[self.mode], 'a+') as f:
            f.write(msg + '\n')

    def close(self):
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, 'all_scalars.json'))
        self.writer.close()




