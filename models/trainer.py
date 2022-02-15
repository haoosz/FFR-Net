import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from pretrain.model_ir_se50 import Backbone, ir_se_50_512
import sys
sys.path.append('..')
from utils import utils
from utils.adabound import  AdaBound
import torchvision as tv
from models.recnet import RecNet, selfSimilarity, cosine_sim, init_weights

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.nelement()
    print(net)
    print('Total number of parameters: {}'.format(num_params))

def generate_labelMap(label):
    Map = label.unsqueeze(1).repeat(1,label.size(0))
    labelMap = (Map == Map.t()).int()
    labelMap[labelMap==0]=-1
    labelMap = -labelMap
    return labelMap

class TripletLoss(nn.Module):
    """Recognition Feature Loss
    """
    def __init__(self):
        super(TripletLoss, self).__init__()
        return

    def forward(self, x_feat, y_feat, z_feat):
        margin = 0.1

        pos_cos = 1 - torch.sum(torch.mul(F.normalize(x_feat),F.normalize(y_feat)),1) # non
        neg_cos = 1 - torch.sum(torch.mul(F.normalize(x_feat),F.normalize(z_feat)),1) # ocl
        return F.relu((pos_cos - neg_cos) + margin).mean(), pos_cos.mean(), neg_cos.mean()


def normalization(map):
        max = np.max(np.max(map,1),1)[:, np.newaxis, np.newaxis]
        min = np.min(np.min(map,1),1)[:, np.newaxis, np.newaxis]
        _range = max - min
        return (map - min) / _range

class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.isTrain = self.opts.phase.lower() == 'train'
        self.norm_type = 'bn'
        self.relu_type = 'prelu'
        self.lr = self.opts.lr
        self.encoder = ir_se_50_512()
        self.recnet = RecNet(norm_type=self.norm_type, relu_type=self.relu_type)

        for param in self.encoder.parameters():
            param.requires_grad = False

        init_type = 'kaiming'
        init_weights(self.recnet, init_type)

        if self.opts.gpus > 0:
            self.encoder.to(self.opts.device)
            self.forward_encoder = lambda x: nn.parallel.data_parallel(self.encoder, x, self.opts.gpu_ids) 
            self.recnet.to(self.opts.device)
            self.forward_recnet = lambda x, label: nn.parallel.data_parallel(self.recnet, (x,label), self.opts.gpu_ids)

        if opts.phase.lower() == 'test' or opts.phase == 'val':
            self.encoder.eval()
            self.recnet.eval()

        if self.isTrain:
            self.encoder.eval()
            self.recnet.train()
            self.config_optimizer()

            mile_stones = [5000, 10000, 15000]
            gamma = 0.5
            self.sch = optim.lr_scheduler.MultiStepLR(self.optim, mile_stones, gamma=gamma)

        self.config_criterion()

        if not self.isTrain or opts.continue_train:
            self.load_model(opts.which_file)

        if self.isTrain:
            print('--------------------- Network initialized -----------------------')
            print_network(self.encoder)
            print_network(self.recnet)
            print('-----------------------------------------------------------------')

    def clone_model(self):
        net_copy = {
            'Senet': Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se'),
            'Recnet': RecNet(norm_type=self.norm_type, relu_type=self.relu_type),
        }
        net_copy['Senet'].load_state_dict(self.encoder.state_dict())
        net_copy['Recnet'].load_state_dict(self.recnet.state_dict())

        for param in net_copy['Senet'].parameters():
            param.requires_grad = False

        if self.opts.gpus > 0: 
            net_copy['Senet'].to(self.opts.device)
            net_copy['Recnet'].to(self.opts.device)
            
        return net_copy 

    def config_optimizer(self):
        optimizer_name = self.opts.optimizer.lower()
        param = [x for x in self.recnet.parameters() if x.requires_grad]

        if optimizer_name == 'adam':
            self.optim = optim.Adam(param, self.opts.lr, betas=(self.opts.beta1, self.opts.beta2), weight_decay=self.opts.weight_decay)
        elif optimizer_name == 'rmsprop':
            self.optim    = optim.RMSprop(param, self.opts.lr, momentum=self.opts.momentum, weight_decay=self.opts.weight_decay)
        elif optimizer_name == 'sgd':
            self.optim    = optim.SGD(param, lr=self.opts.lr, momentum=self.opts.momentum, weight_decay=self.opts.weight_decay)
        elif optimizer_name == 'adabound':
            flr = self.opts.lr*100
            self.optim = AdaBound(param, self.opts.lr, final_lr=flr, betas=(self.opts.beta1, self.opts.beta2), weight_decay=self.opts.weight_decay)
        
    def set_input(self, img1, img2, label):
        self.nonocl = img1
        self.ocl = img2
        self.gt_label = label

    def config_criterion(self):
        self.mse_loss = nn.MSELoss()
        self.triplet = TripletLoss()
        self.cross_entropy = nn.CrossEntropyLoss() 

    def forward(self):
        # Generator forward and label prediction in training stage
        self.feat_map_non, self.feat_extract_non = self.forward_encoder(self.nonocl)
        self.feat_map_ocl, self.feat_extract_ocl = self.forward_encoder(self.ocl)

        self.f_non, self.pred_loss_non, self.pred_label_non, self.M_space_non, self.M_channel_non, self.space_non, self.channel_non = self.forward_recnet(self.feat_map_non, self.gt_label)
        self.f_ocl, self.pred_loss_ocl, self.pred_label_ocl, self.M_space_ocl, self.M_channel_ocl, self.space_ocl, self.channel_ocl = self.forward_recnet(self.feat_map_ocl, self.gt_label)

        # ocl accuracy
        label_prob = self.pred_label_ocl # (cos_theta, phi_theta)
        _, pred_label = torch.max(label_prob.data, 1) 
        correct = pred_label.eq(self.gt_label.data).cpu().sum().item()
        self.accuracy = correct / pred_label.shape[0]
        self.pred_label = pred_label

    def backward(self):
        self.loss_items = []
        
        self.ss_space, self.ss_channel = selfSimilarity(self.feat_map_non)
        self.ss_space_non, _ = selfSimilarity(self.space_non)
        self.ss_space_ocl, _ = selfSimilarity(self.space_ocl)
        _, self.ss_channel_non = selfSimilarity(self.channel_non)
        _, self.ss_channel_ocl = selfSimilarity(self.channel_ocl)

        # self similarity loss
        ss_space_loss = (self.mse_loss(self.ss_space,self.ss_space_non) + self.mse_loss(self.ss_space,self.ss_space_ocl))/2
        ss_channel_loss = (self.mse_loss(self.ss_channel,self.ss_channel_non) + self.mse_loss(self.ss_channel,self.ss_channel_ocl))/2
        self.loss_items.append((ss_space_loss + ss_channel_loss)/2)
        #  triplet loss
        triplet_loss, self.pos_loss, self.neg_loss = self.triplet(self.f_ocl, self.feat_extract_non, self.feat_extract_ocl)
        self.loss_items.append(triplet_loss)
        # Identity loss
        self.loss_items.append((self.mse_loss(self.f_non, self.feat_extract_non) + self.mse_loss(self.f_ocl, self.feat_extract_non))/2)
        #  classifier loss
        self.loss_items.append(
            self.cross_entropy(self.pred_loss_non, self.gt_label) / (1e-8 + self.opts.loss_weight[3])  \
            + self.cross_entropy(self.pred_loss_ocl, self.gt_label) 
            )
        
        self.loss_items = [l * w for l, w in zip(self.loss_items, self.opts.loss_weight)]
        loss = sum(self.loss_items)
        loss.backward()

    def optimizer_parameters(self,cur_iters):
        max_clip_value = 1.0
        self.optim.zero_grad()
        self.backward()
        clip_grad_value_(self.recnet.parameters(), max_clip_value)
        self.optim.step()
       
    def get_current_values(self):
        loss_keys = ['SelfSimilarityLoss', 'TripletLoss', 'IdentityLoss', 'ClassifierLoss']
        value_dict = OrderedDict()
        for key, loss in zip(loss_keys, self.loss_items):
            loss_value = loss.item()
            value_dict[key] = '{:.4f}'.format(loss_value)
        value_dict['TrainAcc']  = '{:.4f}'.format(self.accuracy)

        new_keys = ['SelfSimilarityLoss', 'TripletLoss', 'IdentityLoss', 'ClassifierLoss', 'TrainAcc']
        new_value_dict = OrderedDict((k, value_dict[k]) for k in new_keys if k in value_dict.keys())
        return new_value_dict

    def load_model(self, file_name):
        if file_name == 'latest':
            weights = sorted([x for x in os.listdir(self.opts.ckpt_dir) if x.endswith('pth.gzip')])
            file_name = weights[-1]
        else:
            file_name = file_name + '.pth.gzip'
        file_path = os.path.join(self.opts.ckpt_dir, file_name)
        if '/' in file_name:
            file_path = file_name
        weights = utils.load(file_path)

        self.recnet.load_state_dict(weights['RecNet'], strict=False)
        # self.optim.load_state_dict(weights['optimizer'])
        self.start_point = {'epoch': weights['epoch'], 'iter': weights['iter']}

    def save_model(self, file_name, extra_info=None):
        weight_dict = {
                'RecNet': self.recnet.state_dict(),
                'optimizer': self.optim.state_dict(),
                }
        if extra_info is not None:
            weight_dict.update(extra_info)
        file_path = os.path.join(self.opts.ckpt_dir, file_name + '.pth.gzip')
        utils.save(weight_dict, file_path)
        
    def get_lr(self):
        return {'LR': '{:.6f}'.format(self.lr)}

    def get_pos(self):
        return self.pos_loss

    def get_neg(self):
        return self.neg_loss

    def update_learning_rate(self):
        self.sch.step()
        for param_group in self.optim.param_groups:
            self.lr = param_group['lr']