import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from pretrain.senet50_256 import senet50_256, Senet50_256
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

# class AddMarginProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         cos(theta) - m
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.40):
#         super(AddMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', s=' + str(self.s) \
#                + ', m=' + str(self.m) + ')'

class CosLoss(nn.Module):
    """Recognition Feature Loss
    """
    def __init__(self):
        super(CosLoss, self).__init__()
        return

    def forward(self, x_feat, y_feat, z_feat, flag):
        margin = 0.2
        # m = torch.sum(torch.mul(x_feat, y_feat),1)
        # n = torch.mul(torch.sum(x_feat**2,1)**0.5,torch.sum(y_feat**2,1)**0.5)
        # loss1 = torch.div(m,n)

        # m = torch.sum(torch.mul(x_feat, z_feat),1)
        # n = torch.mul(torch.sum(x_feat**2,1)**0.5,torch.sum(z_feat**2,1)**0.5)
        # loss2 = torch.div(m,n)

        pos_cos = torch.sum(torch.mul(F.normalize(x_feat),F.normalize(y_feat)),1)
        neg_cos = torch.sum(torch.mul(F.normalize(x_feat),F.normalize(z_feat)),1)

        return F.relu_type((neg_cos - pos_cos) + margin).mean()

def calculate_loss(cosine, label, s=30.0, m=0.20):
    one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    phi = cosine - m
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= s
    return output

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
        self.encoder = senet50_256()
        self.recnet = RecNet(norm_type=self.norm_type, relu_type=self.relu_type)
        self.flag = 1

        for param in self.encoder.parameters():
            param.requires_grad = False

        init_type = 'kaiming'
        init_weights(self.recnet, init_type)
        #  cosnet.init_weights(self.net_DF, init_type)
        #  cosnet.init_weights(self.net_DP, init_type)

        if self.opts.gpus > 0:
            self.encoder.to(self.opts.device)
            self.forward_encoder = lambda x: nn.parallel.data_parallel(self.encoder, x, self.opts.gpu_ids) 
            self.recnet.to(self.opts.device)
            self.forward_recnet = lambda x: nn.parallel.data_parallel(self.recnet, x, self.opts.gpu_ids)

        if opts.phase.lower() == 'test' or opts.phase == 'val':
            self.encoder.eval()
            self.recnet.eval()

        if self.isTrain:
            self.encoder.eval()
            self.recnet.train()
            self.config_optimizer()

            mile_stones = [10, 20, 30, 40, 50]
            #  mile_stones = [50, 100, 150]
            gamma = 1 
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
            'Senet': Senet50_256(),
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
        self.cos_loss = CosLoss()
        self.cross_entropy = nn.CrossEntropyLoss() 
    def forward(self):
        # Generator forward and label prediction in training stage
        self.feat_map_non, self.feat_extract_non, _ = self.forward_encoder(self.nonocl)
        self.feat_map_ocl, self.feat_extract_ocl, _ = self.forward_encoder(self.ocl)

        # # space + channel
        self.feat_new_v_non, _, _, self.feat_new_non, _, _, self.M_space_non, self.M_channel_non, self.pred_label_non = self.recnet(self.feat_map_non)
        self.feat_new_v_ocl, _, _, self.feat_new_ocl, _, _, self.M_space_ocl, self.M_channel_ocl, self.pred_label_ocl = self.recnet(self.feat_map_ocl)
        
        # # space
        # _, self.feat_new_v_non, _, _, self.feat_new_non, _, self.M_space_non, self.pred_label_non = self.forward_recnet(self.feat_map_non)
        # _, self.feat_new_v_ocl, _, _, self.feat_new_ocl, _, self.M_space_ocl, self.pred_label_ocl = self.forward_recnet(self.feat_map_ocl)

        # ocl accuracy
        label_prob = self.pred_label_ocl # (cos_theta, phi_theta)
        _, pred_label = torch.max(label_prob.data, 1) 
        correct = pred_label.eq(self.gt_label.data).cpu().sum().item()
        self.accuracy = correct / pred_label.shape[0]
        self.pred_label = pred_label

        self.pred_loss_non = calculate_loss(self.pred_label_non, self.gt_label) 
        self.pred_loss_ocl = calculate_loss(self.pred_label_ocl, self.gt_label) 

        #  idx = 1
        #  tv.utils.save_image(self.ocl_input[0], 'ocl_img.jpg', normalize=True)
        #  tv.utils.save_image(self.org_input[0], 'org_img.jpg', normalize=True)
                
        #  ocl_pred = torch.nn.functional.interpolate(self.ocl_pred[5], scale_factor=8, mode='bilinear') 
        #  org_pred = torch.nn.functional.interpolate(self.org_pred[5], scale_factor=8, mode='bilinear') 
        #  ocl_feat = ocl_pred[idx].unsqueeze(1)
        #  org_feat = org_pred[idx].unsqueeze(1)
        #  tv.utils.save_image(ocl_feat, 'ocl_feat_example1.jpg', normalize=True)
        #  tv.utils.save_image(org_feat, 'org_feat_example1.jpg', normalize=True)
        #  exit()

    # def backward_R(self):
    #     loss_R = self.cross_entropy(self.org_pred[0], self.gt_label) # ???
    #     self.loss_r = loss_R
    #     self.loss_rec = self.mse_loss(self.org_pred[3], self.org_input[0])
    #     loss_R += self.loss_rec
    #     loss_R.backward(retain_graph=True)

    # def mask_mse(self, x, y, mask):
    #     b, c, h, w = x.shape
    #     diff = (x * mask - y * mask)**2
    #     avg_diff = torch.sum(diff, (1, 2, 3)) / torch.sum(mask, (1, 2, 3)) 
    #     return avg_diff.mean()

    def backward_g(self):
        self.ss_space_ocl, self.ss_channel_ocl = selfSimilarity(self.feat_new_ocl)
        self.ss_space_non, self.ss_channel_non = selfSimilarity(self.feat_map_non)
        self.loss_items = []
        #  non-occlusion loss
        self.loss_items.append(self.mse_loss(self.feat_new_non, self.feat_map_non))
        #  rectification loss
        self.loss_items.append(self.cos_loss(self.feat_new_v_ocl, self.feat_extract_non, self.feat_extract_ocl, self.flag))
        self.loss_items.append(self.mse_loss(self.feat_new_ocl, self.feat_map_non))
        #  self-similarity loss
        self.loss_items.append(self.mse_loss(self.ss_space_ocl, self.ss_space_non))
        self.loss_items.append(self.mse_loss(self.ss_channel_ocl, self.ss_channel_non))
        #  cross-entropy loss
        self.loss_items.append(
            self.cross_entropy(self.pred_loss_non, self.gt_label) / (1e-8 + self.opts.loss_weight[5])  \
            + self.cross_entropy(self.pred_loss_ocl, self.gt_label) 
            )
        self.loss_items = [l * w for l, w in zip(self.loss_items, self.opts.loss_weight)]
        loss_g = sum(self.loss_items)
        loss_g.backward()

    # def backward_D(self):
    #     fake_feat_f = self.ocl_pred[5].detach()
    #     fake_feat_p = fake_feat_f * (1 - self.ocl_input[1])
    #     real_feat_f = self.org_pred[5].detach()
    #     real_feat_p = real_feat_f * (1 - self.ocl_input[1])

    #     # Fake score calculation
    #     fake_score_f = self.forward_df(fake_feat_f)         
    #     fake_score_p = self.forward_dp(fake_feat_p)
    #     # Real score calculation
    #     real_score_f = self.forward_df(real_feat_f)
    #     real_score_p = self.forward_dp(real_feat_p)

    #     loss_d_f = 0.5 * (self.adv_crit(fake_score_f, self.fake_label) + \
    #             self.adv_crit(real_score_f, self.real_label)) + \
    #             self.dragan_gradient_penalty(real_feat_f, self.forward_df)
    #             #  self.wgan_gradient_penalty(real_feat_f, fake_feat_f, self.forward_df)


    #     loss_d_p = 0.5 * (self.adv_crit(fake_score_p, self.fake_label) + \
    #             self.adv_crit(real_score_p, self.real_label)) + \
    #             self.dragan_gradient_penalty(real_feat_p, self.forward_dp)
    #             #  self.wgan_gradient_penalty(real_feat_p, fake_feat_p, self.forward_dp)

    #     #  loss_d = fake_score_f.mean() - real_score_f.mean() + \
    #             #  fake_score_p.mean() - real_score_p.mean()
        
    #     #  gp = self.wgan_gradient_penalty(real_feat_f, fake_feat_f, self.forward_df) + \
    #             #  self.wgan_gradient_penalty(real_feat_p, fake_feat_p, self.forward_dp)

    #     #  loss_d = (loss_d_f + loss_d_p) * self.opts.loss_weight[2]
    #     #  loss_d = (loss_d_f + loss_d_p) 
    #     loss_d = loss_d_f
    #     loss_d.backward()
    #     self.loss_items.append(loss_d)

    def optimizer_parameters(self,cur_iters):
        max_clip_value = 1.0
        #  clip_value = 1.0
        # ===========================
        # Update G
        # ===========================
        #  ------ Stage one -----------
        #  for k, p in self.net.named_parameters():
            #  if 'feature_ext' in k:
                #  p.requires_grad = True
        #  self.optim.zero_grad()
        #  self.backward_R()
        #  clip_grad_value_(self.net.parameters(), max_clip_value)
        #  self.optim.step()

        # def get_max_norm(parameters, norm_type=2):
        #     total_norm = 0
        #     for p in parameters:
        #         param_norm = p.grad.data.norm(norm_type)
        #         total_norm += param_norm.item() ** norm_type
        #     total_norm = total_norm ** (1. / norm_type)
        #     return total_norm

        # def get_max_value(parameters):
        #     max_value = 0
        #     for p in filter(lambda p: p.grad is not None, parameters):
        #         if p.grad.max().item() > max_value:
        #             max_value = p.grad.max().item()
        #     return max_value

        # ------ Stage two -----------
        #  for k, p in self.net.named_parameters():
            #  if 'feature_ext' in k:
                #  p.requires_grad = False 
        self.optim.zero_grad()
        self.backward_g()
        clip_grad_value_(self.recnet.parameters(), max_clip_value)
        self.optim.step()

        # # ===========================
        # # Update D
        # # ===========================
        # if self.opts.loss_weight[2]:
        #     self.optim_df.zero_grad()
        #     #  self.optim_dp.zero_grad()
        #     self.backward_D()
        #     clip_grad_value_(self.net_DF.parameters(), max_clip_value)
        #     #  clip_grad_value_(self.net_DP.parameters(), max_clip_value)
        #     #  print('Max gradient value and norm Full D:\t', get_max_norm(self.net_DF.parameters()), get_max_value(self.net_DF.parameters()))
        #     #  print('Max gradient value and norm Partial D:\t', get_max_norm(self.net_DP.parameters()), get_max_value(self.net_DP.parameters()))
        #     #  clip_grad_norm_(self.net_DF.parameters(), max_clip_norm_D)
        #     #  clip_grad_norm_(self.net_DP.parameters(), max_clip_norm_D)
        #     self.optim_df.step()
        #     #  self.optim_dp.step()
       
    def get_current_values(self):
        loss_keys = ['NonoclLoss', 'CosLoss', 'OclLoss','SpaceSSLoss', 'ChannelSSLoss', 'CrossEntropyLoss']
        value_dict = OrderedDict()
        for key, loss in zip(loss_keys, self.loss_items):
            loss_value = loss.item()
            #  if key == 'CosLoss':
                #  loss_value = loss.item() + self.loss_r.item()
            value_dict[key] = '{:.4f}'.format(loss_value)
        value_dict['TrainAcc']  = '{:.4f}'.format(self.accuracy)

        new_keys = ['NonoclLoss', 'CosLoss', 'OclLoss', 'SpaceSSLoss', 'ChannelSSLoss', 'CrossEntropyLoss', 'TrainAcc']
        new_value_dict = OrderedDict((k, value_dict[k]) for k in new_keys if k in value_dict.keys())
        return new_value_dict

    def get_current_visuals(self):
        # label_text = []
        # for i in range(self.gt_label.shape[0]):
        #     label_text.append('GT:{:05d}\nPred:{:05d}'.format(
        #         self.gt_label[i].item(), self.pred_label[i].item()))
        self.weightmap_ocl = np.ones([self.M_space_ocl.size(0),67,67]) # 67=(7+3)*6+7
        self.weightmap_non = np.ones([self.M_space_non.size(0),67,67]) 
        map_non = utils.tensor_to_numpy(self.M_space_non.view(self.M_space_non.size(0),7,7,7,7))
        map_ocl = utils.tensor_to_numpy(self.M_space_ocl.view(self.M_space_ocl.size(0),7,7,7,7))
        for i in range(7): # H
            for j in range(7): # W
                self.weightmap_non[...,i*10:i*10+7,j*10:j*10+7]=map_non[...,i,j]
                self.weightmap_ocl[...,i*10:i*10+7,j*10:j*10+7]=map_ocl[...,i,j]
        self.weightmap_ocl = self.weightmap_ocl[:,np.newaxis,:,:]*255
        self.weightmap_non = self.weightmap_non[:,np.newaxis,:,:]*255

        channelweight_ocl = utils.tensor_to_numpy(self.M_channel_ocl)[:,np.newaxis,:,:]*255
        channelweight_non = utils.tensor_to_numpy(self.M_channel_non)[:,np.newaxis,:,:]*255
        channel_non = torch.argmax(self.M_channel_non,2).unsqueeze(2).unsqueeze(3).repeat(1,1,7,7)
        channel_ocl = torch.argmax(self.M_channel_ocl,2).unsqueeze(2).unsqueeze(3).repeat(1,1,7,7)

        featmap_non_array = utils.tensor_to_numpy(self.feat_map_non)
        featmap_ocl_array = utils.tensor_to_numpy(self.feat_map_ocl)

        non_featmap = utils.tensor_to_numpy(torch.gather(self.feat_map_non, 1, channel_non))
        ocl_featmap = utils.tensor_to_numpy(torch.gather(self.feat_map_ocl, 1, channel_ocl))

        channelmap_non = normalization(np.mean(non_featmap,1))*255
        channelmap_ocl = normalization(np.mean(ocl_featmap,1))*255

        # for m in range(channelmap_non.shape[0]):
        #         non_featmap = np.mean(featmap_non_array[m,channel_non[m,:],:,:],1)
        #         channelmap_non[m,:,:,:] = normalization(non_featmap) * 255
        #         ocl_featmap = np.mean(featmap_ocl_array[m,channel_ocl[m,:],:,:],1)
        #         channelmap_ocl[m,:,:,:] = normalization(ocl_featmap) * 255
        
        mean = (131.0912, 103.8827, 91.4953)
        nonocl_image = utils.tensor_to_numpy(self.nonocl).transpose(0,2,3,1) + mean
        ocl_image = utils.tensor_to_numpy(self.ocl).transpose(0,2,3,1) + mean

        out = []
        out.append(nonocl_image.transpose(0,3,1,2))
        out.append(self.weightmap_non)
        out.append(channelweight_non)
        out.append(channelmap_non[:,np.newaxis,:,:])

        out.append(ocl_image.transpose(0,3,1,2))
        out.append(self.weightmap_ocl)
        out.append(channelweight_ocl)
        out.append(channelmap_ocl[:,np.newaxis,:,:])

        for i in range(8):
            if i==0 or i==4:
                out[i] = utils.batch_numpy_to_image(out[i])
            else:
                out[i] = utils.batch_numpy_to_image(out[i], size=(224, 224))

        #  feature_vis = self.net.get_feature_vis()
        #  feature_vis = [utils.tensor_to_numpy(x.repeat(1, 3, 1, 1)) for x in feature_vis]
        #  out += [utils.batch_numpy_to_image(x) for x in feature_vis]
        return out

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

    def update_learning_rate(self):
        self.sch.step()
        for param_group in self.optim.param_groups:
            self.lr = param_group['lr']

if __name__ == '__main__':
    # from utils.options import Options
    # opts = Options()
    # opts = opts.parse()

    # trainer = Trainer(opts)
    # in_lr = torch.randn(32, 3, 224, 224).to(opts.device)
    # in_hr = torch.randn(32, 3, 224, 224).to(opts.device)

    # trainer.set_input(in_lr, in_lr)
    # trainer.forward()
    # trainer.optimizer_parameters()
    # trainer.update_learning_rate()
    # print(trainer.get_current_values())
    # print(trainer.get_lr())
    # print([x.shape for x in trainer.get_current_visuals()])
    x=torch.rand(16,5)
    y=torch.rand(16,5)
    label=torch.rand(16)
    loss = cosloss=CosLoss(x,y,label)
    print(loss)