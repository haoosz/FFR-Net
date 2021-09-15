import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
import math
import numpy as np
from collections import OrderedDict
import torchvision
import sys
sys.path.append('./models')
from transformer.Models import Transformer

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True, use_sn=False, groups=1):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        
        bias = True if norm_type in ['pixel', 'none'] else False 
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias, groups=groups)
        if use_sn:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)
        #  self._init_weights_()

    #  def _init_weights_(self):
        #  nn.init.xavier_normal_(self.conv2d.weight.data)
        #  if hasattr(self.conv2d, 'bias') and self.conv2d.bias is not None:
            #  nn.init.constant_(self.conv2d.bias.data, 0.)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out 

class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, norm_type='bn', normalize_shape=None):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)

class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, inplanes, planes, kernel_size=3, norm_type='none', relu_type='none'):
        super(ResidualBlock, self).__init__()
        conv_args = {'norm_type': norm_type, 'relu_type': relu_type}

        self.conv1 = ConvLayer(inplanes, planes, kernel_size, **conv_args)
        self.conv2 = ConvLayer(planes, planes, kernel_size, **conv_args)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + res 
        return x 

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features=10575, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output, cosine

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class RecNet(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
        super(RecNet, self).__init__()
        self.channel = channel
        self.shape = shape

        conv_args = {'norm_type': norm_type, 'relu_type': relu_type}

        self.trans_channel = Transformer(
        d_k=64,
        d_v=64,
        d_position=512,
        d_model=49,
        d_inner=256,
        n_layers=6,
        n_head=8,
        dropout=0.1)
        
        self.trans_space = Transformer(
        d_k=64,
        d_v=64,
        d_position=49,
        d_model=512,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        dropout=0.1)

        # self.ChannelFlipMerge = nn.Sequential(
        #     ConvLayer(self.channel*2,self.channel, **conv_args),
        #     ResidualBlock(self.channel, self.channel, **conv_args),
        # )

        self.Conv4Merge = nn.Sequential(
                ConvLayer(self.channel*2, self.channel, **conv_args),
                ResidualBlock(self.channel, self.channel, **conv_args),
        )
        self.pool_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        # self.classifier = ArcMarginProduct(self.channel)
        self.classifier = AddMarginProduct(self.channel)

    def forward(self, input, label=None):
        feat_channel = input.view(input.size(0), input.size(1), -1) # N*512*49
        feat_space = input.view(input.size(0), input.size(1), -1).transpose(1,2) # N*49*512
        feat_space, attn_space = self.trans_space(feat_space)
        feat_channel, attn_channel = self.trans_channel(feat_channel)
        feat_space.transpose(1,2)
        feat_space = feat_space.view(input.size(0), input.size(1), 7, 7)
        feat_channel = feat_channel.view(input.size(0), input.size(1), 7, 7)

        feat_cat = torch.cat((feat_space,feat_channel),1) # N, 2*C, H, W
        feat_new = self.Conv4Merge(feat_cat)

        feat_new_v = self.pool_7x7(feat_new).view(feat_new.size(0), -1)

        feat_space_v = self.pool_7x7(feat_space).view(feat_space.size(0), -1)
        feat_channel_v = self.pool_7x7(feat_channel).view(feat_channel.size(0), -1)

        if label is None:
            return feat_new_v, feat_space_v, feat_channel_v, feat_new, feat_space, feat_channel, attn_space, attn_channel
        else:
            pred_loss, pred_label = self.classifier(feat_new_v,label)
            return feat_new_v, feat_space_v, feat_channel_v, feat_new, feat_space, feat_channel, attn_space, attn_channel, pred_loss, pred_label
