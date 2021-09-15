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
import gzip

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func> 

def load(read_path):
    if read_path.endswith('.gzip'):
        with gzip.open(read_path, 'rb') as f:
            weight = torch.load(f)
    else:
        weight = torch.load(read_path)
    return weight
  
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

class HGBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment 
    --------------------------
    """
    def __init__(self, depth, c_in, c_out,
            c_mid=64,
            norm_type='bn',
            relu_type='prelu',
            ):
        super(HGBlock, self).__init__()
        self.depth     = depth
        self.c_in      = c_in
        self.c_mid     = c_mid
        self.c_out     = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        
        self._generate_network(self.depth)
        self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                )

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs)) 
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs)) 
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs)) 

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x):
        x = self._forward(self.depth, x)
        x = self.out_block(x)
        return x

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

def cosine_sim(x1, x2, dim=1): # N, C, H*W / N, H*W, C
    x1 = F.normalize(x1,dim=2)
    x2 = F.normalize(x2,dim=2)
    ip = torch.bmm(x1, x2.permute(0,2,1))
    return ip

def selfSimilarity(x): # N, C, H, W
    height = x.size(2)
    width = x.size(3)
    x = x.view(x.size(0), x.size(1),-1) # N, C, H*W

    ss_space = cosine_sim(x.permute(0,2,1),x.permute(0,2,1))
    ss_channel = cosine_sim(x,x)

    ss_space = ss_space.view(ss_space.size(0),ss_space.size(1),height,width) # N, H*W, H, W
    
    return ss_space, ss_channel

# def cosine_sim(x1, x2, dim=1, eps=1e-8):
#     ip = torch.mm(x1, x2.t())
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
#     return ip / torch.ger(w1,w2).clamp(min=eps)

# class MarginCosineProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#     """
#     def __init__(self, in_features, out_features=10575):
#         super(MarginCosineProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, input):
#         # cosine = cosine_sim(input, self.weight)
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         # --------------------------- convert label to one-hot ---------------------------
#         # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
#         return cosine

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) 

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

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features=10575, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output, cosine



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x), x

class RecMatrix_Block(nn.Module):
    def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
        super(RecMatrix_Block, self).__init__()

        conv_args = {'norm_type': norm_type, 'relu_type': relu_type}

        self.channel = channel
        self.shape = shape

        self.Conv4Space = nn.Sequential(
            ConvLayer(self.channel + self.shape**2, self.shape**2, **conv_args),
            ResidualBlock(self.shape**2, self.shape**2, **conv_args),

            nn.Sigmoid(),
            # ConvLayer(self.channel + self.shape**2, 256, **conv_args),
            # ResidualBlock(256, 256, **conv_args),
            # ConvLayer(256, 128, **conv_args),
            # ResidualBlock(128, 128, **conv_args),
            # ConvLayer(128, self.shape**2, **conv_args),
            # ResidualBlock(self.shape**2, self.shape**2, **conv_args),

            # nn.Sigmoid(),
        )
        self.Conv4Channel = nn.Sequential(
            nn.Linear(self.channel + self.shape**2, 32),
            ReluLayer(512, 'prelu'),
            nn.Linear(32, self.channel),

            nn.Sigmoid(),
            # nn.Linear(self.channel + self.shape**2, 32),
            # ReluLayer(512, 'prelu'),
            # nn.Linear(32, self.channel),

            # nn.Linear(self.channel, 32),
            # ReluLayer(512, 'prelu'),
            # nn.Linear(32, self.channel),

            # nn.Linear(self.channel, 32),
            # ReluLayer(512, 'prelu'),
            # nn.Linear(32, self.channel),

            # nn.Sigmoid(),
        )
        # self.ChannelFlipMerge = nn.Sequential(
        #     ConvLayer(self.channel*2,self.channel, **conv_args),
        #     ResidualBlock(self.channel, self.channel, **conv_args),

        #     nn.Sigmoid(),
        # )
        # self.Conv4Merge = nn.Sequential(
        #     ConvLayer(self.channel*2, self.channel, **conv_args),
        #     ResidualBlock(self.channel, self.channel, **conv_args),

        #     nn.Sigmoid(),
        # )
        self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)

    def forward(self, input):
        # ss_space, ss_channel = selfSimilarity(input)
        ss_space, _ = selfSimilarity(input)
        spaceF_cat = torch.cat((input, ss_space), 1) # N, (H*W+C), 7, 7
        # channelF_cat = torch.cat((input.view(input.size(0),input.size(1),-1), ss_channel),2) 

        M_space = self.Conv4Space(spaceF_cat) # N, (H*W), H, W
        M_space = M_space.view(M_space.size(0),M_space.size(1),-1) # N, (H*W), (H*W)
        # M_channel = self.Conv4Channel(channelF_cat) # N, C, C

        input_flatten = input.view(input.size(0),input.size(1),-1) # N, C, H*W
        feat_space = torch.matmul(input_flatten, M_space) # N, C, H*W
        # feat_channel = torch.matmul(M_channel, input_flatten) # N, C, H*W

        feat_space_map = feat_space.view(feat_space.size(0),feat_space.size(1),self.shape,-1) # N,C,H,W

        _, ss_channel = selfSimilarity(feat_space_map)
        channelF_cat = torch.cat((feat_space, ss_channel),2) # N,C,(H*W+C)
        M_channel = self.Conv4Channel(channelF_cat) # N, C, C
        feat_channel = torch.matmul(M_channel, feat_space) # N, C, H*W
        feat_channel = feat_channel.view(feat_channel.size(0),feat_channel.size(1),self.shape,-1)

        # ----------- Flip -------------
        # feat_channel_flip = torch.flip(feat_channel,[3])
        # feat_channel_cat = torch.cat((feat_channel_flip, feat_channel),1)
        # feat_channel = self.ChannelFlipMerge(feat_channel_cat)
        # feat_cat = torch.cat((feat_space,feat_channel),1) # N, 2*C, H, W
        # feat_new = self.Conv4Merge(feat_cat)

        # feat_new_v = self.pool5_7x7(feat_new) # N, C, 1
        # f_space = self.pool5_7x7(feat_space)
        f = self.pool5_7x7(feat_channel)
        # feat_cat = torch.cat((f_space,f_channel), 2)
        return f

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class RecNet(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        our FINAL model.
    """
    def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
        super(RecNet, self).__init__()
        self.channel = channel
        self.shape = shape

        conv_args = {'norm_type': norm_type, 'relu_type': relu_type}
        self.Conv4Space = nn.Sequential(
            ConvLayer(self.channel + self.shape**2, 256, **conv_args),
            ResidualBlock(256, 256, **conv_args),
            ConvLayer(256, 128, **conv_args),
            ResidualBlock(128, 128, **conv_args),
            ConvLayer(128, self.shape**2, **conv_args),
            ResidualBlock(self.shape**2, self.shape**2, **conv_args),

            nn.Sigmoid(),
        )
        self.Conv4Channel = nn.Sequential(
            nn.Linear(self.channel + self.shape**2, 32),
            ReluLayer(512, 'prelu'),
            nn.Linear(32, self.channel),

            nn.Linear(self.channel, 32),
            ReluLayer(512, 'prelu'),
            nn.Linear(32, self.channel),

            nn.Linear(self.channel, 32),
            ReluLayer(512, 'prelu'),
            nn.Linear(32, self.channel),

            nn.Sigmoid(),
        )
        self.ChannelFlipMerge = nn.Sequential(
            ConvLayer(self.channel*2,self.channel, **conv_args),
            ResidualBlock(self.channel, self.channel, **conv_args),
        )
        self.Conv4Merge = nn.Sequential(
                ConvLayer(self.channel*3, self.channel, **conv_args),
                ResidualBlock(self.channel, self.channel, **conv_args),
        )
        self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.classifier = AddMarginProduct(self.channel)

    def forward(self, input, label=None):
        ss_space, ss_channel = selfSimilarity(input)

        spaceF_cat = torch.cat((input, ss_space), 1) # N, (H*W+C), 7, 7
        channelF_cat = torch.cat((input.view(input.size(0),input.size(1),-1), ss_channel),2) 

        M_space = self.Conv4Space(spaceF_cat) # N, (H*W), H, W
        M_space = M_space.view(M_space.size(0),M_space.size(1),-1) # N, (H*W), (H*W)
        M_channel = self.Conv4Channel(channelF_cat) # N, C, C

        input_flatten = input.view(input.size(0),input.size(1),-1) # N, C, H*W
        feat_space = torch.matmul(input_flatten, M_space) # N, C, H*W
        feat_channel = torch.matmul(M_channel, input_flatten) # N, C, H*W

        feat_space = feat_space.view(feat_space.size(0),feat_space.size(1),self.shape,-1)
        feat_channel = feat_channel.view(feat_channel.size(0),feat_channel.size(1),self.shape,-1)

        # ----------- Flip -------------
        feat_channel_flip = torch.flip(feat_channel,[3])
        feat_channel_cat = torch.cat((feat_channel_flip, feat_channel),1)
        feat_channel = self.ChannelFlipMerge(feat_channel_cat)

        feat_cat = torch.cat((feat_space, feat_channel, input),1) # N, 3*C, H, W
        feat_new = self.Conv4Merge(feat_cat)

        feat_new_v = self.pool5_7x7(feat_new).view(feat_new.size(0), -1)
        # feat_space_v = self.pool5_7x7(feat_space).view(feat_space.size(0), -1)
        # feat_channel_v = self.pool5_7x7(feat_channel).view(feat_channel.size(0), -1)

        if label is None:
            return feat_new_v, feat_new
        else:
            pred_loss, pred_label = self.classifier(feat_new_v,label)
            return feat_new_v, pred_loss, pred_label, M_space, M_channel, feat_space, feat_channel

# class RecNet_chn(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         our FINAL model.
#     """
#     def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
#         super(RecNet_chn, self).__init__()
#         self.channel = channel
#         self.shape = shape

#         conv_args = {'norm_type': norm_type, 'relu_type': relu_type}
#         self.Conv4Channel = nn.Sequential(
#             nn.Linear(self.channel + self.shape**2, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Sigmoid(),
#         )
#         self.ChannelFlipMerge = nn.Sequential(
#             ConvLayer(self.channel*2,self.channel, **conv_args),
#             ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.Conv4Merge = nn.Sequential(
#                 ConvLayer(self.channel*2, self.channel, **conv_args),
#                 ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
#         self.classifier = AddMarginProduct(self.channel)

#     def forward(self, input, label=None):
#         _, ss_channel = selfSimilarity(input)

#         channelF_cat = torch.cat((input.view(input.size(0),input.size(1),-1), ss_channel),2) 
#         M_channel = self.Conv4Channel(channelF_cat) # N, C, C

#         input_flatten = input.view(input.size(0),input.size(1),-1) # N, C, H*W
#         feat_channel = torch.matmul(M_channel, input_flatten) # N, C, H*W

#         feat_channel = feat_channel.view(feat_channel.size(0),feat_channel.size(1),self.shape,-1)

#         # ----------- Flip -------------
#         feat_channel_flip = torch.flip(feat_channel,[3])
#         feat_channel_cat = torch.cat((feat_channel_flip, feat_channel),1)
#         feat_channel = self.ChannelFlipMerge(feat_channel_cat)
    
#         feat_cat = torch.cat((feat_channel, input),1) # N, 2*C, H, W
#         feat_new = self.Conv4Merge(feat_cat)

#         feat_new_v = self.pool5_7x7(feat_new).view(feat_new.size(0), -1)
#         # feat_space_v = self.pool5_7x7(feat_space).view(feat_space.size(0), -1)
#         # feat_channel_v = self.pool5_7x7(feat_channel).view(feat_channel.size(0), -1)

#         if label is None:
#             return feat_new_v, feat_new
#         else:
#             pred_loss, pred_label = self.classifier(feat_new_v,label)
#             return feat_new_v, pred_loss, pred_label

# class RecNet_spc(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         our FINAL model.
#     """
#     def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
#         super(RecNet_spc, self).__init__()
#         self.channel = channel
#         self.shape = shape

#         conv_args = {'norm_type': norm_type, 'relu_type': relu_type}
#         self.Conv4Space = nn.Sequential(
#             ConvLayer(self.channel + self.shape**2, 256, **conv_args),
#             ResidualBlock(256, 256, **conv_args),
#             ConvLayer(256, 128, **conv_args),
#             ResidualBlock(128, 128, **conv_args),
#             ConvLayer(128, self.shape**2, **conv_args),
#             ResidualBlock(self.shape**2, self.shape**2, **conv_args),

#             nn.Sigmoid(),
#         )

#         self.Conv4Merge = nn.Sequential(
#                 ConvLayer(self.channel*2, self.channel, **conv_args),
#                 ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
#         self.classifier = AddMarginProduct(self.channel)

#     def forward(self, input, label=None):
#         ss_space, _ = selfSimilarity(input)

#         spaceF_cat = torch.cat((input, ss_space), 1) # N, (H*W+C), 7, 7
        
#         M_space = self.Conv4Space(spaceF_cat) # N, (H*W), H, W
#         M_space = M_space.view(M_space.size(0),M_space.size(1),-1) # N, (H*W), (H*W)
        

#         input_flatten = input.view(input.size(0),input.size(1),-1) # N, C, H*W
#         feat_space = torch.matmul(input_flatten, M_space) # N, C, H*W
#         feat_space = feat_space.view(feat_space.size(0),feat_space.size(1),self.shape,-1)
    
#         feat_cat = torch.cat((feat_space,input),1) # N, 2*C, H, W
#         feat_new = self.Conv4Merge(feat_cat)

#         feat_new_v = self.pool5_7x7(feat_new).view(feat_new.size(0), -1)
#         # feat_space_v = self.pool5_7x7(feat_space).view(feat_space.size(0), -1)
#         # feat_channel_v = self.pool5_7x7(feat_channel).view(feat_channel.size(0), -1)

#         if label is None:
#             return feat_new_v, feat_new
#         else:
#             pred_loss, pred_label = self.classifier(feat_new_v,label)
#             return feat_new_v, pred_loss, pred_label

# class RecNet(nn.Module):
#     def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
#         super(RecNet, self).__init__()
#         self.channel = channel
#         self.shape = shape

#         self.rec_chn = RecNet_chn()
#         self.rec_spc = RecNet_spc()
#         weight_spc = load('/app/Occluded-Face-RecNet/pretrained_weight/spc_weight.pth.gzip')
#         weight_chn = load('/app/Occluded-Face-RecNet/pretrained_weight/chn_weight.pth.gzip')
#         self.rec_spc.load_state_dict(weight_spc['RecNet'], strict=False)
#         self.rec_chn.load_state_dict(weight_chn['RecNet'], strict=False)
#         for p in self.rec_spc.parameters():
#             p.requires_grad = False
#         for p in self.rec_chn.parameters():
#             p.requires_grad = False
    
#     def forward(self, input, label=None):
#         f_chn, _ = self.rec_chn(input)
#         f_spc, _ = self.rec_spc(input)
        
#         f_new = f_spc*0.8 + f_chn*0.2

#         if label is None:
#             return f_new
#         else:
#             pred_loss, pred_label = self.classifier(f_new,label)
#             return f_new, pred_loss, pred_label


# class RecNet(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         our FINAL model.
#     """
#     def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
#         super(RecNet, self).__init__()
#         self.channel = channel
#         self.shape = shape

#         conv_args = {'norm_type': norm_type, 'relu_type': relu_type}
#         self.Conv4Space = nn.Sequential(
#             ConvLayer(self.channel + self.shape**2, 256, **conv_args),
#             ResidualBlock(256, 256, **conv_args),
#             ConvLayer(256, 128, **conv_args),
#             ResidualBlock(128, 128, **conv_args),
#             ConvLayer(128, self.shape**2, **conv_args),
#             ResidualBlock(self.shape**2, self.shape**2, **conv_args),

#             nn.Sigmoid(),
#         )
#         self.Conv4Channel = nn.Sequential(
#             nn.Linear(self.channel + self.shape**2, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Sigmoid(),
#         )
#         self.ChannelFlipMerge = nn.Sequential(
#             ConvLayer(self.channel*2,self.channel, **conv_args),
#             ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.Conv4Merge = nn.Sequential(
#                 ConvLayer(self.channel*2, self.channel, **conv_args),
#                 ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
#         self.classifier = AddMarginProduct(self.channel)

#     def forward(self, input, label=None):
#         ss_space, ss_channel = selfSimilarity(input)

#         spaceF_cat = torch.cat((input, ss_space), 1) # N, (H*W+C), 7, 7
#         channelF_cat = torch.cat((input.view(input.size(0),input.size(1),-1), ss_channel),2) 

#         M_space = self.Conv4Space(spaceF_cat) # N, (H*W), H, W
#         M_space = M_space.view(M_space.size(0),M_space.size(1),-1) # N, (H*W), (H*W)
#         M_channel = self.Conv4Channel(channelF_cat) # N, C, C

#         input_flatten = input.view(input.size(0),input.size(1),-1) # N, C, H*W
#         feat_space = torch.matmul(input_flatten, M_space) # N, C, H*W
#         feat_channel = torch.matmul(M_channel, input_flatten) # N, C, H*W

#         feat_space = feat_space.view(feat_space.size(0),feat_space.size(1),self.shape,-1)
#         feat_channel = feat_channel.view(feat_channel.size(0),feat_channel.size(1),self.shape,-1)

#         # # ----------- Flip -------------
#         # feat_channel_flip = torch.flip(feat_channel,[3])
#         # feat_channel_cat = torch.cat((feat_channel_flip, feat_channel),1)
#         # feat_channel = self.ChannelFlipMerge(feat_channel_cat)

#         feat_cat = torch.cat((feat_space,feat_channel),1) # N, 2*C, H, W
#         feat_new = self.Conv4Merge(feat_cat)

#         feat_new_v = self.pool5_7x7(feat_new).view(feat_new.size(0), -1)
#         # feat_space_v = self.pool5_7x7(feat_space).view(feat_space.size(0), -1)
#         # feat_channel_v = self.pool5_7x7(feat_channel).view(feat_channel.size(0), -1)

#         if label is None:
#             return feat_new_v, M_space, M_channel
#         else:
#             pred_loss, pred_label = self.classifier(feat_new_v,label)
#             return feat_new_v, pred_loss, pred_label


# class RecNet(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#     """
#     def __init__(self, channel=512, shape=7, norm_type='bn', relu_type='prelu'):
#         super(RecNet, self).__init__()
#         self.channel = channel
#         self.shape = shape
#         self.N_head = 8
#         conv_args = {'norm_type': norm_type, 'relu_type': relu_type}
#         self.Conv4Space = nn.Sequential(
#             ConvLayer(self.channel + self.shape**2, 256, **conv_args),
#             ResidualBlock(256, 256, **conv_args),
#             ConvLayer(256, 128, **conv_args),
#             ResidualBlock(128, 128, **conv_args),
#             ConvLayer(128, self.shape**2, **conv_args),
#             ResidualBlock(self.shape**2, self.shape**2, **conv_args),

#             nn.Sigmoid(),
#         )
#         self.Conv4Channel = nn.Sequential(
#             nn.Linear(self.channel + self.shape**2, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Linear(self.channel, 32),
#             ReluLayer(512, 'prelu'),
#             nn.Linear(32, self.channel),

#             nn.Sigmoid(),
#         )
#         self.ChannelFlipMerge = nn.Sequential(
#             ConvLayer(self.channel*2,self.channel, **conv_args),
#             ResidualBlock(self.channel, self.channel, **conv_args),
#         )
#         self.Conv4Merge = nn.Sequential(
#                 ConvLayer(self.channel*2, self.channel, **conv_args),
#                 ResidualBlock(self.channel, self.channel, **conv_args),
#                 )
#         self.pool5_7x7 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
#         self.classifier = AddMarginProduct(self.channel)

#         # self.layer_stack = nn.ModuleList([
#         #     RecMatrix_Block(self.channel, self.shape, **conv_args)
#         #     for _ in range(self.N_head)])

#         # self.Linear = nn.Linear(self.N_head, 1)

#     def forward(self, input, label=None):
#         # feat_lst = [] 
#         # for rec_layer in self.layer_stack:
#         #     f = rec_layer(input)
#         #     feat_lst.append(f.view(f.size(0),f.size(1), -1))
        
#         # feat_cat = torch.cat(feat_lst, 2) # N, C, N_head
#         # feat_new = self.Linear(feat_cat).view(feat_cat.size(0), -1)
#         # if label is None:
#         #     return l2_norm(feat_new), feat_lst
#         # else:
#         #     pred_loss, pred_label = self.classifier(feat_new, label)
#         #     return l2_norm(feat_new), feat_lst, pred_loss, pred_label

#         # feat_new_v = self.pool5_7x7(feat_new).view(feat_new.size(0), -1)

#         # feat_space_v = self.pool5_7x7(feat_space).view(feat_space.size(0), -1)
#         # feat_channel_v = self.pool5_7x7(feat_channel).view(feat_channel.size(0), -1)
        
#         # --------- seblock -----------
#         b, c, _, _ = input.size()
#         y = self.avg_pool(input).view(b, c)
#         y = self.seblock(y).view(b, c, 1, 1)
#         output = input + input * y.expand_as(input)
#         feat_out_v = self.pool5_7x7(output).view(output.size(0),-1)
#         feat_new = output
#         feat_new_v = feat_out_v

#         # space feat
#         feat_new = feat_space
#         feat_new_v = feat_space_v

#         # channel feat
#         feat_new = feat_channel
#         feat_new_v = feat_channel_v
        
#         if label is None:
#             return feat_new_v, feat_space_v, feat_channel_v, feat_new, feat_space, feat_channel, M_space, M_channel
#         else:
#             pred_loss, pred_label = self.classifier(feat_new_v,label)
#             return feat_new_v, feat_space_v, feat_channel_v, feat_new, feat_space, feat_channel, M_space, M_channel, pred_loss, pred_label

if __name__ == '__main__':
    # model=RecNet()
    # input = torch.randn(32, 256, 7, 7)
    # output = model(input)
    # print(output.shape)
    x=torch.randn(32,512,7,7)
    model = RecNet()
    x, y = model(x)
    print(x)
    print(y)