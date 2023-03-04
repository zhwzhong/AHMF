# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   bestnet.py
@Time    :   2020/1/2 17:06
@Desc    :
"""
import math
import torch
import numpy as np
from torch import nn
from collections import Iterable
import switchable_norm as sn
from torch.nn.functional import softplus
from torch.nn.modules.utils import _pair
from torch.nn.functional import interpolate
from partialconv2d import PartialConv2d
import torch.nn.functional as F

def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num, trainable_num = clever_format([total_num, trainable_num])
    return {'Total': total_num, 'Trainable': trainable_num}


class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, y // ratio, x // ratio)
class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, partial=False, vcnn=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()
        if partial:
            self.layers = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif vcnn:
            self.layers = VConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'SN':
            self.norm = sn.SwitchNorm2d(out_channels, using_moving_average=True, using_bn=True)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Mish':
            self.act = Mish()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, *inputs):
        if len(inputs) == 1:
            out = self.layers(inputs[0])
        else:
            out = self.layers(inputs[0], inputs[1])

        if self.norm is not None:
            out = self.norm(out)

        if self.act is not None:
            out = self.act(out)
        return out

class FeatureInitialization(nn.Module):
    # 提取Depth和RGB的特征，变为64个通道
    def __init__(self, num_features, scale, guidance_channel=1):
        super(FeatureInitialization, self).__init__()

        self.rgb_shuffle = invPixelShuffle(ratio=scale)

        self.depth_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.guidance_in = nn.Sequential(
            nn.Conv2d(in_channels=guidance_channel, out_channels=num_features, kernel_size=3, padding=1),
            nn.PReLU(),
            InvUpSampler(scale=scale, n_feats=num_features)
        )

    def forward(self, depth, guidance):
        # guide_shuffle = self.rgb_shuffle(guidance)
        return self.depth_in(depth), self.guidance_in(guidance), None
    
class UpSampler(nn.Sequential):
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)

class InvUpSampler(nn.Sequential):
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):

                m.append(invPixelShuffle(2))
                m.append(nn.Conv2d(in_channels=n_feats * 4, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PReLU())
        super(InvUpSampler, self).__init__(*m)

class Compress(nn.Module):
    def __init__(self, num_features, act, norm, fuse_way='add'):
        super(Compress, self).__init__()
        self.fuse_way = fuse_way

        self.layers = ResNet(num_features=num_features, act=act, norm=norm)

        if self.fuse_way == 'cat':
            self.compress_out = ConvBNReLU2D(in_channels=2 * num_features, out_channels=num_features, kernel_size=1,
                                             padding=0, act=act)

    def forward(self, *inputs):
        if len(inputs) == 2:
            if self.fuse_way == 'add':
                out = inputs[0] + inputs[1]
            else:
                out = self.compress_out(torch.cat(([inputs[0], inputs[1]]), dim=1))
        else:
            out = inputs[0]
        return self.layers(out)

class ResNet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(*[
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, norm=norm)
        ])
        self.act = get_act(act=act)

    def forward(self, input_feature):
        return self.act(self.layers(input_feature) + input_feature)

def variance_pool(x):
    my_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    return (x - my_mean).pow(2).mean(dim=3, keepdim=False).mean(dim=2, keepdim=False).view(x.size()[0], x.size()[1], 1, 1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

def pool_func(x, pool_type=None):
    b, c = x.size()[:2]
    if pool_type == 'avg':
        ret = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'max':
        ret = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'lp':
        ret = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    else:
        ret = variance_pool(x)
    return ret.view(b, c)

class GateConv2D(nn.Module):
    def __init__(self, num_features):
        super(GateConv2D, self).__init__()
        self.Attention = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.Feature = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self, inputs):
        return self.Attention(inputs) * self.Feature(inputs)

class ConvGRUCell(nn.Module):
    """
    Basic CGRU cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super(ConvGRUCell, self).__init__()

        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.update_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size, padding=self.padding,
                                     bias=self.bias)
        self.reset_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    bias=self.bias)

        self.out_gate = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur = cur_state
        # data size is [batch, channel, height, width]
        x_in = torch.cat([input_tensor, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        x_out = torch.tanh(self.out_gate(torch.cat([input_tensor, h_cur * reset], dim=1)))
        h_new = h_cur * (1 - update) + x_out * update

        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w).cuda()


class ConvGRU(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers=2,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvGRUCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class MMAB(nn.Module):
    def __init__(self, num_features, reduction_ratio=4):
        super(MMAB, self).__init__()

        self.squeeze = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features * 2 // reduction_ratio,
                                    kernel_size=3, act='PReLU', padding=1)

        self.excitation1 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')
        self.excitation2 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')

    def forward(self, depth, guidance):
        fuse_feature = self.squeeze(torch.cat((depth, guidance), 1))
        fuse_statistic = pool_func(fuse_feature, 'avg') + pool_func(fuse_feature)
        squeeze_feature = fuse_statistic.unsqueeze(2).unsqueeze(3)
        depth_out = self.excitation1(squeeze_feature)
        guidance_out = self.excitation2(squeeze_feature)
        return (depth_out * depth).div(2), (guidance_out * guidance).div(2)

class FuseNet(nn.Module):
    def __init__(self, num_features, reduction_ratio, act, norm):
        super(FuseNet, self).__init__()

        self.filter_conv = GateConv2D(num_features=num_features)
        self.filter_conv1 = GateConv2D(num_features=num_features)
        self.attention_layer = MMAB(num_features=num_features, reduction_ratio=reduction_ratio)
        self.res_conv = ResNet(num_features=num_features, act=act, norm=norm)


    def forward(self, depth, guide):
        guide = self.filter_conv(guide)
        depth = self.filter_conv1(depth)
        depth, guide = self.attention_layer(depth=depth, guidance=guide)

        fuse_feature = self.res_conv(depth + guide)

        return fuse_feature

def get_act(act):
    if act == 'PReLU':
        ret_act = torch.nn.PReLU()
    elif act == 'SELU':
        ret_act = torch.nn.SELU(True)
    elif act == 'LeakyReLU':
        ret_act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
    elif act == 'ELU':
        ret_act = torch.nn.ELU(inplace=True)
    elif act == 'ReLU':
        ret_act = torch.nn.ReLU(True)
    elif act == 'Mish':
        ret_act = Mish()
    else:
        print('ACT ERROR')
        ret_act = torch.nn.ReLU(True)
    return ret_act
    
class AHMF(nn.Module):
    def __init__(self, scale=4):
        super(AHMF, self).__init__()
        
        self.head = FeatureInitialization(num_features=64, scale=scale, guidance_channel=3)

        # Forward Backward None ALL
        self.rgb_conv = nn.ModuleList()
        self.fuse_conv = nn.ModuleList()
        self.depth_conv = nn.ModuleList()
        self.compress_out = nn.ModuleList()

        self.forward_gru_cell = nn.ModuleList()
        self.reverse_gru_cell = nn.ModuleList()

        for _ in range(3):
            self.rgb_conv.append(
                ConvBNReLU2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, act='PReLU')
            )

        for _ in range(3):
            self.depth_conv.append(
                ConvBNReLU2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, act='PReLU')
            )


        for _ in range(4):
            self.fuse_conv.append(
                FuseNet(num_features=64, reduction_ratio=4, act='PReLU', norm=None)
            )

            self.compress_out.append(
                Compress(num_features=64, act='PReLU', norm=None)
            )


        self.forward_gru_cell = ConvGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)

        self.reverse_gru_cell = ConvGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)

        self.up_conv = nn.Sequential(
            ConvBNReLU2D(in_channels=64 * 4, out_channels=64,
                         kernel_size=1, padding=0, act='PReLU'),
            *UpSampler(scale=scale, n_feats=64),
            ConvBNReLU2D(in_channels=64, out_channels=1, kernel_size=3, padding=1, vcnn=False, norm=None)
        )

    def forward(self, lr, rgb, lr_up):
        depth_feature, guide_feature, _ = self.head(lr, rgb)

        depth_out = [depth_feature]
        guide_out = [guide_feature]

        for i in range(3):
            guide_feature = self.rgb_conv[i](guide_feature)
            guide_out.append(guide_feature)

        for i in range(3):
            depth_feature = self.depth_conv[i](depth_feature)
            depth_out.append(depth_feature)

        fuse_feature = []
        for i in range(4):
            tmp = self.fuse_conv[i](depth=depth_out[3 - i],
                                    guide=guide_out[3 - i])
            fuse_feature.append(tmp)
        
        forward_hidden_list, _ = self.forward_gru_cell(torch.stack(fuse_feature, dim=1))
        forward_hidden_list = forward_hidden_list[-1]


        reversed_idx = list(reversed(range(4)))

        reverse_hidden_list, _ = self.reverse_gru_cell(torch.stack(fuse_feature, dim=1)[:, reversed_idx, ...])
        reverse_hidden_list = reverse_hidden_list[-1]
        reverse_hidden_list = reverse_hidden_list[:, reversed_idx, ...]

        fuse_out = []

        for i in range(4):
            tmp_out = self.compress_out[i](forward_hidden_list[:, i], reverse_hidden_list[:, i])
            fuse_out.append(tmp_out)

        out = self.up_conv(torch.cat(tuple(fuse_out), dim=1))
        return [out + lr_up]

    
