import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




class E_SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv2 = Conv(c_ * 3, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.a(y0)
        y2 = self.m(y1)
        # y3 = self.m(y2)

        # all = torch.cat(y0,y1,y2,y3, 1)
        all = torch.cat((y0, y1, y2), dim=1)
        end = self.cv2(all)

        return end
    

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from models.common import C3

#from ultralytics.nn.modules import (C2f)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.drop_path(self.mlp(x))
        return shortcut + x


    # def forward(self, x):
    #     if self.adjust_channel is not None:
    #         x = self.adjust_channel(x)
    #     shortcut = x
    #     x = self.spatial_mixing(x)
    #     x = shortcut + self.drop_path(self.mlp(x))
    #     return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x




# class C3_Faster(C3):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))
    
class C3_Faster(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 左支
        self.cv2 = Conv(c1, c_, 1, 1)  # 右支
        self.cv3 = Conv(2 * c_, c2, 1, 1)  # 输出融合
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.m(self.cv1(x))   # 左支：经过 Faster_Block
        y2 = self.cv2(x)           # 右支：直接卷积
        y = torch.cat((y1, y2), 1) # 拼接：通道翻倍
        return self.cv3(y)         # 输出：还原到 c2


"""
class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))
"""

import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity
from thop import profile


class CSFblock(nn.Module):
    def __init__(self, high_channels, low_channels,outChannels):
        super().__init__()

        self.UpChannels = nn.Sequential(
            nn.Conv2d(low_channels, outChannels, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannels),)
        self.DownChannels = nn.Sequential(
            nn.Conv2d(high_channels, outChannels, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannels),)

        self.AdaptiveAvgPool2d_1 = nn.AdaptiveAvgPool2d(1)
        self.AdaptiveAvgPool2d_2 = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs): 

        if len(inputs) != 2:
            raise ValueError("CSFblock requires exactly 2 input tensors.")
        High, Low = inputs[0], inputs[1] 
        #x1 = torch.Size([1, 256, 40, 40])
        #x2 = torch.Size([1, 128, 40, 40])

        N_High = self.DownChannels(High)
        N_Low = self.UpChannels(Low)

        key_H = self.AdaptiveAvgPool2d_1(N_High)
        key_L = self.AdaptiveAvgPool2d_2(N_Low)


        key_twice = torch.cat([key_H, key_L], 2)

        softKey = self.softmax(key_twice)

        doorH = torch.unsqueeze(softKey[:, :, 0], 2)
        doorL = torch.unsqueeze(softKey[:, :, 1], 2)

        good_high = doorH * N_High
        good_low = doorL * N_Low

        well_dool = good_high+ good_low

        return well_dool
    
# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
 
 
# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)




