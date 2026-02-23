import jittor as jt
from jittor import init
from jittor import nn

import collections.abc
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def rescale_2x(x: jt.Var, scale_factor=2):
    return nn.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def resize_to(x: jt.Var, tgt_hw: tuple):
    return nn.interpolate(x, size=tgt_hw, mode='bilinear', align_corners=False)

def global_avgpool(x: jt.Var):
    return x.mean(dims=(-1, -2), keepdims=True)

def _get_act_fn(act_name, inplace=True):
    if (act_name == 'relu'):
        return nn.ReLU()
    elif (act_name == 'leaklyrelu'):
        return nn.LeakyReLU(scale=0.1)
    elif (act_name == 'gelu'):
        return nn.GELU()
    elif (act_name == 'sigmoid'):
        return nn.Sigmoid()
    else:
        raise NotImplementedError

class ConvBN(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, g=1, bias=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv(in_dim, out_dim, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
        self.bn = nn.BatchNorm(out_dim)

    def execute(self, x):
        return self.bn(self.conv(x))

class CBR(nn.Module):

    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, bias=True):
        super().__init__()
        self.conv = nn.Conv(in_dim, out_dim, k, stride=s, padding=p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm(out_dim)
        self.relu = nn.ReLU()

    def execute(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBNReLU(nn.Sequential):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if is_transposed:
            conv_module = nn.ConvTranspose
        else:
            conv_module = nn.Conv
        self.add_module('conv', conv_module(
            in_planes, out_planes, 
            kernel_size=kernel_size, 
            stride=to_2tuple(stride), 
            padding=to_2tuple(padding), 
            dilation=to_2tuple(dilation), 
            groups=groups, 
            bias=bias
        ))
        self.add_module('bn', nn.BatchNorm(out_planes))
        if (act_name is not None):
            self.add_module(act_name, _get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=8,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        """
        执行流程Conv2d => GroupNormalization [=> Activation]

        Args:
            in_planes: 模块输入通道数
            out_planes: 模块输出通道数
            kernel_size: 内部卷积操作的卷积核大小
            stride: 卷积步长
            padding: 卷积padding
            dilation: 卷积的扩张率
            groups: 卷积分组数，需满足pytorch自身要求
            gn_groups: GroupNormalization的分组数，默认为4
            bias: 是否启用卷积的偏置，默认为False
            act_name: 使用的激活函数，默认为relu，设置为None的时候则不使用激活函数
            inplace: 设置激活函数的inplace参数
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module('conv', nn.Conv(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias
        ))
        
        self.add_module('gn', nn.GroupNorm(gn_groups, out_planes))
        if (act_name is not None):
            self.add_module(act_name, _get_act_fn(act_name=act_name, inplace=inplace))

            
class PixelNormalizer(nn.Module):

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        'Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.\n\n        Args:\n            mean (tuple, optional): the mean value. Defaults to (0.485, 0.456, 0.406).\n            std (tuple, optional): the std value. Defaults to (0.229, 0.224, 0.225).\n        '
        super().__init__()

        self.mean = jt.array(mean).reshape(3, 1, 1).stop_grad()
        self.std = jt.array(std).reshape(3, 1, 1).stop_grad()

    def __repr__(self):
        return (self.__class__.__name__ + f'(mean={self.mean.flatten()}, std={self.std.flatten()})')

    def execute(self, x):
        'normalize x by the mean and std values\n\n        Args:\n            x (torch.Tensor): input tensor\n\n        Returns:\n            torch.Tensor: output tensor\n\n        Albumentations:\n\n        ```\n            mean = np.array(mean, dtype=np.float32)\n            mean *= max_pixel_value\n            std = np.array(std, dtype=np.float32)\n            std *= max_pixel_value\n            denominator = np.reciprocal(std, dtype=np.float32)\n\n            img = img.astype(np.float32)\n            img -= mean\n            img *= denominator\n        ```\n        '
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class LayerNorm2d(nn.Module):
    '\n    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py\n    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119\n    '

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = jt.ones(num_channels)
        self.bias = jt.zeros(num_channels)
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(1, keepdims=True)
        s = (x - u).pow(2).mean(1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
