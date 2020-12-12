import collections
import math
import re
from functools import partial

import torch.nn as nn
import torch
from torch.nn import functional as F

GlobalParams = collections.namedtuple('GlobalParams',
                                      ['batch_norm_momentum', 'batch_norm_epsilon', "image_size", "width_coefficient",
                                       "depth_divisor", "min_depth"])
BlockArgs = collections.namedtuple('BlockArgs', ["se_ratio", 'id_skip', ])


def round_filters(filters, global_params):
    """
    每层宽度的自动调整，即调整自动调整卷积核的数目
    :param filters: 初始卷积核的数目
    :param global_params: 全局参数信息
    :return: 调整后的卷积核数目
    """
    multiplire = global_params.width_coefficient
    if not multiplire:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplire
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dDynamicSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, input_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv2dDynamicSamePadding, self).__init__(input_channels, out_channels, kernel_size, stride, 0, dilation,
                                                       groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        # math.ceil返回一个大于或等于传入参数的的最小整数
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)

        # o = ((i - k + 2p) / s) + 1 =====> p
        # kernel_size = (kernel_h - 1) * self.dilation[0] + 1
        # 2p = (o - 1) * s + k - i
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def efficientnet_params(model_name):
    """
     获取网络的形状变化信息

     :param model_name: 网络型号选择，从efficientnet-b{0~7}
     :return: 长度为4的网络参数信息，从左至右为网络的宽度变化比例，网络深度变化比例，输入图片的长宽信息，Dropout的参数失活比例
     """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """
    网络信息和参数编码字符串&字符串网络和参数编码
    """

    @staticmethod
    def encode(blocks_arge):
        """
        网络信息编码单个字符串
        :param blocks_arge: 包含网络信息的BlockArgs
        :return: 完成网络信息编码后的字符串
        """
        block_strings = []
        for block in blocks_arge:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

    @staticmethod
    def _encode_block_string(block):
        """
        将网络子模块信息编码及网络层的其他信息组合成字符串
        :param block:
        :return: 编码过后的字符串
        """
        args = []
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        逐个解码字符串列表中的字符串
        :param string_list: 完成字符串编码的字符串列表
        :return: 包含多个字符串解码信息后的BlockArgs的列表
        """
        assert isinstance(string_list, list)
        block_args = []
        for block_string in string_list:
            block_args.append(BlockDecoder._decode_block_string(block_string))
        return block_args

    @staticmethod
    def _decode_block_string(block_string):
        """
        解码字符串
        :param block_string: 传入编码过后的字符串
        :return: 包含解码信息后的BlockArgs
        """
        assert isinstance(block_string, str)  # 确保输入数据类型正确
        ops = block_string.split('_')  # 特定字符串分解
        options = {}
        for op in ops:  # 将解码后的字符串再次解码获取信息后传入字典
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                options[splits[0]] = splits[1]
        assert (('s' in options and len(options['s']) == 1) or (
                len(options['s']) == 1 and options['s'][0] == options['s'][1]))  # 核对解码后获取的步长是否正确
        return BlockArgs(se_ratio=float(options['se']) if "se" in options else None,
                         id_skip=('noskip' not in block_string),  # 若字符串种不包含'noskip'则返回True表示接入残差结构
                         )


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2,
                 image_size=None, numclasses=1000):
    """
    返回需要生成的efficiennet的子模块参数和全局参数
    :param width_coefficient:
    :param depth_coefficient:
    :param dropout_rate:
    :param drop_connect_rate:
    :param image_size:
    :param numclasses:
    :return:
    """
    blocks_args = []  # 包含网络模块信息的字符串列表，可通过BlockDecoder中的内置方法生成
    blocks_args = BlockDecoder.decode(blocks_args)

    # 全局网络信息
    global_params = GlobalParams(batch_norm_momentum=0.99,
                                 batch_norm_epsilon=1e-3,
                                 image_size=image_size,
                                 width_coefficient=width_coefficient,
                                 depth_divisor=8,
                                 min_depth=None,
                                 )
    return blocks_args, global_params


def get_model_params(model_name, other_params):
    if model_name.startwith('efficientnet'):
        width_coefficient, depth_coefficient, dropout_rate, image_size = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient, depth_coefficient, dropout_rate, image_size)
    else:
        raise NotImplementedError(f"模型名称错误，输入名称为{model_name}")
    if other_params:
        global_params = global_params._replace(**other_params)
    return blocks_args, global_params


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backwatd(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.save_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
