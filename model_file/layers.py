import collections
import math
import re
from functools import partial

import torch.nn as nn
import torch
from torch.nn import functional as F

GlobalParams = collections.namedtuple('GlobalParams',
                                      ['batch_norm_momentum', 'batch_norm_epsilon', "image_size", "width_coefficient",
                                       "depth_divisor", "min_depth", "depth_coefficient", "num_classes",
                                       "dropout_rate", "drop_connect_rate"])
BlockArgs = collections.namedtuple('BlockArgs', ["se_ratio", 'id_skip', 'input_filters', "expand_ratio", "kernel_size",
                                                 "stride", "output_filters", "num_repeat"])


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


def round_repeats(repeats, global_params):
    """
    通过调整子模块的重复次数而达到调整网络整体深度的目的

    :param repeats: b0初始子模块深度
    :param global_params: 全局参数信息
    :return: 调整后的模块重复次数
    """
    mutiplire = global_params.depth_coefficient
    if not mutiplire:
        return repeats
    return int(math.ceil(mutiplire * repeats))


def get_same_padding_conv2d(image_size=None):
    """
    在保存ONNX文件时需要传入image_size参数，此处作为选择，在保存ONNX时自动调整
    :param image_size:
    :return:
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)  # 此处partial作用位置


# return partial(Conv2dS, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """动态padding, 此类为nn.Conv2d的子类"""

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

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [left, right, top, bottom])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Module):
    """静态padding, 此类为nn.Module的子类"""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, bias=True, groups=1):
        super(Conv2dStaticSamePadding, self).__init__()
        # 实例化卷积层
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, bias=bias, groups=groups)

        # 获取卷积层的步长、卷积核尺寸和空洞个数
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        # 确保获取的步长和卷积核尺寸是一个长度为2的列表
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        input_h, input_w = x.shape[-2:]

        # (i / s - 1) * s -i + k
        #
        extra_h = (math.ceil(input_w / self.stride[1]) - 1) * self.stride[1] - input_w + self.kernel_size[1]
        extra_w = (math.ceil(input_h / self.stride[0]) - 1) * self.stride[0] - input_h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_w // 2
        bottom = extra_w - top
        # pad内部细节已经封装
        x = F.pad(x, [left, right, top, bottom])
        return self.conv(x)


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
                len(options['s']) == 2 and options['s'][0] == options['s'][1]))  # 核对解码后获取的步长是否正确
        return BlockArgs(se_ratio=float(options['se']) if "se" in options else None,
                         id_skip=('noskip' not in block_string),  # 若字符串种不包含'noskip'则返回True表示接入残差结构
                         input_filters=int(options['i']),
                         expand_ratio=int(options['e']),
                         kernel_size=int(options['k']),
                         stride=[int(options['s'][0])],
                         output_filters=int(options['o']),
                         num_repeat=int(options['r']))


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2,
                 image_size=None, num_classes=1000):
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
    # 包含网络模块信息的字符串列表，可通过BlockDecoder中的内置方法生成
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    # 全局网络信息
    global_params = GlobalParams(batch_norm_momentum=0.99,
                                 batch_norm_epsilon=1e-3,
                                 image_size=image_size,
                                 width_coefficient=width_coefficient,
                                 depth_divisor=8,
                                 min_depth=None,
                                 depth_coefficient=depth_coefficient,
                                 num_classes=num_classes,
                                 dropout_rate=dropout_rate,
                                 drop_connect_rate=drop_connect_rate)
    return blocks_args, global_params


def get_model_params(model_name, other_params):
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, image_size, dropout_rate = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient, depth_coefficient, dropout_rate, image_size)
    else:
        raise NotImplementedError(f"模型名称错误，输入名称为{model_name}")
    if other_params:
        global_params = global_params._replace(**other_params)
    return blocks_args, global_params


# 网络层中的激活函数实现
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.save_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


# 网络层中的激活函数实现
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def drop_connect(inputs, p, training):
    """
    手动实现drop_out， 通过在最后一层将矩阵部分置为0实现在该模块中的一个dropout
    :param inputs: 输入的张良
    :param p: 权重失活比例
    :param training: 是否是在训练中
    :return: 修改后的矩阵，已经失活的位置在该矩阵中等于0
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    # torch.rand()从[0, 1]抽取随机数
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    # torch.floor()对tensor向下取整
    binary_tensor = torch.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


class MaxPool2dStaticSamePadding(nn.Module):
    """自适应pad补图最大池化"""

    def __init__(self, *args, **kwargs):
        super(MaxPool2dStaticSamePadding, self).__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        # 获取池化实例的步长和池化卷积尺寸
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.stride) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        input_w, input_h = x.shape[-2:]

        extra_h = (math.ceil(input_w / self.stride[1]) - 1) * self.stride[1] - input_w + self.kernel_size[1]
        extra_w = (math.ceil(input_h / self.stride[0]) - 1) * self.stride[0] - input_h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_w // 2
        bottom = extra_w - top
        # pad内部细节已经封装
        x = F.pad(x, [left, right, top, bottom])
        return self.pool(x)
