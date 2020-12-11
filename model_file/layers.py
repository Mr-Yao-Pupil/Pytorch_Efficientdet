import collections
import re
import torch.nn as nn

import torch

GlobalParams = collections.namedtuple('GlobalParams', [])
BlockArgs = collections.namedtuple('BlockArgs', [])


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
        return BlockArgs()


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
    global_params = GlobalParams()  # 全局网络信息
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

