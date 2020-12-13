import torch
import torch.nn as nn
from torch.nn import functional as F
from model_file.layers import (round_filters, get_same_padding_conv2d, MemoryEfficientSwish, drop_connect, Swish,
                               round_repeats, get_model_params, efficientnet_params)


class Main_Conv_Block(nn.Module):
    """图解见model_file/Main_Conv_Block"""

    def __init__(self, bolck_args, global_params):
        super(Main_Conv_Block, self).__init__()
        self._block_args = bolck_args
        # 获得网络标准化参数
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon

        # 注意力机制的缩放比例
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        # 是否接入残差结构
        self.id_skip = bolck_args.id_skip

        # 根据是否需要保存ONNX选择调用的卷积类
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # 1x1卷积通道扩张
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        # 深度可分离卷积
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwish = Conv2d(oup, oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        # (通道)注意力机制模块组， 线进行通道数的收缩后再扩张
        if self.has_se:  # 确定注意力机制的缩放比例是否有效
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # 先缩减通道后将通道数还原
            self._se_reduce = Conv2d(oup, num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(num_squeezed_channels, oup, kernel_size=1)

        # 子模块输出部分
        final_out = self._block_args.output_filters
        self._project_conv = Conv2d(oup, final_out, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_out, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # 对输入进行通道扩张
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        # 深度可分离卷积模块
        x = self._swish(self._bn1(self._depthwish(x)))

        # 通道注意力机制
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, output_size=1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))

        # 是否添加残差
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        # 如果添加了残差也就添加了dropout
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet网络实现"""

    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__()
        # 使用断言避免身材错误的网络信息
        assert isinstance(blocks_args, list), f'blocks_args应该是一个列表,传入的数据类型为{type(blocks_args)}'
        assert len(blocks_args) > 0, f"blocks_args的长度必须大于0,传入的数据长度为{len(blocks_args)}"
        self._global_params = global_params
        self._blocks_args = blocks_args

        # 获取卷积实例
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # 获得标准化参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # 网络开始, 定义初始输入数据通道和获取hand运算后的通道数目
        input_channels = 3
        output_channels = round_filters(32, self._global_params)
        # 初次卷积与标准化,padding会根据传入的kernel_size和stride自动补边或者下采样
        self._conv_stem = Conv2d(input_channels, output_channels, kernel_size=3, stride=2, bias=False)

        self._bn0 = nn.BatchNorm2d(output_channels, momentum=bn_mom, eps=bn_eps)

        # 按照解码后的字符串信息逐块生成子模块
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)):
            # 根据全局参数对每个子模块中的每层网络逐一修改
            # _replace:将第一个参数的值替换为第二个参数的值
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters=round_filters(self._blocks_args[i].input_filters, self._global_params),
                output_filters=round_filters(self._blocks_args[i].output_filters, self._global_params),
                num_repeat=round_repeats(self._blocks_args[i].num_repeat, self._global_params))

            # 第一次Block中卷积需要考虑步长和输入的通道数
            self._blocks.append(Main_Conv_Block(self._blocks_args[i], self._global_params))

            # 第一层网络生成后将步长和输入特征图的层数进行更改
            if self._blocks_args[i].num_repeat > 1:
                # 替换掉stride使得在除了进行采样时都可以进行残差和大模块之间特征图尺寸不变
                self._blocks_args[i] = self._blocks_args[i]._replace(input_filters=self._blocks_args[i].output_filters,
                                                                     stride=1)

            # 使用生成器剩余的参数继续生成网络
            for _ in range(self._blocks_args[i].num_repeat - 1):
                self._blocks.append(Main_Conv_Block(self._blocks_args[i], self._global_params))

        # 模型在分类前的特征收尾，将最后一层block的输出替换成输入，输出层采用b0或者基于b0变化后的outputfilters
        input_channels = self._blocks_args[len(self._blocks_args) - 1].output_filters
        output_channels = round_filters(1280, self._global_params)

        # 卷积+标准化
        self._conv_head = Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(output_channels, momentum=bn_mom, eps=bn_eps)

        # 线性全连接
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        self._fc = nn.Linear(output_channels, self._global_params.num_classes)

        # swish激活
        self._swish = MemoryEfficientSwish()

    def extract_features(self, inputs):
        """返回最后一层卷积层的输出结果"""
        # 输入层
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                # 按照失活比例获得失活神经元的总数目
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            print(x.shape)
        # out
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # 网络主函数运算
        x = self.extract_features(inputs)

        # 最后的池化和输出
        x = self._fc(self._dropout(self._avg_pooling(x).reshape(batch_size, -1)))
        return x

    @classmethod
    def from_name(cls, model_name, other_params=None):
        """
        返回一个实例化的著网络的class
        :param model_name: 网络型号选择，从efficientnet-b{0~7}
        :param other_params: 网络的其他参数，如分类层分类数目
        :return: 一个实例化后的efficientnet-b{0~7}
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, other_params=other_params)
        return cls(blocks_args, global_params)

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('模型名称必须是这些当中的一个:' + ", ".join(valid_models))

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def get_model(cls, model_name, num_classes):
        model = cls.from_name(model_name, other_params={'num_classes': num_classes})
        return model


if __name__ == '__main__':
    from torchsummary import summary
    model_name = r"efficientnet-b0"
    image_size = EfficientNet.get_image_size(model_name)
    model = EfficientNet.get_model(model_name, 1000).cuda()
    summary(model, (3, image_size, image_size), device='cuda')
