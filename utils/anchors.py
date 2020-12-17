import itertools
import torch
import torch.nn as nn
import numpy as np


class Anchors(nn.Module):
    """"""

    def __init__(self, anchor_scale=4., pyramid_levels=None):
        super(Anchors, self).__init__()
        self.anchor_scale = anchor_scale
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        # [8, 16, 32, 64, 128]---->在p3, p4, p5, p6, p7上输出的特征图相对于原图的缩放倍数
        self.strides = [2 ** x for x in self.pyramid_levels]

        #  array([1.        , 1.25992105, 1.58740105])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        # 正方形、胖矩形， 瘦矩形
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def forward(self, image):
        """输入图片张量"""
        image_shape = image.shape[2:]
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            # itertools.product(self.scales, self.ratios)返回一个迭代器，迭代内容使用之灵栏next查看
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('输出尺寸必须能被步长整除！')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)

                xv, yv = np.meshgrid(x, y)
                xv, yv = xv.reshape(-1), yv.reshape(-1)

                boxes = np.vstack(
                    (yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_y_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        return anchor_boxes
