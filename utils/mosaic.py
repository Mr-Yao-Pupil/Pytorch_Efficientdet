from PIL import Image, ImageDraw
import numpy as np
import math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

"""离线mosaic数据增强"""
"""
txt的数据格式
/home/gp/dukto/Xray_match/data/train/JPEGImages/300059.jpg 91,263,219,324,2
/home/gp/dukto/Xray_match/data/train/JPEGImages/200408.jpg 274,99,358,139,1
/home/gp/dukto/Xray_match/data/train/JPEGImages/400329.jpg 370,356,467,439,2 588,107,720,160,0
/home/gp/dukto/Xray_match/data/train/JPEGImages/400566.jpg 115,52,201,199,0
/home/gp/dukto/Xray_match/data/train/JPEGImages/400327.jpg 151,184,236,292,2 159,210,418,298,0 313,110,382,216,4
"""


def rand(a=0, b=1):
    """生成随机数"""
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    """
    新target框的坐标点计算
    :param bboxes: 组合之后的bbox二维张量
    :param cutx: 新图在w上的裁剪位置
    :param cuty: 信徒在h上的裁剪位置
    :return: 重新整理后的实际框信息
    """
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # 第一场图片,即左上角的图片
            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cuty:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            # 第二张图片，即左下角图片
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cuty
                    if x2 - x1 < 5:
                        continue

            # 第三张图片，即右下角的图片
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            # 第四张图片，即右上角的图片
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_image=True,
                    flip_probability=0.5):
    """

    :param annotation_line:
    :param input_shape: 数据增强后的图片尺寸, 数据类型为tuple
    :param random:
    :param hue:
    :param sat:
    :param val:
    :param proc_image:
    :param flip_probability: 左右翻转图片的概率,输入为0~1之间, 数据类型为float
    :return:
    """
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4

    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]  # 四张图片x1的起始位置
    place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]  # 四张图片y1的起始位置

    for line in annotation_line:
        line_content = line.split()  # 对一行数据进行切割，获得一张图片的信息
        image = Image.open(line_content[0]).convert('RGB')  # 通过图片路径打开图片,且保证是三通道图片

        image_w, image_h = image.size  # 获取图片的宽和高
        # 将txt中的所有数据转换成array向量, 形状为[box_num, 5]
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        flip = rand() < flip_probability  # 按照概率生成是否反转图片
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 图片左右反转
            box[:, [0, 2]] = image_w - box[:, [2, 0]]  # 实际框的左右反转,反转后左边变右边,故x1和x2的就变成到图片右边界的距离

        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:  # h < w
            new_h = int(scale * h)  # 对较小边进行缩放
            new_w = int(new_ar * new_h)  # 对较大边进行等比缩放
        else:  # w < h
            new_w = int(scale * w)
            new_h = int(new_ar * new_w)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
        # h:色调, s:饱和度, v:明度
        x = rgb_to_hsv(np.array(image) / 255.)  # 数据归一化,这样生成的hsv也是归一化的矩阵
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1  # 色度是一个角度，本取值0~360,经过归一化之后大于一的相当于多转了一圈
        x[..., 0][x[..., 0] < 0] += 1  # 色度是一个角度，本取值0~360,经过归一化之后小于零的相当于少转了一圈
        x[..., 1] *= sat  # 饱和度的变化
        x[..., 2] *= val  # 明度的变化
        x[x > 1] = 1  # 强制大于1的为1,反算RGB时不会出错
        x[x < 0] = 0  # 强制小于0的为0,反算RGB时不会出错
        image = Image.fromarray((hsv_to_rgb(x) * 255).astype(np.int8))

        # 放置图片
        dx = place_x[index]
        dy = place_y[index]

        new_image = Image.new("RGB", size=(w, h), color=(128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255

        index += 1
        # 对BOX重新调整
        box_data = []
        if len(box) > 0:
            # 按0轴打乱顺序：This function only shuffles the array along the first axis
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * new_w / image_w + dx  # new_w / image_w:相较于原图x坐标的变化关系
            box[:, [1, 3]] = box[:, [1, 3]] * new_h / image_h + dy  # 同上
            box[:, 0:2][box[:, 0:2] < 0] = 0  # 实际框x1, y1超出范围的强制回归
            box[:, 2][box[:, 2] > w] = w  # 实际框x2超出范围的强制回归
            box[:, 3][box[:, 3] > h] = h  # 实际框y2超出范围的强制回归
            box_w = box[:, 2] - box[:, 0]  # 实际框的w
            box_h = box[:, 3] - box[:, 1]  # 实际框的h
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 去除实际框h,w不足1的框

            # 新图片的所有框进行封装
            box_data = np.zeros(len(box), 5)
            box_data[:len(box)] = box
        # 综合所有的组合图片和实际框数据
        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将列表中的所有图片进行拼接
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros((h, w, 3))
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    new_boxes = merge_bboxes(box_datas, cutx, cuty)
    return new_image, new_boxes


# def normal_(annotation_line, input_shape):
#     line = annotation_line.split()
#     image = Image.open(line[0])
#     box = np.array([np.array(list(map(int, box))) for box in line[1:]])
#
#     image_w, image_h = image.size
#     image = image.transpose(Image.FLIP_LEFT_RIGHT)
#     box[:, [0, 2]] = image_w - box[:, [0, 2]]
#     return image, box
