from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


def train_trans():
    return transforms.Compose([transforms.ToTensor()])


def preprocess_input(image):
    """
    输入图片的标准化
    :param image: 图片张量或numpy数组
    :return: 标准化后的图片张量或numpy数组
    """
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image = (image - mean) / std
    return image


class EfficientDet_Dataset(Dataset):
    def __init__(self, train_lines, image_size):
        self.train_lines = train_lines
        self.image_size = image_size
        self.trans = train_trans()

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, idx):
        lines = self.train_lines

        image, label = self.get_random_data(lines[idx], self.image_size[0:2])
        if len(label) != 0:
            # 将boxes的坐标数据转换数据类型，而分类的数据类型不变
            boxes = np.array(label[:, :4], dtype=np.float32)
            label = np.concatenate([boxes, label[:, -1:]], axis=1)
        label = np.array(label, np.float32)
        return self.trans(image), label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        """
        图片在线增强
        :param annotation_line:
        :param input_shape:
        :param jitter:
        :param hue:
        :param sat:
        :param val:
        :return:
        """
        line = annotation_line.split()
        image = Image.open(line[0])

        image_w, image_h = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            new_h = int(scale * h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * w)
            new_h = int(new_w * new_ar)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        dx = int(self.rand(0, w - new_w))  # 确保缩小后的图片在input区域内,放大图片不会出现留白
        dy = int(self.rand(0, h - new_h))  # 确保缩小后的图片在input区域内,放大图片不会出现留白
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255),))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 进行色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        # h:色调, s:饱和度, v:明度
        x = rgb_to_hsv(np.array(image) / 255.)  # 数据归一化,这样生成的hsv也是归一化的矩阵
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1  # 色度是一个角度，本取值0~360,经过归一化之后大于一的相当于多转了一圈
        x[..., 0][x[..., 0] < 0] += 1  # 色度是一个角度，本取值0~360,经过归一化之后小于零的相当于少转了一圈
        x[..., 1] *= sat  # 饱和度的变化
        x[..., 2] *= val  # 明度的变化
        x[x > 1] = 1  # 强制大于1的为1,反算RGB时不会出错
        x[x < 0] = 0  # 强制小于0的为0,反算RGB时不会出错
        image_data = Image.fromarray((hsv_to_rgb(x) * 255).astype(np.int8))

        # 调整目标框坐标
        box_data = np.zeros(len(box), 5)
        if len(box_data) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * new_w / image_w + dx
            box[:, [1, 3]] = box[:, [1, 3]] * new_h / image_h + dy

            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            box_data = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(box_data) == 0:
            return image_data, []
        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []
