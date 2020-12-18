from torch.utils.data import Dataset


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

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, idx):
        pass
