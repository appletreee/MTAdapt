import numpy as np
from os.path import join as PJ
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as transforms_F


#######################################################
# Classification dataset
#######################################################
class ClassificationDataset(Dataset):
    """ VisDA closedset/openset classification dataset. """
    def __init__(self, data_root, file_name, transform=None, domain='source'):
        """
        Args:
            data_root (string):     Directory with all the images.
            file_name (string):     Path to the image with annotations.
            transform (function):   Transform image(Resize, RandomCrop, Normalize, ...etc).
            pad       (boolean):    Pading in image to square. (Fasle means no pad.)
            domain    (string):     Using domain classification dataset. ('source' or 'target')
        """
        with open(PJ(data_root, file_name), 'r') as file:
            self.image_info = file.readlines()
        self.image_info = [path.strip().split() for path in self.image_info]
        self.data_root = PJ(data_root, 'validation') if domain is 'target' else PJ(data_root, 'train')
        # self.pad = pad
        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        name = self.image_info[idx][0]
        original_image = Image.open(PJ(self.data_root, name)).convert('RGB')

        image = self.transform(original_image) if self.transform is not None else original_image

        label = int(self.image_info[idx][1])
        label = torch.Tensor([label]).to(torch.long)
        return image, label, name


#######################################################
# Semantic segmentation dataset
#######################################################
class SemanticSegmentationDataset(Dataset):
    def __init__(self, data_root, file_name, transform=None):
        ignore_label = 19
        self.ignore_label = ignore_label
        self.id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                         3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                         9: ignore_label, 10: ignore_label, 14: ignore_label, 15: ignore_label,
                         16: ignore_label, 18: ignore_label, 29: ignore_label, 30: ignore_label,
                         7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                         23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                        220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                        0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        self.classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky', 'person', 'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle', 'bicycle']

    def map_to_class_id(self, label):
        label_id = self.ignore_label * np.ones_like(label, dtype=np.uint8)
        for id, cls_id in self.id2label.items():
            label_id[label == id] = int(cls_id)
        return label_id

    def resize(self, image1, image2, size):
        # Must be NEAREST, or label id will change
        operate1 = transforms.Resize(size=size)
        operate2 = transforms.Resize(size=size, interpolation=Image.NEAREST)
        image1, image2 = operate1(image1), operate2(image2)
        return image1, image2

    def random_crop(self, image1, image2, size):
        i, j, h, w = transforms.RandomCrop.get_params(image1, output_size=size)
        image1 = transforms_F.crop(image1, i, j, h, w)
        image2 = transforms_F.crop(image2, i, j, h, w)
        return image1, image2

    def random_horizontal_flip(self, image1, image2):
        if torch.rand(1) > 0.5:
            image1 = transforms_F.hflip(image1)
            image2 = transforms_F.hflip(image2)
        return image1, image2


class CityScapesDataset(SemanticSegmentationDataset):
    """ VisDA Cityscapes dataset. (Target domain) """
    def __init__(self, data_root, file_name, transform=None):
        super().__init__(data_root, file_name, transform)
        """
        Args:
            data_root (string):     Directory with all the images.
            file_name (string):     Name of data list file (ex: train_list.txt)
            transform (function):   Transform image(Resize, RandomCrop, Normalize, ...etc).
        """

        # Load data list file
        data_root = PJ(data_root, "CityScapes")
        with open(PJ(data_root, file_name), 'r') as file:
            image_info = file.readlines()
        # Parse data names
        image_info = [path.strip() for path in image_info]

        root = self._prepare_data_root(data_root)
        self.image_info = image_info
        self.image_root = root['image']
        self.label_root = root['label']
        self.transform = transform
        self.transform_list = transform.keys() if transform else {}
        self.label_type = "gtFine_labelIds.png"

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        name = self.image_info[idx]
        image = Image.open(PJ(self.image_root, name)).convert('RGB')
        label_name = name.replace("leftImg8bit.png", self.label_type)
        label = Image.open(PJ(self.label_root, label_name))
        label = np.asarray(label)
        label = self.map_to_class_id(label)
        label = Image.fromarray(label, 'L')

        # Transforms
        if "resize" in self.transform_list:
            size = self.transform["resize"]["size"]
            image, label = self.resize(image, label, size)
        if "random_crop" in self.transform_list:
            size = self.transform["random_crop"]["size"]
            image, label = self.random_crop(image, label, size)
        if "random_horizontal_flip" in self.transform_list:
            image, label = self.random_horizontal_flip(image, label)

        image = transforms_F.to_tensor(image)
        image = transforms_F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        label = torch.from_numpy(np.array(label)).to(torch.long)
        return image, label, name

    def _prepare_data_root(self, data_root):
        image_folder = "leftImg8bit_trainvaltest/leftImg8bit"
        label_folder = "gtFine_trainvaltest/gtFine"
        # Using val set of CityScapes as our testing set
        image_root = PJ(data_root, image_folder, 'val')
        label_root = PJ(data_root, label_folder, 'val')
        return {'image': image_root, 'label': label_root}


class GTA5Dataset(SemanticSegmentationDataset):
    """ VisDA GTA5 segmentation dataset. (Source domain) """
    def __init__(self, data_root, file_name, transform=None):
        super().__init__(data_root, file_name, transform)
        """
        Args:
            data_root (string):     Directory with all the images.
            file_name (string):     Name of data list file (ex: train_list.txt)
            transform (function):   Transform image(Resize, RandomCrop, Normalize, ...etc).
        """
        data_root = PJ(data_root, "GTA5")
        # Load data list file
        with open(PJ(data_root, file_name), 'r') as file:
            image_info = file.readlines()
        # Parse data names
        image_info = image_info[0].split(', ')

        self.image_info = image_info
        self.image_root = PJ(data_root, "images")
        self.label_root = PJ(data_root, "labels")
        self.transform = transform
        self.transform_list = transform.keys() if transform else {}

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        name = self.image_info[idx]
        image = Image.open(PJ(self.image_root, name)).convert('RGB')
        label = Image.open(PJ(self.label_root, name))
        label = np.asarray(label)
        label = self.map_to_class_id(label)
        label = Image.fromarray(label, 'L')

        # Transforms
        if "resize" in self.transform_list:
            size = self.transform["resize"]["size"]
            image, label = self.resize(image, label, size)
        if "random_crop" in self.transform_list:
            size = self.transform["random_crop"]["size"]
            image, label = self.random_crop(image, label, size)
        if "random_horizontal_flip" in self.transform_list:
            image, label = self.random_horizontal_flip(image, label)
        image = transforms_F.to_tensor(image)
        image = transforms_F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        label = torch.from_numpy(np.array(label)).to(torch.long)
        return image, label, name


if __name__ == '__main__':
    data_root = "./VisDA_17/semantic_segmentation"
    file_name = "train_list.txt"
    transform = {"random_crop": {"size": (600, 600)}, "random_horizontal_flip222": 1,
                 "resize": {"size": 1024}}
    mode = 'train'
    dataset = GTA5Dataset(data_root, file_name, transform, mode)
    image, label, name = next(iter(dataset))

    data_root = "./VisDA_17/semantic_segmentation"
    file_name = "test_list.txt"
    transform = None
    mode = 'test'
    dataset = CityScapesDataset(data_root, file_name, transform, mode)
    image, label, name = next(iter(dataset))
