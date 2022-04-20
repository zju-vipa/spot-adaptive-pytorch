import os
import sys
import pickle
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 此处为我自己训练cifar
transform_cifar10_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


class CIFAR10_Spilt(VisionDataset):
    # Spilt Cifar10 dataset into two parts, and have different indices as label 
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_Spilt, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []

        self._load_meta()
        # print(self.class_to_idx)
        # print(self.idx_to_class)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # print('path: {}'.format(path))
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
            # print('classes: {}'.format(self.classes))
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.idx_to_class = {str(self.class_to_idx[key]): key for key in self.class_to_idx.keys()}

        return self.class_to_idx, self.idx_to_class

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100_Spilt(CIFAR10_Spilt):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }



def get_idx_to_class(root='/home/zhfeing/datasets/cifar', dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        dataset = CIFAR10_Spilt(root=root)
    elif dataset == 'CIFAR100':
        dataset = CIFAR100_Spilt(root=root)


if __name__ == "__main__":
    dataset_folder = '/home/zhfeing/datasets/cifar'
    # trainset1 = CIFAR10_Spilt(root=dataset_folder, 
    #                     target_classes = ['bird', 'cat','deer', 'dog', 'frog', 'horse'], train=True,
    #                     download=False, transform=transform_cifar10_train)
    # testset1 = CIFAR10_Spilt(root=dataset_folder, 
    #                     target_classes = ['bird', 'cat','deer', 'dog', 'frog', 'horse'], train=False,
    #                     download=False, transform=transform_cifar10_test)

    # trainset2 = CIFAR10_Spilt(root=dataset_folder, 
    #                     target_classes = ['airplane', 'automobile', 'ship', 'truck'], train=True,
    #                     download=False, transform=transform_cifar10_train)
    # testset2 = CIFAR10_Spilt(root=dataset_folder, 
    #                     target_classes = ['airplane', 'automobile', 'ship', 'truck'], train=False,
    #                     download=False, transform=transform_cifar10_test)

    # # print('train data size: {}'.format(len(trainset1)))
    # # print('test data size: {}'.format(len(testset1)))

    # print('train data size: {} {}'.format(len(trainset1), len(trainset2)))
    # print('test data size: {} {}'.format(len(testset1), len(testset2)))

    # random_labels = random.sample(cifar100_classes, 10)
    
    # print(random_labels)

    # trainset1 = CIFAR100_Spilt(root=dataset_folder, 
    #                     target_classes=random_labels, train=True,
    #                     download=False, transform=transform_cifar10_train)

    random_labels = torch.load('checkpoint/cifar100_spilt/32/128_0.1/ckpt.pth')['labels']
    print(random_labels)
    exit("0")
    root = '/nfs/xmq/data/dataset'

    # target_classes = [['snail', 'keyboard', 'tractor', 'snake', 'ray'], ['chair', 'whale', 'road', 'sea', 'possum']]
    # trainset = CIFAR100_Spilt_Branch(root='/nfs/xmq/data/dataset', 
    #                                 target_classes_lists=target_classes, train=True,
    #                                 download=False, transform=transform_cifar10_train)

    # dataset = CIFAR100_Spilt_Branch(root=root)
    # CIFAR100_Spilt_Branch._fetch_dataloaders_random_(
    #             '/nfs/xmq/data/dataset/cifar-100-python', 1000, 1, random_labels)

    dataset = CIFAR100_Spilt_Random(root=root, target_labels=random_labels,
                                    random_data_num=1000, batch_size=1)
    dataloader = dataset._fetch_dataloaders_random_()

    print(type(dataloader))

    for i, data in enumerate(dataloader['data']):
        print(i, len(data))
