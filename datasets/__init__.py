import os
import cv2
import random
import torch.utils.data as uData
import h5py as h5

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    # elif config.data.dataset == 'CELEBA':
    #     if config.data.random_flip:
    #         dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
    #                          transform=transforms.Compose([
    #                              transforms.CenterCrop(140),
    #                              transforms.Resize(config.data.image_size),
    #                              transforms.RandomHorizontalFlip(),
    #                              transforms.ToTensor(),
    #                          ]), download=True)
    #     else:
    #         dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
    #                          transform=transforms.Compose([
    #                              transforms.CenterCrop(140),
    #                              transforms.Resize(config.data.image_size),
    #                              transforms.ToTensor(),
    #                          ]), download=True)
    #
    #     test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
    #                           transform=transforms.Compose([
    #                               transforms.CenterCrop(140),
    #                               transforms.Resize(config.data.image_size),
    #                               transforms.ToTensor(),
    #                           ]), download=True)

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root='D:\datasets', split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=False)
        else:
            dataset = CelebA(root=os.path.join(args.exp_my, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=False)

        test_dataset = CelebA(root='D:\datasets', split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=False)

    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if config.data.seis_rescaled:
        X = (X+1.)*0.5

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.
    if config.data.seis_rescaled:
        X = (X*2)-1
    # return torch.clamp(X, 0.0, 1.0)
    return torch.clamp(X, -1.0, 1.0)

def inverse_data_transform_1(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.
    if config.data.seis_rescaled:
        X = (X*2)-1
    # return torch.clamp(X, 0.0, 1.0)
    return torch.clamp(X, 0, 1.0)



# Base Datasets
class BaseDataSetImg(uData.Dataset):
    def __init__(self, im_list, length, pch_size=(128,128)):
        '''
        Args:
            im_list (list): path of each image
            length (int): length of Datasets
            pch_size (int): patch size of the cropped patch from each image
        '''
        super(BaseDataSetImg, self).__init__()
        self.im_list = im_list
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(im_list)

    def __len__(self):
        return self.length

    def crop_patch(self, im):
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size[0] or W < self.pch_size[0]:
            H = max(self.pch_size[0], H)
            W = max(self.pch_size[0], W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size[0])
        ind_W = random.randint(0, W-self.pch_size[0])
        pch = im[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0]]
        return pch

class BaseDataSetH5(uData.Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        C = int(C2/2)
        ind_H = random.randint(0, H-self.pch_size[0])
        ind_W = random.randint(0, W-self.pch_size[0])
        im_noisy = np.array(imgs_sets[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0], :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0], C:])
        return im_gt, im_noisy

