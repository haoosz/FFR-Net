import torch
from torchvision import transforms
import torchvision.transforms.functional as tf

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import random
import math
from data.dataset import transform_data

def add_occluder_block(origin_img, occ_img, bsz):
    origin_img = np.array(origin_img).astype(float)
    ih, iw = origin_img.shape[:2]

    lx = np.random.randint(0, iw - bsz)
    ly = np.random.randint(0, ih - bsz)
    rx = lx + bsz 
    ry = ly + bsz 
    lx = max(0, lx)
    rx = min(iw, rx)
    ly = max(0, ly)
    ry = min(ih, ry)
    full_mask = np.ones((ih, iw, 1), dtype=float)
    full_mask[ly:ry, lx:rx] = 0
    origin_img[ly:ry, lx:rx] = 0 
    return origin_img, full_mask


def add_occluder(origin_img, occ_img, occ_type):
    origin_img = np.array(origin_img).astype(float) 
    occ_img = np.array(occ_img).astype(float)
    #  origin_img = np.array(origin_img)
    #  occ_img = np.array(occ_img)
    ih, iw = origin_img.shape[:2]
    #  Locations for 112x96
    if ih == 112:
        locations = {
            'forehead': [[48, 30], [90, 32]], 
            'lefteye': [[30.3, 51.7], [30, 30]],       
            'righteye': [[65.5, 51.5], [30, 30]],
            'eyes': [[48, 51.5], [80, 42]],
            'mouth_nose': [[48.1, 88.3], [86, 48]],
            'mouth': [[48.1, 88], [40, 999]],
            'leftface': [[30, 71], [45, 80]],
            'rightface': [[70, 71], [45, 80]],
            }      
    elif ih == 128:
    #  Locations for 128x128
        locations = {
                'forehead': [[68, 16], [100, 30]], 
                'lefteye': [[39, 39], [40, 30]],       
                'righteye': [[92, 39], [40, 30]],
                'twoeyes': [[65.5, 39], [92, 30]],
                'mouth': [[64, 90], [70, 50]],
                'leftface': [[40, 70], [55, 100]],
                'rightface': [[90, 70], [55, 100]],
                }

    oh, ow = occ_img.shape[:2]
    occ_center, max_occ_size = locations[occ_type]
    lx, ly = (np.array(occ_center) - np.array([ow/2, oh/2])).astype(int)
    rx, ry = (np.array(occ_center) + np.array([ow/2, oh/2])).astype(int)
    # Resize occluders
    if occ_type == 'mouth':
        ly = int(occ_center[1])
        ry = oh + ly
    if occ_type == 'random':
        lx = np.random.randint(0, iw - ow)
        ly = np.random.randint(0, ih - oh)
        rx = lx + ow
        ry = ly + oh
    lx = max(0, lx)
    rx = min(iw, rx)
    ly = max(0, ly)
    ry = min(ih, ry)
    # ---------- Put occlusion --------------
    mask = np.ones((ry-ly, rx-lx, 1))
    occ_img = occ_img[0: ry-ly, 0: rx-lx, :3]
    origin_img[ly:ry, lx:rx] = origin_img[ly:ry, lx:rx] * (1 - mask) + occ_img * mask

    # Seamless Clone
    #  kernel = np.ones((8, 8), np.uint8)
    #  smask = cv2.dilate(mask, kernel, iterations=1)
    #  origin_img = cv2.seamlessClone((occ_img*255).astype(np.uint8), (origin_img*255).astype(np.uint8), 
            #  (smask*255).astype(np.uint8), ((lx+rx)//2, (ly+ry)//2), cv2.NORMAL_CLONE)
    #  origin_img = origin_img.astype(float)/255

    # ---------- Generate mask --------------
    full_mask = np.ones((ih, iw, 1), dtype=float)
    full_mask[ly:ry, lx:rx] = full_mask[ly:ry, lx:rx] * (1 - mask)
    # extend mask
    kernel = np.ones((8, 8), np.uint8)
    full_mask = 1 - cv2.dilate(1 - full_mask, kernel, iterations=1)

    return origin_img, full_mask[..., np.newaxis]


class RandomHorizontalFlip():
    """
    Horizontally flip ALL the given video frames randomly with a probability of 0.5.
    """
    def __init__(self):
        self.p = 0.5

    def __call__(self, sample):
        if random.random() < self.p:
            sample['img1'] = tf.hflip(sample['img1'])
            sample['img2'] = tf.hflip(sample['img2'])
        return sample

class MAddMask():
    """
    Add up/down mask.
    -----------------------
    # Args
        - size_range: max percentage of occlusion size.
    """
    def __init__(self, max_mask_percent=None, mask_prob=0.5):
        self.masks = [
                (0, 0, 28, 14), # upper mask
                (0, 14, 28, 28), # lower mask
                ]
        self.mask_prob = mask_prob

    def __call__(self, sample):
        dic = {}
        w, h = sample.size
        mask = np.ones((w, h))
        lx, ly, rx, ry = random.choice(self.masks)
        mask[ly:ry, lx:rx] = 0
        sample_array = np.array(sample)
        tmp_prob = np.random.random()
        if tmp_prob  < self.mask_prob:
            sample_array = sample_array * mask + (1-mask)*128
            mask_label = 1
        else:
            mask_label = 0
        dic['img'] = sample_array[:, :, np.newaxis].astype(np.float32)
        dic['mask_label'] = mask_label
        dic['mask'] = mask[None, ...].astype(np.float32)
        return dic

class MToTensor():
    """
    #  Normalize the image to [-1, 1], and convert them to tensor.
    Convert numpy array to tensor.
    Swap axis of face: (H, W, C) -> (C, H, W)
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        sample['img'] = self.to_tensor(sample['img']) 
        sample['mask_label'] = torch.tensor(sample['mask_label']).long()
        sample['mask'] = torch.tensor(sample['mask'])
        return sample 

class MNormalize():
    """
    #  Normalize the image tensor"""
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, sample):
        sample['img'] = self.normalize(sample['img'])
        return sample 

class AddOcl():
    """
    Add predefined occluders.
    -----------------------
    # Args
        - size_range: max percentage of occlusion size.
        - mask_percent: how many images to add mask 
    """
    def __init__(self, bocc_sz):
        self.bocc_sz = bocc_sz 

    def __call__(self, sample):
        face_img = np.array(sample['img2'])
        ocl_img = np.array(sample['ocl_img'])
        ocl_type = sample['ocl_type']
        tmp_prob = np.random.random()
        mask = np.ones((face_img.shape[0], face_img.shape[1], 1), dtype=float)
        if self.bocc_sz > 0:
            face_img, mask = add_occluder_block(face_img, ocl_img, self.bocc_sz)
        else:
            face_img, mask = add_occluder(face_img, ocl_img, ocl_type)
        mask_label = 1

        sample['img2'] = Image.fromarray(face_img.astype(np.uint8))
        sample['mask_label'] = mask_label
        sample['mask'] = mask

        return sample 

class ToTensor():
    """
    #  Normalize the image to [-1, 1], and convert them to tensor.
    Convert numpy array to tensor.
    Swap axis of face: (H, W, C) -> (C, H, W)
    """
    def __init__(self):
        self.dic = {}

    def __call__(self, sample):
        dic = self.dic
        dic['img1'] = torch.from_numpy(transform_data(sample['img1']).transpose(2,0,1)).float()
        dic['img2'] = torch.from_numpy(transform_data(sample['img2']).transpose(2,0,1)).float()
        
        dic['label'] = torch.tensor(sample['label']).long()
        dic['mask_label'] = torch.tensor(sample['mask_label']).long()
        dic['ocl_label'] = (torch.tensor(sample['ocl_label']) * sample['mask_label']).float()
        if 'mask' in sample.keys():
            dic['mask'] = torch.tensor(sample['mask'].transpose(2, 0, 1)).float()
        return dic 

class LFWAddOcl():
    """
    Add predefined occluders.
    -----------------------
    # Args
        - size_range: max percentage of occlusion size.
        - mask_percent: how many images to add mask 
    """
    def __init__(self, ocl_nums):
        self.ocl_nums = ocl_nums

    def __call__(self, sample):
        face_img1 = np.array(sample['img1'])
        face_img2 = np.array(sample['img2'])
        ocl_img1 = np.array(sample['ocl_img1'])
        ocl_img2 = np.array(sample['ocl_img2'])

        if self.ocl_nums >= 1:
            face_img1, mask1 = add_occluder(face_img1, ocl_img1, sample['ocl_type1'])
        if self.ocl_nums >= 2:
            face_img2, mask2 = add_occluder(face_img2, ocl_img2, sample['ocl_type2'])

        sample['img1'] = Image.fromarray(face_img1.astype(np.uint8))
        sample['img2'] = Image.fromarray(face_img2.astype(np.uint8))
        
        return sample 

class LFWToTensor():
    """
    #  Normalize the image to [-1, 1], and convert them to tensor.
    Convert numpy array to tensor.
    Swap axis of face: (H, W, C) -> (C, H, W)
    """
    def __init__(self):
        self.dic = {}

    def __call__(self, sample):
        dic = self.dic
        dic['img1'] = torch.from_numpy(transform_data(sample['img1']).transpose(2,0,1)).float()
        dic['img2'] = torch.from_numpy(transform_data(sample['img2']).transpose(2,0,1)).float()
        dic['label'] = torch.tensor(sample['label_same']).long() 
        return dic 


