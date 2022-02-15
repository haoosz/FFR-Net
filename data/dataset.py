from __future__ import absolute_import
from __future__ import print_function
import PIL
import cv2
import torch
import os
import glob as gb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as tf
from random import choice

def get_mask_path(data_path, image_path): 
    folder, img = image_path.strip().split("/")
    num, suffix = img.strip().split(".")
    dirs = os.listdir(os.path.join(data_path,folder))
    for file in dirs:
        if file.startswith(num) and file != img:
            return os.path.join(folder,file)
    print("Cannot Find Mask Image!!!")

class LFWData(Dataset):
    """
    LFW dataset, used for face verification test.
    """
    def __init__(self, face_root, pairs_list, test_ocl_num, transform=None, img_size=(112, 112)):
        self.img_size = img_size
        self.face_img_dir = face_root
        self.pair_txt = pairs_list
        self.ocl_nums = test_ocl_num
        self.transform = transform
        self.get_pair_info()

    def get_pair_info(self):
        # Read image information
        with open(self.pair_txt) as f:
            pairs_lines = f.readlines()[1:]
        self.pair_names = []
        self.label = []
        for i in pairs_lines:
            p = i.strip().split()
            if 3==len(p):
                sameflag = 1
                name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
            if 4==len(p):
                sameflag = 0
                name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
            self.pair_names.append([name1, name2])
            self.label.append(sameflag)
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx): 
        sample = {}
        path1 = self.pair_names[idx][0]
        path2 = self.pair_names[idx][1]
        
        if self.ocl_nums >= 1:
            path1 = get_mask_path(self.face_img_dir, path1)
        if self.ocl_nums >= 2:
            path2 = get_mask_path(self.face_img_dir, path2)

        img1 = Image.open(os.path.join(self.face_img_dir, path1)).convert("RGB")
        img2 = Image.open(os.path.join(self.face_img_dir, path2)).convert("RGB")
        r1,g1,b1 = img1.split()
        img1=Image.merge('RGB',(b1,g1,r1))
        r2,g2,b2 = img2.split()
        img2=Image.merge('RGB',(b2,g2,r2))

        # random flip
        self.p = 0.5
        if random.random() < self.p:
            img1 = tf.hflip(img1)
            img2 = tf.hflip(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        sample['img1'] = img1
        sample['img2'] = img2
        sample['label'] = torch.tensor(self.label[idx]).long() 
        sample['idx'] = torch.tensor(idx).long()
        return sample

class CASIA(Dataset):
    """CASIA web face dataset.
    -----------------------------------
    # Args:
        - data_txt: txt file contains image names and label, in the following format:
            `path/to/image label`
        - size: (W, H)
    """
    def __init__(self, data_root, img_list, transform=None, shuffle=False, size=(112, 112)):
        self.face_img_dir  = data_root
        self.face_img_list = img_list
        self.shuffle       = shuffle
        self.size          = size
        self.ids           = list(range(10575))
        self.transform     = transform
        self.get_img_label()

    def get_img_label(self):
        # Get face image names and labels
        self.list_all = []
        self.img_paths = {i:[] for i in self.ids}
        for i in open(self.face_img_list).readlines():
            path, label = i.strip().split()
            label = int(label)
            self.list_all.append([path, label])
            self.img_paths[label].append(path)
         
        if self.shuffle:
            random.shuffle(self.ids)
            random.shuffle(self.list_all)

    def __len__(self):
        #  return len(self.ids)
        return len(self.list_all)

    def __getitem__(self, idx):
        sample = {}

        img_path = os.path.join(self.face_img_dir, self.list_all[idx][0])
        another_path = self.list_all[idx][0]

        another_path = get_mask_path(self.face_img_dir, another_path)

        mask_path = os.path.join(self.face_img_dir, another_path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        r1,g1,b1 = img.split()
        img=Image.merge('RGB',(b1,g1,r1))
        r2,g2,b2 = mask.split()
        mask=Image.merge('RGB',(b2,g2,r2))

        if img.size != self.size:
            img = img.resize(self.size, Image.BICUBIC)
        if mask.size != self.size:
            mask = mask.resize(self.size, Image.BICUBIC)

        # random flip
        self.p = 0.5
        if random.random() < self.p:
            img = tf.hflip(img)
            mask = tf.hflip(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        sample['img1'] = img
        sample['img2'] = mask
        sample['label'] = self.list_all[idx][1]
        sample['label'] = torch.tensor(sample['label']).long()

        return sample
