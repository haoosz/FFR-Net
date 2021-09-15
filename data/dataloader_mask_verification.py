from __future__ import absolute_import
from __future__ import print_function
import PIL
import cv2
import torch
import os
import glob as gb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf

def dataloader_mask_verification(opts):
    is_train = (opts.phase.lower() == 'train')

    shuffle   = True if is_train and not opts.debug else False
    drop_last = True if is_train and not opts.debug else False
         
    face_root = '../mask_data' 
    data = Mask_Data(face_root)

    data_loader = DataLoader(data, opts.batch_size, shuffle, num_workers=opts.nThread, 
        pin_memory=True, drop_last=drop_last)

    #  data_loader = ar_dataset.get_dataloader()
    return data_loader

class Mask_Data(Dataset):
    """
    LFW dataset, used for face verification test.
    """
    def __init__(self, face_root, img_size=(112, 112)):
        self.face_root = face_root
        self.face_img_dir = os.path.join(face_root, 'masked_whn_112_align_v4')
        self.pair_txt = os.path.join(face_root, 'masked_pairs_new_v4.txt')
        self.get_pair_info()

    def get_pair_info(self):
        # Read image information
        with open(self.pair_txt) as f:
            pairs_lines = f.readlines()
        self.pair_names = []
        self.label = []
        for i in pairs_lines:
            p = i.strip().split()
            sameflag = 0
            masked = p[0]
            nonmasked = p[1]
            self.label.append(int(p[2]))
            self.pair_names.append([masked, nonmasked])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx): 
        sample = {}

        img_name_1 = self.pair_names[idx][0]
        img_name_2 = self.pair_names[idx][1]
        
        img1 = Image.open(os.path.join(self.face_img_dir, img_name_1))
        img2 = Image.open(os.path.join(self.face_img_dir, img_name_2))

        tf_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        if random.random() < 0.5:
            img1 = tf.hflip(img1)
            img2 = tf.hflip(img2)

        img1 = tf_transform(img1)
        img2 = tf_transform(img2)
        
        sample['img1'] = img1
        sample['img2'] = img2
        sample['label'] = torch.tensor(self.label[idx]).long() 
        sample['idx'] = torch.tensor(idx).long()
        # if self.transform:
        #     sample = self.transform(sample)

        return sample
        
if __name__ == "__main__":
    face_root = '../../mask_data' 
    data = Mask_Data(face_root)
    sample = data[1]
    img1 = sample['img1']
    img2 = sample['img2']
    label = sample['label']
    idx = sample['idx']
    print(img1)
    print(img2)
    print(label)
    print(idx)