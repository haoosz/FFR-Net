"""
Evaluate lfw accuracy.
Reference: https://github.com/clcarwin/sphereface_pytorch/blob/master/lfw_eval.py
Note:
    - To keep consistent with face SR task, the faces are not aligned.
    - Flipped features are used.
"""

import os,sys,cv2,random,datetime
import numpy as np
import zipfile
from PIL import Image

from utils import utils
import multiprocessing as mp
from time import time
import skimage.transform as tf 
from matlab_cp2tform import get_similarity_transform_for_cv2

import torch
from torch.utils.data import Dataset, DataLoader

from models.model_spa import SPANet
from skimage.io import imsave


class LFWData(Dataset):
    """
    img_size: cropped image size, (W, H)
    """
    def __init__(self, data_root, img_size=(96, 112)):
        self.data_root = data_root
        self.img_size = img_size
        self.img_dir = os.path.join(data_root, 'images')
        self.pair_txt = os.path.join(data_root, 'pairs.txt')
        self.landmark_txt = os.path.join(data_root, 'lfw_landmark.txt')
        self.get_pair_info()

    def get_pair_info(self):
        # Read landmarks
        self.image_names = []
        self.landmark = {} 
        with open(self.landmark_txt) as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n','').split('\t')
            self.landmark[l[0]] = [int(k) for k in l[1:]]
            self.image_names.append(l[0])

    def align(self, src_img, src_pts):
        src_img = np.array(src_img)
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        #  trans = tf.SimilarityTransforTruem()
        #  trans.estimate(s, r)
        #  face_img = cv2.warpAffine(src_img, trans.params[:2], self.img_size)
        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, self.img_size)
        return face_img

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx): 
        img_name = self.image_names[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = self.align(img, self.landmark[img_name])
        img = np.array(img)
        inc = 0 
        img = np.pad(img, ((8+inc, 8+inc), (16+inc, 16+inc), (0, 0)), 'constant')
        #  img_hr = self.align(img, self.landmark[img_name])
        img_hr = cv2.resize(img, (128, 128), cv2.INTER_CUBIC)
        img_hr = ((img_hr - 127.5)/128).transpose(2, 0, 1)
        hr_img_bgr = torch.from_numpy(img_hr).float()

        img_lr = cv2.resize(img, (16, 16), cv2.INTER_LINEAR)
        img_lr = cv2.resize(img_lr, (128, 128), cv2.INTER_CUBIC)
        img_lr = ((img_lr - 127.5)/128).transpose(2, 0, 1)
        lr_img_bgr = torch.from_numpy(img_lr).float()

        return hr_img_bgr, lr_img_bgr


def calculate_distance(data_root, use_flip=False, use_gpu=True):

    weight_path = './pretrain_models/sphere20a_20171020.pth' 
    rec_model = sphere20a()
    rec_model.load_state_dict(utils.load(weight_path))

    weight_path = '/ciufengchen/Face-SR/weight/model_spa_net-loss_l2-gan-res16-norm_bn_bn-fsrdata-weight-1.0000_0.0000_0.0000_0.0000_0.0000/16-128-0044000-G.pth.gzip'
    sr_model = SPANet()
    sr_model.load_state_dict(utils.load(weight_path))
    if use_gpu:
        rec_model.cuda()
        sr_model.cuda()

    dataset = LFWData(data_root)
    data_loader = DataLoader(dataset, 24, False, num_workers=8, pin_memory=True)

    all_distance = []
    all_label = []
    count = 0
    for data in data_loader:
        hr_img_bgr, lr_img_bgr = data 
        if use_gpu: 
            hr_img_bgr = hr_img_bgr.cuda()
            lr_img_bgr = lr_img_bgr.cuda()
        lr_img_rgb = lr_img_bgr[:, [2, 1, 0]]
        hr_img_rgb = hr_img_bgr[:, [2, 1, 0]]

        _, sr_img_rgb = sr_model(lr_img_rgb)

        visual_imgs = [lr_img_rgb, sr_img_rgb, hr_img_rgb]
        visual_imgs = [utils.batch_tensor_to_img(x) for x in visual_imgs]
        visual_comb = np.concatenate(visual_imgs, 2)
        for i in range(visual_comb.shape[0]):
            count += 1
            imsave(os.path.join('./test_model', '{:05d}.jpg'.format(count)), 
                    visual_comb[i].astype(np.uint8))
        exit()

if __name__ == '__main__':
    from models.trainer_rec import RecNet
    from models.net_sphere import sphere20a
    data_root = '/ciufengchen/data_sr/LFW/'
    calculate_distance(data_root)
    #  model = sphere20a()
    #  model.load_state_dict(utils.load(weight_path))
    #  get_avg_accuracy(model, data_root, True)

