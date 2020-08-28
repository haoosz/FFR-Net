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
from lfw.matlab_cp2tform import get_similarity_transform_for_cv2

import torch
from torch.utils.data import Dataset, DataLoader

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
        self.landmark = {} 
        with open(self.landmark_txt) as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n','').split('\t')
            self.landmark[l[0]] = [int(k) for k in l[1:]]

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

    def gen_occlusion_mask(self, size):
        w, h = self.img_size
        mask = np.ones((h, w, 1))
        pos_y = np.random.randint(0, w - size[0])
        pos_x = np.random.randint(0, h - size[1])
        mask[pos_y: pos_y+size[0], pos_x : pos_x+size[1]] = 0
        return mask

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
        return len(self.label)

    def __getitem__(self, idx): 
        imgs_tensor = []
        mask = self.gen_occlusion_mask(MASK_SIZE)
        for i in range(2):
            img_name = self.pair_names[idx][i]
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            img = self.align(img, self.landmark[img_name])
            if i == 0:
                img = img * mask
            img = ((img - 127.5)/128).transpose(2, 0, 1)
            imgs_tensor.append(torch.from_numpy(img).float())

        label = self.label[idx]
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask).float()
        return imgs_tensor[0], imgs_tensor[1], mask, label

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    if shuffle: random.shuffle(base)
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold 


def calculate_distance(data_loader, encoder, recnet, use_flip=False, use_gpu=True):
    #  model.eval()
    #  if use_gpu: model.cuda()
    all_distance_new = []
    all_distance = []
    all_label = []
    for data in data_loader:
        img1, img2, labels = data['img1'], data['img2'], data['label'] 
        if use_gpu: 
            img1 = img1.cuda()
            img2 = img2.cuda()

        with torch.no_grad():
            # # sapce + channel
            # feat_map1, f1, _ = encoder(img1)
            # f1_new, _, _, _, _, _, _ = recnet(feat_map1)
            # feat_map2, f2, _ = encoder(img2)
            # f2_new, _, _, _, _, _, _ = recnet(feat_map2)

            #space
            feat_map1, f1, _ = encoder(img1)
            f1_new, _, _, _, _, _, _, _, _ = recnet(feat_map1)
            feat_map2, f2, _ = encoder(img2)
            f2_new, _, _, _, _, _, _, _, _ = recnet(feat_map2)

        # if use_flip:
        #     f1_flip = model(imgs1.flip(3))
        #     f2_flip = model(imgs2.flip(3))
        #     f1 = torch.cat((f1, f1_flip), dim=1)
        #     f2 = torch.cat((f2, f2_flip), dim=1)
        cosdistance_new = torch.sum(f1_new * f2_new, dim=1) /(f1_new.norm(dim=1)*f2_new.norm(dim=1)+1e-8)
        all_distance_new += cosdistance_new.tolist()
        cosdistance = torch.sum(f1 * f2, dim=1) /(f1.norm(dim=1)*f2.norm(dim=1)+1e-8)
        all_distance += cosdistance.tolist()
        all_label += labels.tolist()
    return np.array([all_distance_new, all_label]).T, np.array([all_distance, all_label]).T


def get_fold_accuracy(fold, predicts):
    thresholds = np.arange(-1.0, 1.0, 0.005)
    best_thresh = find_best_threshold(thresholds, predicts[fold[0]])
    test_acc = eval_acc(best_thresh, predicts[fold[1]])
    return best_thresh, test_acc 

def get_accuracy(result,verbose=False):
    avg_acc = 0
    for res in result:
        threshold, acc = res.get()
        if verbose:
            print('Best threshold: {:.4f}; Test accuracy: {:.4f}'.format(threshold, acc))
        avg_acc += acc
    avg_acc /= 10
    if verbose: print('Average accuracy: {}'.format(avg_acc))
    return avg_acc

def get_avg_accuracy(encoder, recnet, data_loader, verbose=False):
    pred_new, pred = calculate_distance(data_loader, encoder, recnet)
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    
    pool = mp.Pool(processes=10)
    result = []
    result_new = []
    for f in folds:
        result_new.append(pool.apply_async(get_fold_accuracy, (f, pred_new)))
        result.append(pool.apply_async(get_fold_accuracy, (f, pred)))
    pool.close()
    pool.join()

    avg_acc = get_accuracy(result)
    avg_acc_new = get_accuracy(result_new)
    return avg_acc_new, avg_acc

if __name__ == '__main__':
    from models.net_sphere import sphere20a
    data_root = '/ciufengchen/data_sr/LFW/'
    weight_path = './pretrain_models/sphere20a_20171020.pth' 
    model = sphere20a(False, 10574, True)
    #  weight_path = './weight/model_partial_sphereface-loss_sphere-train/0116000pth.gzip' 
    #  model = sphere20a(True, 10575, True)
    model.load_state_dict(utils.load(weight_path))
    get_avg_accuracy(model, data_root, (20, 30), True)

