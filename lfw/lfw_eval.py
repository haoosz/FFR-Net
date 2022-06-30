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
from models.trainer import normalization
from data.dataloader_mask_verification import Mask_Data
from PIL import Image
from pretrain.model_ir_se50 import ir_se_50_512
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

def save_wrong_imgs(wrong_idx, new):
    face_root = '../mask_data' 
    data = Mask_Data(face_root)
    Tensor2Image = transforms.ToPILImage()
    for i in wrong_idx:
        sample = data[i]
        img1 = sample['img1']*0.5+0.5
        img2 = sample['img2']*0.5+0.5
        img1 = Tensor2Image(img1).convert('RGB')
        img2 = Tensor2Image(img2).convert('RGB')
        if new == 0:
            img1.save('./wrong_images/{:4d}_1.png'.format(i))
            img2.save('./wrong_images/{:4d}_2.png'.format(i))
        if new == 1:
            img1.save('./wrong_images_new/{:4d}_1.png'.format(i))
            img2.save('./wrong_images_new/{:4d}_2.png'.format(i))

def eval_acc(threshold, diff, save_wrong, new=0):
    y_true = []
    y_predict = []
    y_idx = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
        y_idx.append(int(d[2]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    if save_wrong == 1:
        y_idx= np.array(y_idx)
        wrong_idx = y_idx[y_true!=y_predict] 
        save_wrong_imgs(wrong_idx, new)   
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts, save_wrong=0)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold 

def visualize(nonocl, M_space_non, M_channel_non, feat_map_non, ocl, M_space_ocl, M_channel_ocl, feat_map_ocl):
    map_non = utils.tensor_to_numpy(M_space_non.view(M_space_non.size(0),7,7,7,7))
    map_ocl = utils.tensor_to_numpy(M_space_ocl.view(M_space_ocl.size(0),7,7,7,7))
    weightmap_ocl = np.ones([M_space_ocl.size(0),67,67])*np.max(map_ocl) # 67=(7+3)*6+7
    weightmap_non = np.ones([M_space_non.size(0),67,67])*np.max(map_non) 
    for i in range(7): # H
        for j in range(7): # W
            weightmap_non[...,i*10:i*10+7,j*10:j*10+7]=map_non[...,i,j]
            weightmap_ocl[...,i*10:i*10+7,j*10:j*10+7]=map_ocl[...,i,j]
    weightmap_ocl = normalization(weightmap_ocl)[:,np.newaxis,:,:]*255
    weightmap_non = normalization(weightmap_non)[:,np.newaxis,:,:]*255

    channelweight_ocl = normalization(utils.tensor_to_numpy(M_channel_ocl))[:,np.newaxis,:,:]*255
    channelweight_non = normalization(utils.tensor_to_numpy(M_channel_non))[:,np.newaxis,:,:]*255

    channel_non = torch.argmax(M_channel_non,2).unsqueeze(2).unsqueeze(3).repeat(1,1,7,7)
    channel_ocl = torch.argmax(M_channel_ocl,2).unsqueeze(2).unsqueeze(3).repeat(1,1,7,7)

    featmap_non_array = utils.tensor_to_numpy(feat_map_non)
    featmap_ocl_array = utils.tensor_to_numpy(feat_map_ocl)

    non_featmap = utils.tensor_to_numpy(torch.gather(feat_map_non, 1, channel_non))
    ocl_featmap = utils.tensor_to_numpy(torch.gather(feat_map_ocl, 1, channel_ocl))

    channelmap_non = normalization(np.mean(non_featmap,1))*255
    channelmap_ocl = normalization(np.mean(ocl_featmap,1))*255

    # for m in range(channelmap_non.shape[0]):
    #         non_featmap = np.mean(featmap_non_array[m,channel_non[m,:],:,:],1)
    #         channelmap_non[m,:,:,:] = normalization(non_featmap) * 255
    #         ocl_featmap = np.mean(featmap_ocl_array[m,channel_ocl[m,:],:,:],1)
    #         channelmap_ocl[m,:,:,:] = normalization(ocl_featmap) * 255
    
    mean = (131.0912, 103.8827, 91.4953)
    nonocl_image = utils.tensor_to_numpy(nonocl).transpose(0,2,3,1) + mean
    ocl_image = utils.tensor_to_numpy(ocl).transpose(0,2,3,1) + mean

    out = []
    out.append(nonocl_image.transpose(0,3,1,2))
    out.append(weightmap_non)
    out.append(channelweight_non)
    out.append(channelmap_non[:,np.newaxis,:,:])

    out.append(ocl_image.transpose(0,3,1,2))
    out.append(weightmap_ocl)
    out.append(channelweight_ocl)
    out.append(channelmap_ocl[:,np.newaxis,:,:])

    for i in range(8):
        if i==0 or i==4:
            out[i] = utils.batch_numpy_to_image(out[i])
        else:
            out[i] = utils.batch_numpy_to_image(out[i], size=(224, 224))

    imgs = []
    for i in range(20):
        tmp_imgs = [x[i] for x in out]
        imgs.append(np.hstack(tmp_imgs))
    imgs = np.vstack(imgs)
    return imgs
    

def calculate_distance(data_loader, encoder, recnet, flag=0, use_flip=False, use_gpu=True):
    #  model.eval()
    #  if use_gpu: model.cuda()
    all_distance_new = []
    all_distance = []
    all_label = []
    all_idx = []
    iter = 0
    for data in data_loader:
        img1, img2, labels, idx = data['img1'], data['img2'], data['label'], data['idx']
        if use_gpu: 
            img1 = img1.cuda()
            img2 = img2.cuda()

        with torch.no_grad():
            feat_map1, f1 = encoder(img1)
            f1_new, _ = recnet(feat_map1)
            feat_map2, f2 = encoder(img2)
            f2_new, _ = recnet(feat_map2)
        iter += 1
        cosdistance_new = torch.sum(f1_new * f2_new, dim=1) /(f1_new.norm(dim=1)*f2_new.norm(dim=1)+1e-8)
        all_distance_new += cosdistance_new.tolist()
        cosdistance = torch.sum(f1 * f2, dim=1) /(f1.norm(dim=1)*f2.norm(dim=1)+1e-8)
        all_distance += cosdistance.tolist()
        all_label += labels.tolist()
        all_idx += idx.tolist()
    return np.array([all_distance_new, all_label, all_idx]).T, np.array([all_distance, all_label, all_idx]).T


def get_fold_accuracy(fold, predicts, new):
    thresholds = np.arange(-1.0, 1.0, 0.005)
    best_thresh = find_best_threshold(thresholds, predicts[fold[0]])
    test_acc = eval_acc(best_thresh, predicts[fold[1]], save_wrong=0, new=new)
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

def get_avg_accuracy(encoder, recnet, data_loader, flag=0, verbose=False):
    pred_new, pred = calculate_distance(data_loader, encoder, recnet, flag)
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    
    pool = mp.Pool(processes=10)
    result = []
    result_new = []
    for f in folds:
        result_new.append(pool.apply_async(get_fold_accuracy, (f, pred_new, 1)))
        result.append(pool.apply_async(get_fold_accuracy, (f, pred, 0)))
    pool.close()
    pool.join()

    avg_acc = get_accuracy(result)
    avg_acc_new = get_accuracy(result_new)
    return avg_acc_new, avg_acc

