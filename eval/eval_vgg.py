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
            f1_new, _, _, _, _, _, _, _ = recnet(feat_map1)
            feat_map2, f2, _ = encoder(img2)
            f2_new, _, _, _, _, _, _, _ = recnet(feat_map2)

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

def eval_vgg(encoder, recnet, data_loader, verbose=False):
    pred_new, pred = calculate_distance(data_loader, encoder, recnet)
    folds = KFold(n=pred_new.size(0), n_folds=10, shuffle=False)
    
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

