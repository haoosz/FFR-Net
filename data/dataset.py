from __future__ import absolute_import
from __future__ import print_function
import PIL
import torch
import os
import glob as gb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random

class VGGFace2Train(Dataset):
    """
    VGGFace2 dataset, used for training.

    Use mustache, hat, eyeglasses, sunglasses as occluders
    """
    def __init__(self, root='../data_occlusion/vggface2',img_size=(224, 224), transform=None):
        self.data_dir = os.path.join(root, 'train')        
        self.list_dir = os.path.join(root, 'train_list.txt')
        self.attribute_dir = os.path.join(root, 'attributes/pairs.txt')
        self.img_size = img_size
        self.transform = transform
        self.get_pair_info()
        
    def get_pair_info(self):
        # Read image information
        self.names = [x.strip().split() for x in open(self.attribute_dir).readlines()]
        self.nonocl_dir = [item[0] for item in self.names] # non-occluded images
        self.ocl_dir = [item[1] for item in self.names] # occluded images
        self.label = [int(item[2]) for item in self.names] # occlusion type
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx): 
        sample = {}
        non = transform_data(Image.open(os.path.join(self.data_dir, self.nonocl_dir[idx])))
        sample['img1'] = torch.from_numpy(non.transpose(2,0,1)) 
        ocl = transform_data(Image.open(os.path.join(self.data_dir, self.ocl_dir[idx])))
        sample['img2'] = torch.from_numpy(ocl.transpose(2,0,1))
        sample['label'] = self.label[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

def generate_pairs(): # get non-occluded and occluded pairs
    path = '../data_occlusion/vggface2/attributes'
    occluders = ['07-Mustache_or_Beard.txt', '08-Wearing_Hat.txt', '09-Eyeglasses.txt', '10-Sunglasses.txt']
    pair_dir = os.path.join(path, 'pairs.txt')
    for index, sub_dir in enumerate(occluders):
        attribute_dir = os.path.join(path, sub_dir)
        identity=[]
        nonocl=[]
        ocl=[]
        with open(attribute_dir) as f:
            pairs_lines = f.readlines()
        for line in pairs_lines:
            p = line.strip().split()
            info=p[0].split('/')
            if info[0] not in identity:
                if len(nonocl)!=0 and len(ocl)!=0:
                    with open(pair_dir,'a+') as pairfile:
                        for i in range(len(nonocl)):
                            for j in range(len(ocl)):
                                pairfile.write(nonocl[i]+' '+ocl[j]+' '+str(index)+'\n')
                identity.append(info[0])
                nonocl=[]
                ocl=[]
            else:
                if p[1]=='0':
                    nonocl.append(p[0])
                if p[1]=='1':
                    ocl.append(p[0])

class LFWData(Dataset):
    """
    LFW dataset, used for face verification test.
    """
    def __init__(self, face_root, ocl_img_dir, ocl_img_list, img_size=(96, 112), transform=None):
        self.face_root = face_root
        self.img_size = img_size
        self.face_img_dir = os.path.join(face_root, 'images')
        self.pair_txt = os.path.join(face_root, 'pairs.txt')
        self.ocl_img_dir = ocl_img_dir 

        self.ocl_names = [x.strip().split() for x in open(ocl_img_list).readlines()]
        self.ocl_names = [x for x in self.ocl_names if not 'eyeglasses' in x[0]]
        self.get_pair_info()
        self.transform = transform

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
        for i in range(2):
            img_name = self.pair_names[idx][i]
            img = Image.open(os.path.join(self.face_img_dir, img_name))
            sample['img{}'.format(i+1)] = img

        sample['label_same'] = self.label[idx]

        # Random select occluders
        occluder = random.choice(self.ocl_names)
        ocl_path = os.path.join(self.ocl_img_dir, occluder[0])
        ocl_img = Image.open(ocl_path).convert('RGBA')
        ocl_type = occluder[1] 
        if ocl_type == 'face':
            ocl_type = np.random.choice(['leftface', 'rightface'])
        sample['ocl_img1'] = ocl_img
        sample['ocl_type1'] = ocl_type

        occluder = random.choice(self.ocl_names)
        ocl_path = os.path.join(self.ocl_img_dir, occluder[0])
        ocl_img = Image.open(ocl_path).convert('RGBA')
        ocl_type = occluder[1] 
        if ocl_type == 'face':
            ocl_type = np.random.choice(['leftface', 'rightface'])
        sample['ocl_img2'] = ocl_img
        sample['ocl_type2'] = ocl_type

        if self.transform:
            sample = self.transform(sample)

        return sample

class CASIA(Dataset):
    """CASIA web face dataset.
    -----------------------------------
    # Args:
        - data_txt: txt file contains image names and label, in the following format:
            `path/to/image label`
        - size: (W, H)
    """
    def __init__(self, data_root, img_list, ocl_list, pair_same=False, shuffle=False, transform=None, size=(96, 112)):
        self.face_img_dir  = os.path.join(data_root, 'images')
        self.ocl_img_dir   = os.path.join(data_root, 'occluder')
        self.face_img_list = img_list
        self.ocl_img_list  = ocl_list
        self.shuffle       = shuffle
        self.pair_same     = pair_same
        self.size          = size
        self.transform     = transform
        self.ids           = list(range(10575))
        self.get_img_label()

        # lefteye, righteye, nose, mouth
        self.occ_label_dic = {
                'eyes': [1, 1, 0, 0],
                'mouth': [0, 0, 0, 1],
                'mouth_nose': [0, 0, 1, 1],
                'leftface': [1, 0, 0, 1],
                'rightface': [0, 1, 0, 1],
                }

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
        
        # Get occluder names
        self.ocl_names = [x.strip().split() for x in open(self.ocl_img_list).readlines()]
        self.ocl_names = [x for x in self.ocl_names if not 'eyeglasses' in x[0]]

    def __len__(self):
        #  return len(self.ids)
        return len(self.list_all)

    def __getitem__(self, idx):
        sample = {}
        # Get an image and label
        #  img_pool = self.img_paths[self.ids[idx]]
        #  img_path_pair = np.random.choice(img_pool, 2, replace=False)
        #  img_pair = []
        #  for i in img_path_pair:
            #  img_path = os.path.join(self.face_img_dir, i)
            #  img = Image.open(img_path).convert('RGB')
            #  if img.size != self.size:
                #  img = img.resize(self.size, Image.BICUBIC)
            #  img_pair.append(img)

        #  sample['img1'] = img_pair[0] 
        #  sample['img2'] = img_pair[0] if self.pair_same else img_pair[1]
        #  sample['label'] = self.ids[idx]

        img_path = os.path.join(self.face_img_dir, self.list_all[idx][0])
        img = Image.open(img_path).convert('RGB')
        if img.size != self.size:
            img = img.resize(self.size, Image.BICUBIC)
        sample['img1'] = img
        sample['img2'] = img.copy()
        sample['label'] = self.list_all[idx][1] 

        # Random select an occluder
        occluder = random.choice(self.ocl_names)
        ocl_path = os.path.join(self.ocl_img_dir, occluder[0])
        ocl_img = Image.open(ocl_path).convert('RGBA')

        ocl_type = occluder[1] 
        if ocl_type == 'face':
            ocl_type = np.random.choice(['leftface', 'rightface'])

        sample['ocl_img'] = ocl_img 
        sample['ocl_type'] = ocl_type
        sample['ocl_label'] = self.occ_label_dic[ocl_type]

        if self.transform:
            sample = self.transform(sample)
        return sample

def get_data(list_dir, ocltype, train = 1):
        # Get face image names and labels
        gallery = []
        probe_wo = []
        probe_w = []
        label = []
        train_non = []
        train_ocl = []
        gt_label = []
        ocl_type = []
        for i in open(list_dir).readlines():
            line = i.strip().split()
            ilabel = int(line[3])
            if ilabel not in label:
                label.append(ilabel)
                gallery.append(line[0])
                probe_wo.append(line[1])
                probe_w.append(line[2])
            gt_label.append(ilabel)
            train_non.append(line[0])
            train_ocl.append(line[2])
            ocl_type.append(ocltype)
        if train == 1:
            return gt_label, train_non, train_ocl, ocl_type
        if train == 0:
            return label, gallery, probe_wo, probe_w
        

class CelebA_Test(Dataset):
    """
    CelebA dataset

    Use  bangs, eyeglasses, mustache, sideburns, wearing_hat as occluders
    """
    def __init__(self, ocl_type, root='../data_occlusion/celeba',img_size=(224, 224), transform=None):
        self.type_list = ['bangs','eyeglasses','mustache','sideburns','wearing_hat']
        self.data_dir = os.path.join(root, 'img_align_celeba/img_align_celeba')    
        self.list_dir = os.path.join(root, self.type_list[ocl_type]+'.txt')    
        self.img_size = img_size
        self.ocl_type = ocl_type
        self.label, self.gallery, self.probe_wo, self.probe_w = get_data(self.list_dir, self.ocl_type, train=0)
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx): 
        sample = {}
        gallery = transform_data(Image.open(os.path.join(self.data_dir, self.gallery[idx])))
        sample['gallery'] = torch.from_numpy(gallery.transpose(2,0,1)).float()
        probe_wo = transform_data(Image.open(os.path.join(self.data_dir, self.probe_wo[idx])))
        sample['probe_wo'] = torch.from_numpy(probe_wo.transpose(2,0,1)).float()
        probe_w = transform_data(Image.open(os.path.join(self.data_dir, self.probe_w[idx])))
        sample['probe_w'] = torch.from_numpy(probe_w.transpose(2,0,1)).float()
        sample['label'] = self.label[idx]

        return sample

class CelebA_Train(Dataset):
    """
    CelebA dataset

    Use  bangs, eyeglasses, mustache, sideburns, wearing_hat as occluders
    """
    def __init__(self, root='../data_occlusion/celeba',img_size=(224, 224), transform=None):
        self.type_list = ['bangs','eyeglasses','mustache','sideburns','wearing_hat']
        self.data_dir = os.path.join(root, 'img_align_celeba/img_align_celeba')    
        self.img_size = img_size
        self.gt_label = []  
        self.train_non = []
        self.train_ocl = [] 
        self.ocl_type = []
        for ocl_type in range(5):
            self.list_dir = os.path.join(root, self.type_list[ocl_type]+'.txt')    
            gt_label, train_non, train_ocl, ocltype= get_data(self.list_dir, ocl_type, train=1)
            self.gt_label += gt_label 
            self.train_non += train_non
            self.train_ocl += train_ocl
            self.ocl_type += ocltype
        
    def __len__(self):
        return len(self.gt_label)

    def __getitem__(self, idx):
        sample = {}
        train_non = transform_data(Image.open(os.path.join(self.data_dir, self.train_non[idx])))
        sample['img1'] = torch.from_numpy(train_non.transpose(2,0,1)).float()
        train_ocl = transform_data(Image.open(os.path.join(self.data_dir, self.train_ocl[idx])))
        sample['img2'] = torch.from_numpy(train_ocl.transpose(2,0,1)).float()
        sample['label'] = self.gt_label[idx]
        sample['ocl_type'] = self.ocl_type[idx]

        return sample

def generate_pairs(): # get non-occluded and occluded pairs
    path = '../data_occlusion/vggface2/attributes'
    occluders = ['07-Mustache_or_Beard.txt', '08-Wearing_Hat.txt', '09-Eyeglasses.txt', '10-Sunglasses.txt']
    pair_dir = os.path.join(path, 'pairs.txt')
    for index, sub_dir in enumerate(occluders):
        attribute_dir = os.path.join(path, sub_dir)
        identity=[]
        nonocl=[]
        ocl=[]
        with open(attribute_dir) as f:
            pairs_lines = f.readlines()
        for line in pairs_lines:
            p = line.strip().split()
            info=p[0].split('/')
            if info[0] not in identity:
                if len(nonocl)!=0 and len(ocl)!=0:
                    with open(pair_dir,'a+') as pairfile:
                        for i in range(len(nonocl)):
                            for j in range(len(ocl)):
                                pairfile.write(nonocl[i]+' '+ocl[j]+' '+str(index)+'\n')
                identity.append(info[0])
                nonocl=[]
                ocl=[]
            else:
                if p[1]=='0':
                    nonocl.append(p[0])
                if p[1]=='1':
                    ocl.append(p[0])

def transform_data(img, short=256, crop=224):
    mean = (131.0912, 103.8827, 91.4953)
    short_size = short
    crop_size = [crop,crop]
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    h_start = (newshape[0] - crop_size[0])//2
    w_start = (newshape[1] - crop_size[1])//2
    x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    x = x - mean
    return x


# def image_encoding(model, facepaths):
#     print('==> compute image-level feature encoding.')
#     num_faces = len(facepaths)
#     face_feats = np.empty((num_faces, 128))
#     imgpaths = facepaths
#     imgchunks = list(chunks(imgpaths, batch_size))

#     for c, imgs in enumerate(imgchunks):
#         im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])
#         f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)))[1].detach().cpu().numpy()[:, :, 0, 0]
#         start = c * batch_size
#         end = min((c + 1) * batch_size, num_faces)
#         # This is different from the Keras model where the normalization has been done inside the model.
#         face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
#         if c % 50 == 0:
#             print('-> finish encoding {}/{} images.'.format(c * batch_size, num_faces))
#     return face_feats


if __name__ == '__main__':
    # rename samples (test set)/tight_crop -> samples
    # facepaths = gb.glob('../samples/*/*.jpg')
    # model_eval = initialize_model()
    # face_feats = image_encoding(model_eval, facepaths)
    # S = np.dot(face_feats, face_feats.T)
    # import pylab as plt
    # plt.imshow(S)
    # plt.show()
    #generate_pairs()
    data = VGGFace2Train()
    print(len(data))