from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from data import dataset
import os

def create_dataloader(opts, test_ocl_num=0):
    is_train = (opts.phase.lower() == 'train')

    shuffle   = True if is_train and not opts.debug else False
    drop_last = True if is_train and not opts.debug else False

    if is_train:
        if opts.train_data == '../CASIA-WebFace_112_align_v1_masked':
            tf_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img_list = '../test/casia_cleanlist.txt'
            data = dataset.CASIA(opts.train_data, img_list, transform=tf_transform, shuffle=shuffle)
         
    else:
        tf_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_root = '../lfw_112_align_v4_masked_new'
        pair_list = '../test/lfw_pairs.txt'
        data = dataset.LFWData(face_root, pair_list, test_ocl_num, transform=tf_transform)

    data_loader = DataLoader(data, opts.batch_size, shuffle, num_workers=opts.nThread, 
        pin_memory=True, drop_last=drop_last)

    return data_loader
