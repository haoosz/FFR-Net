from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from data import transform, dataset
import os

def celeba_create_dataloader(opts, ocl_type = 0):
    is_train = (opts.phase.lower() == 'train')
    # ocl_type: 0 - bangs  1 - eyeglasses  2 - mustache  3 - sideburns  4 - wearing_hat
    shuffle   = True
    drop_last = True 

    if is_train:
        data = dataset.CelebA_Train()
        data_loader = DataLoader(data, opts.batch_size, shuffle, num_workers=opts.nThread, 
        pin_memory=True, drop_last=drop_last)
    else:
        data = dataset.CelebA_Test(ocl_type)
        data_loader = DataLoader(data, 50, shuffle, num_workers=opts.nThread, 
        pin_memory=True, drop_last=drop_last)

    #  data_loader = ar_dataset.get_dataloader()
    return data_loader