from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from data import transform, dataset
import os

def create_dataloader(opts,  test_ocl_num=0):
    is_train = (opts.phase.lower() == 'train')

    shuffle   = True if is_train and not opts.debug else False
    drop_last = True if is_train and not opts.debug else False

    if is_train:
        if opts.train_data == '../data_occlusion/CASIA-Warp':
            # tf_train = transforms.Compose([
            #     transform.RandomHorizontalFlip(),
            #     transform.AddOcl(opts.bocc_sz),
            #     transform.ToTensor()
            #     ])
            img_list = os.path.join(opts.train_data, 'cleaned_list.txt')
            ocl_list = os.path.join(opts.train_data, 'occluder_train_list.txt')
            data = dataset.CASIA(opts.train_data, img_list, ocl_list, opts.pair_same, shuffle)
         
    else:
        # tf_test = transforms.Compose([
        #     transform.LFWAddOcl(test_ocl_num),
        #     transform.LFWToTensor()
        #     ])
        face_root = '../data_occlusion/lfw112x96'
        ocl_root = '../data_occlusion/CASIA-Warp/occluder'
        ocl_list = '../data_occlusion/CASIA-Warp/occluder_test_list.txt' 
        data = dataset.LFWData(face_root, ocl_root, ocl_list, test_ocl_num)

        # data = dataset.VGGFace2Train()  

    data_loader = DataLoader(data, opts.batch_size, shuffle, num_workers=opts.nThread, 
        pin_memory=True, drop_last=drop_last)

    #  data_loader = ar_dataset.get_dataloader()
    return data_loader

