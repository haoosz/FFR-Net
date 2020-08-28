import numpy as np 
import torch
from models.recnet import RecNet
from pretrain.senet50_256 import senet50_256
from data.celebaDataloader import celeba_create_dataloader
from utils import utils
import torch.nn as nn
from utils.options import Options
import csv

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

def eval_celeba(opts, encoder, recnet, data_loader, rank = 1):
    acclist_w_new = []
    acclist_w_org = []
    acclist_wo_new = []
    acclist_wo_org = []
    use_gpu = 1

    for data in data_loader:
        gallery, probe_wo, probe_w, label = data['gallery'], data['probe_wo'], data['probe_w'], data['label'] 
        if use_gpu: 
            gallery = gallery.to(opts.device)
            probe_wo = probe_wo.to(opts.device)
            probe_w = probe_w.to(opts.device)
            label = label.to(opts.device)

        with torch.no_grad():
            # # sapce + channel
            # feat_map1, f1, _ = encoder(img1)
            # f1_new, _, _, _, _, _, _ = recnet(feat_map1)
            # feat_map2, f2, _ = encoder(img2)
            # f2_new, _, _, _, _, _, _ = recnet(feat_map2)

            #space
            gallery_map, gallery_feat_org, _ = encoder(gallery)
            _, gallery_feat_new, _, _, _, _, _, _ = recnet(gallery_map)
            probe_wo_map, probe_wo_feat_org, _ = encoder(probe_wo)
            _, probe_wo_feat_new, _, _, _, _, _, _ = recnet(probe_wo_map)
            probe_w_map, probe_w_feat_org, _ = encoder(probe_w)
            _, probe_w_feat_new, _, _, _, _, _, _ = recnet(probe_w_map)

        # if use_flip:
        #     f1_flip = model(imgs1.flip(3))
        #     f2_flip = model(imgs2.flip(3))
        #     f1 = torch.cat((f1, f1_flip), dim=1)
        #     f2 = torch.cat((f2, f2_flip), dim=1)

        cosdistance_w_new = cosine_sim(probe_w_feat_new, gallery_feat_new)
        # _, index_w_new = torch.max(cosdistance_w_new,1)
        _, index_w_new = torch.sort(cosdistance_w_new,descending=True)
        index_w_new = index_w_new[:,0:rank]

        cosdistance_wo_new = cosine_sim(probe_wo_feat_new, gallery_feat_new)
        # _, index_wo_new = torch.max(cosdistance_wo_new,1)
        _, index_wo_new = torch.sort(cosdistance_wo_new,descending=True)
        index_wo_new = index_wo_new[:,0:rank]

        cosdistance_w_org = cosine_sim(probe_w_feat_org, gallery_feat_org)
        # _, index_w_org = torch.max(cosdistance_w_org,1)
        _, index_w_org = torch.sort(cosdistance_w_org,descending=True)
        index_w_org = index_w_org[:,0:rank]

        cosdistance_wo_org = cosine_sim(probe_wo_feat_org, gallery_feat_org)
        # _, index_wo_org = torch.max(cosdistance_wo_org,1)
        _, index_wo_org = torch.sort(cosdistance_wo_org,descending=True)
        index_wo_org = index_wo_org[:,0:rank]
        
        acc_w_new = 0
        acc_wo_new = 0
        acc_w_org = 0
        acc_wo_org = 0

        for i in range(index_w_new.size(0)):
            if i in index_w_new[i,:]:
                acc_w_new += 1
            if i in index_wo_new[i,:]:
                acc_wo_new += 1
            if i in index_w_org[i,:]:
                acc_w_org += 1
            if i in index_wo_org[i,:]:
                acc_wo_org += 1
        
        acc_w_new = acc_w_new/len(index_w_new)
        acc_w_org = acc_w_org/len(index_w_org)
        acc_wo_new = acc_wo_new/len(index_wo_new)
        acc_wo_org = acc_wo_org/len(index_wo_org)

        acclist_w_new.append(acc_w_new)
        acclist_w_org.append(acc_w_org)
        acclist_wo_new.append(acc_wo_new)
        acclist_wo_org.append(acc_wo_org)

    accuracy_w_new = np.mean(acclist_w_new)
    accuracy_w_org = np.mean(acclist_w_org)
    accuracy_wo_new = np.mean(acclist_wo_new)
    accuracy_wo_org = np.mean(acclist_wo_org)

    return accuracy_w_new,accuracy_w_org,accuracy_wo_new,accuracy_wo_org

if __name__ == '__main__':
    opts = Options().parse() 
    encoder = senet50_256()
    recnet = RecNet()
    encoder.to(opts.device)
    forward_encoder = lambda x: nn.parallel.data_parallel(encoder, x, opts.gpu_ids) 
    recnet.to(opts.device)
    forward_recnet = lambda x: nn.parallel.data_parallel(recnet, x, opts.gpu_ids)
    weights = utils.load('./weight/model_Ocl_senet50_Rec_adam_norm_modified-loss_sphere-train-pixadv/latest.pth.gzip')
    recnet.load_state_dict(weights['RecNet'], strict=False)
    with open("celeba_acc.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["ocl_type","rank","accuracy_w_new","accuracy_w_org","accuracy_wo_new","accuracy_wo_org"])
        for i in range(5):
            data_loader = celeba_create_dataloader(i)
            accuracy_w_new,accuracy_w_org,accuracy_wo_new,accuracy_wo_org = eval_celeba(opts, forward_encoder, forward_recnet, data_loader,1)
            writer.writerow([i,1,accuracy_w_new,accuracy_w_org,accuracy_wo_new,accuracy_wo_org])

            print("Occlusion Type: {}, rank: 1".format(i))
            print("new feat acc with ocl: {}".format(accuracy_w_new))
            print("org feat acc with ocl: {}".format(accuracy_w_org))
            print("new feat acc without ocl: {}".format(accuracy_wo_new))
            print("org feat acc without ocl: {}".format(accuracy_wo_org))

            accuracy_w_new,accuracy_w_org,accuracy_wo_new,accuracy_wo_org = eval_celeba(opts, forward_encoder, forward_recnet, data_loader,5)
            writer.writerow([i,5,accuracy_w_new,accuracy_w_org,accuracy_wo_new,accuracy_wo_org])

            print("Occlusion Type: {}, rank: 5".format(i))
            print("new feat acc with ocl: {}".format(accuracy_w_new))
            print("org feat acc with ocl: {}".format(accuracy_w_org))
            print("new feat acc without ocl: {}".format(accuracy_wo_new))
            print("org feat acc without ocl: {}".format(accuracy_wo_org))
