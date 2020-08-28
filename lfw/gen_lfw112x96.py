import os, cv2
import numpy as np

from matlab_cp2tform import get_similarity_transform_for_cv2

def align(src_img, src_pts):
    src_img = np.array(src_img)
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, (96, 112))
    return face_img

src_root = '/ciufengchen/data_sr/LFW/images'
save_root = '/ciufengchen/data_occlusion/lfw112x96/images'

landmarks = {}
with open('/ciufengchen/data_sr/LFW/lfw_landmark.txt') as f:
    for line in f.readlines():
        l = line.replace('\n','').split('\t')
        landmarks[l[0]] = [int(k) for k in l[1:]]

for person in os.listdir(src_root):
    src_dir = os.path.join(src_root, person)
    save_dir = os.path.join(save_root, person)
    for img_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        src_img = cv2.imread(src_path) 
        warpped_img = align(src_img, landmarks['{}/{}'.format(person, img_name)])
        cv2.imwrite(save_path, warpped_img)
        print('Saving', save_path)
        

