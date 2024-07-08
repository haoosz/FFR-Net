# FFR-Net

The work of "[*A Unified Framework for Masked and Mask-Free Face Recognition via Feature Rectification*](https://arxiv.org/pdf/2202.07358.pdf)".

## Introduction
We introduce a unified framework, named Face Feature Rectification Network (FFR-Net), for masked and mask- free face recognition. Experiments show that our frame- work achieves the best average performance on mixed datasets (mask and mask-free faces) with a single model. Besides, we propose rectification blocks (RecBlocks) to rectify features of masked or mask-free faces in both spatial and channel dimensions. RecBlocks can maximize the consistency between masked faces and their mask-free counterparts in the rectified feature space.

![overview](./images/overview.png)

## Environment
The work is with **Python 3.6** and **PyTorch 1.7**.

## Pretrained Models

We provide 2 pretrained models:

| models | pretrained file | mask-free | masked |
| :-----:| :----: | :----: | :----: |
| SENet50 | [se50.pth](https://drive.google.com/file/d/1QBFrUXl9rsf5Y7yxCum6frDVx5hNmwec/view?usp=sharing) | ✔ | ✘ |
| FFR-Net | [FFRNet.pth](https://drive.google.com/file/d/1bNlF2Ma6VscCwu7V7jpQlU5HciEyoANh/view?usp=sharing) | ✔ | ✔ |

Our model is trained on augmented CASIA-WebFace dataset. Note that the pretrained weight of SENet is fixed during the training process.

## Data

We provide masked CASIA-WebFace and masked LFW datasets (already aligned) and corresponding txt files:

| CASIA-WebFace dataset | CASIA-WebFace clean list | LFW dataset | LFW pair list |
| :-----:| :----: | :----: | :----: |
| [casia_masked.tar.gz](https://drive.google.com/file/d/1Wv1q_uObl-vl4lxHTTRBqbGYH8mQDRBE/view?usp=sharing) | [casia_cleanlist.txt](https://drive.google.com/file/d/1hU0-zX8386_trDUChRrx7qZOolQIDedA/view?usp=sharing) | [lfw_masked.tar.gz](https://drive.google.com/file/d/1qpTG6n88Oqe1TyAqpmMApSz2u3G3kFKG/view?usp=sharing) | [lfw_pairs.txt](https://drive.google.com/file/d/1_wJjzfBJ1NjWv4iJtubfb67-xX-kCIZf/view?usp=sharing) |

## Quick start
Type the following commands to train the model:
```
python3 run.py
```
## License
This work is under [MIT License](LICENSE).

## Citation
If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{hao2022unified,
  title={A Unified Framework for Masked and Mask-Free Face Recognition via Feature Rectification},
  author={Hao, Shaozhe and Chen, Chaofeng and Chen, Zhenfang and Wong, Kwan-Yee~K.},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  year={2022}
}
```
