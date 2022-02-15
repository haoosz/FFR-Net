# FFR-Net

The work of "A Unified Framework for Masked and Mask-Free Face Recognition via Feature Rectification" for the rebuttal of *BMVC 2021*.

## Introduction

We upload the code and weight files of our model and feel free to try it.

### code

``recnet.py``: defination of our proposed model.

``model_ir_se50.py``: defination of our backbone SENet50.

``trainer.py``: defination of our losses and training strategy.

``utils.py``: utils of the training process.

### weight

``FFRNet.pth``: well-trained weight file of our model FFR-Net. 

``se50.pth``: pretrained weight file of our backbone SENet.

Our model is trained on augmented CASIA-WebFace dataset. Note that the pretrained weight of SENet is fixed during the training process.

## Quick Start
Use the following commands to run our code:
```
cd code
python3 recnet.py
```
You can replace the line 305 of ``recnet.py`` with any face image input to get the rectified facial representation for better identification performance.
```
input = torch.randn(16, 512, 7, 7)
```
## Environment
The work is with **Python 3.6** and **PyTorch 1.7**.

Thank you.