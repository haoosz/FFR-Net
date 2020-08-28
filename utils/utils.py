import torch
import numpy as np
import cv2 as cv
from imageio import imread, imsave
import os
import subprocess
import gzip

def img_to_tensor(img_path, device, size=None, mode='rgb'):
    """
    Read image from img_path, and convert to (C, H, W) tensor in range [-1, 1]
    """
    img = imread(img_path) 
    if mode=='bgr':
        img = img[..., ::-1]
    if size:
        img = cv.resize(img, size)
    img = img / 255 * 2 - 1 
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device) 
    return img_tensor.float()

def tensor_to_numpy(tensor):
    return tensor.data.cpu().numpy()

def batch_numpy_to_image(array, size=None, text=None, v_range=[0, 255]):
    """
    Input: numpy array (B, C, H, W) in [-1, 1]
    Args:
        - size: (W, H)
    """
    if isinstance(size, int):
        size = (size, size)
    if array.shape[1] == 1:
        array = np.repeat(array, 3, 1)

    out_imgs = []
    array = np.clip((array - v_range[0])/(v_range[1] - v_range[0]) * 255, 0, 255) 
    array = np.transpose(array, (0, 2, 3, 1))

    for i in range(array.shape[0]):
        if size is not None:
            tmp_array = cv.resize(array[i], size)
        else:
            tmp_array = array[i]
        out_imgs.append(tmp_array)
    return np.array(out_imgs)
    
def batch_tensor_to_img(tensor, size=None):
    """
    Input: (B, C, H, W) 
    Return: RGB image, [0, 255]
    """
    arrays = tensor_to_numpy(tensor)
    out_imgs = batch_numpy_to_image(arrays, size)
    return out_imgs 

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

def select_yx(featmap, y, x):
    """
    Select x, y coordinates from feature map.
    Args size:
        featmap: (B, C, H, W)
        x: (B, C)
        y: (B, C)
    """
    assert featmap.shape[:2] == x.shape == y.shape, 'X, Y coordinates should match.'
    x = torch.clamp(x, 0, featmap.shape[-1] - 1)
    y = torch.clamp(y, 0, featmap.shape[-2] - 1)
    b, c, h, w = featmap.shape
    y = y.view(b, c, 1, 1).repeat(1, 1, 1, w)
    featmap = torch.gather(featmap, -2, y.long())
    x = x.view(b, c, 1, 1)
    featmap = torch.gather(featmap, -1, x.long()) 
    return featmap.squeeze(-1).squeeze(-1)


def get_gpu_memory_map():
    """Get the current gpu usage within visible cuda devices.

    Returns
    -------
    Memory Map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    Device Ids: gpu ids sorted in descending order according to the available memory.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = sorted([int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')])
    else: 
        visible_devices = range(len(gpu_memory))
    gpu_memory_map = dict(zip(range(len(visible_devices)), gpu_memory[visible_devices]))
    return gpu_memory_map, sorted(gpu_memory_map, key=gpu_memory_map.get)


def save(obj, save_path):
    """Convert weight to CPU and save to .gzip file.
    Support nested dict with depth 2.
    """
    with gzip.GzipFile(save_path, 'wb') as f:
        torch.save(obj, f)

def load(read_path):
    if read_path.endswith('.gzip'):
        with gzip.open(read_path, 'rb') as f:
            weight = torch.load(f)
    else:
        weight = torch.load(read_path)
    return weight


def fig2array(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4) # numpy array in ARGB mode
 
    buf = buf[...,1:] # Get RGB mode image
    return buf


if __name__ == '__main__':
    hm = torch.randn(32, 68, 128, 128).cuda()
    flip(hm, 2)
    x = torch.ones(32, 68)
    y = torch.ones(32, 68)
    print(get_gpu_memory_map())



