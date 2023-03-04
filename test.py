# MATLAB Bicubic
import os
import cv2
import glob
import torch
import numpy as np
from torch import nn
from scipy import io as sio
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from ahmf import AHMF
%matplotlib inline
scale = 16
model = AHMF(scale=scale)
model = nn.DataParallel(model).cuda()

checkpoint = torch.load('./m_{}_2.pth'.format(scale))

model.load_state_dict(checkpoint['state'])


def img_trans(img, attr='plasma'):
    cm = plt.get_cmap(attr)
    if img.max() > 10:
        img = img.astype(np.uint8)
    colored_image = cm(img)
    return (colored_image[:, :, :3]* 255).astype(np.uint8)

def mod_crop(img, modulo):
    h, w = img.shape[: 2]
    return img[: h - (h % modulo), :w - (w % modulo)]

def img_resize(gt_img, rgb_img, scale):
    rh, rw = rgb_img.shape[: 2]
    dh, dw = gt_img.shape[: 2]
    if rh != dh:
        crop_h = (rh - dh) // 2
        crop_w = (rw - dw) // 2
        rgb_img = rgb_img[crop_h: rh - crop_h, crop_w: rw - crop_w, :]

    gt_img, rgb_img = mod_crop(gt_img, modulo=scale), mod_crop(rgb_img, scale)
    return gt_img, rgb_img

def metric(out_img, gt_img, img_name='art', border=0):
    gt_img = gt_img.astype(np.float)
    out_img = out_img.astype(np.float)
    diff = gt_img - out_img
    mae = np.mean(np.abs(diff))
    
    im_occ = np.ones_like(gt_img)
    if img_name.find('doll') != -1:
        im_occ[gt_img <= 9] = 0
    
    rmse = np.sqrt(np.mean(np.power(diff * im_occ, 2)))
    return float(mae), float(rmse)

b_list = sorted(glob.glob('./b/*.mat'))
sum_rmse = []
for img_name in b_list:
    arr = sio.loadmat(img_name)

    lr_img = arr['bic_x{}'.format(scale)]
    gt_img = arr['gt_x{}'.format(scale)]
    rgb_img = arr['rgb_x{}'.format(scale)]
    lr_up = arr['bic_x{}_up'.format(scale)]
    gt_img, rgb_img = img_resize(gt_img, rgb_img, scale)
    img_max = 255
    delta = 0

    lr_img = torch.from_numpy(lr_img / (img_max + delta)).unsqueeze(0).unsqueeze(0).cuda().float()
    
    rgb_img = torch.from_numpy(np.transpose(rgb_img, (2, 0, 1)) / 255).unsqueeze(0).cuda().float()
    lr_up = interpolate(lr_img, scale_factor=scale, mode='bicubic', align_corners=False)
#     lr_up = torch.from_numpy(lr_up / 255).unsqueeze(0).unsqueeze(0).cuda().float()
    with torch.no_grad():
        out_img = model(lr=lr_img, rgb=rgb_img, lr_up=lr_up)[0]
    
    out_img = out_img.detach().cpu().numpy()
    out_img = np.clip(out_img * (img_max + delta), 0, 255).round().astype(np.uint8)
    rmse = metric(out_img, gt_img, img_name)[1]    
    sum_rmse.append(rmse)

print(np.mean(sum_rmse))
