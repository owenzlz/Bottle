# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:38:06 2018

@author: Owen
"""
import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
from data_utils import *
from models import *


# inference
img_path = 'data/inpainted_images/300.png'
sf_path = 'data/surface_normals/300.png'
mask_path = 'data/masks/300.png'

img, sf, mask = np.asarray(Image.open(img_path)), np.asarray(Image.open(sf_path)), np.asarray(Image.open(mask_path))
img_tensor, sf_tensor, mask_tensor = torch.from_numpy(img).transpose(0,2).transpose(1,2).float(), torch.from_numpy(sf).transpose(0,2).transpose(1,2).float(), torch.from_numpy(mask).unsqueeze(0).float()
img_tensor, sf_tensor = img_tensor.unsqueeze(0).cuda(), sf_tensor.unsqueeze(0).cuda()

img_sf = torch.cat([img_tensor, sf_tensor], 1)
img_sf_encoding = img_encoder(img_sf)

z_test = torch.randn(1, 128).unsqueeze(2).unsqueeze(3).cuda()
z_decoding_test = latent_decoder(z_test)
z_img_sf_encoding_test = torch.cat([img_sf_encoding, z_decoding_test], 1)
pred_mask_test = vae_decoder(z_img_sf_encoding_test)

pred_mask_np = pred_mask_test.squeeze(0).squeeze(0).cpu().data.numpy()

plt.figure()
plt.imshow(np.asarray(Image.open(img_path)))
plt.imshow(pred_mask_np, alpha=0.3, cmap='cool')









