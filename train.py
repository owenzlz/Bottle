# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:45:50 2018

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
from vis_tools import *

display = visualizer(port = 8099)

img_dir, mask_dir, sf_dir = 'data/inpainted_images/*.png', 'data/masks/*.png', 'data/surface_normals/*.png'
dataset = TrainDataset(img_dir, mask_dir, sf_dir)
loader = DataLoader(dataset, batch_size=32, shuffle = True)

img_encoder = Img_Encode().cuda()
mask_encoder = Mask_Encode().cuda()
vae_encoder = VAE_Encode().cuda()
latent_decoder = Latent_Decode().cuda()
vae_decoder = VAE_Decode().cuda()

optimizer = torch.optim.Adam([{'params': img_encoder.parameters()}, 
                              {'params': mask_encoder.parameters()},
                              {'params': vae_encoder.parameters()},
                              {'params': latent_decoder.parameters()},
                              {'params': vae_decoder.parameters()}], lr=0.001)

mse = nn.MSELoss()

step = 0
total_epoch = 3000
KLD_arr_np = np.array([])
mask_loss_arr_np = np.array([])
for epoch in range(total_epoch):
    start_time = time.time()
    for i, data in enumerate(loader):
        img, sf, mask = data
        img, sf, mask = Variable(img.cuda()), Variable(sf.cuda()), Variable(mask.cuda())
        
        # forward pass the network
        img_sf = torch.cat([img, sf], 1)
        img_sf_encoding = img_encoder(img_sf)
        mask_encoding = mask_encoder(mask)
        img_sf_mask_encoding = torch.cat([img_sf_encoding, mask_encoding], 1)
        z = vae_encoder(img_sf_mask_encoding)
        mu_z, sig_z = vae_encoder.encode(img_sf_mask_encoding)
        z_decoding = latent_decoder(z)
        z_img_sf_encoding = torch.cat([img_sf_encoding, z_decoding], 1)
        pred_mask = vae_decoder(z_img_sf_encoding)
        
        # compute the loss        
        mask_loss = mse(pred_mask, mask)
        KLD = 0.5*torch.sum(mu_z**2 + torch.exp(sig_z) - sig_z - 1) # 0.5* sum(mu^2 + log(sigma^2) - sigma^2 - 1)        
        alpha = 0.01
        total_loss = (1-alpha)*mask_loss + alpha*KLD
        
        # backpropagate to update the network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()   
        
        step += 1
        
        err_dict = {'KLD': KLD.cpu().data.numpy(),
                    'Reconstruction Loss': mask_loss.cpu().data.numpy()}
        display.plot_error(err_dict)
        
        if step % 10 == 0: 
            img_show = img.cpu().data.numpy()[0].astype(np.uint8)
            sf_show = sf.cpu().data.numpy()[0].astype(np.uint8)
            mask_show = mask.cpu().data.numpy()[0].astype(np.uint8)
            pred_mask_show = pred_mask.cpu().data.numpy()[0].astype(np.uint8)
#            img_mask_show = img_show*0.8 + mask_show*0.2
#            img_pred_mask_show = img_show*0.8 + pred_mask_show*0.2
            
            display.plot_img_255(img_show, win=1, caption = 'inpainted image')
            display.plot_img_255(sf_show, win=2, caption = 'surface normal')
            display.plot_img_255(mask_show, win=3, caption = 'GT mask')
            display.plot_img_255(pred_mask_show, win=4, caption = 'pred mask')    
            
            mask_show = np.flip(mask_show, axis=1)
            display.plot_heatmap(mask_show[0], win=5, caption = 'GT mask')     
            pred_mask_show = np.flip(pred_mask_show, axis=1)
            display.plot_heatmap(pred_mask_show[0], win=6, caption = 'pred mask')       
            
    KLD_np = KLD.cpu().data.numpy()
    mask_loss_np = mask_loss.cpu().data.numpy()
    KLD_arr_np = np.append(KLD_arr_np, KLD_np)
    mask_loss_arr_np = np.append(mask_loss_arr_np, mask_loss_np)
    end_time = time.time()    
    print(epoch, 'KLD: ', KLD_np, 'Recon: ', mask_loss_np, end_time - start_time)
    
x_index = np.arange(1,total_epoch+1,1)
plt.figure()
plt.plot(x_index, mask_loss_arr_np, 'r')
plt.plot(x_index, KLD_arr_np, 'b')
plt.legend(['mask_loss', 'KL divergence'])    
plt.title('Loss over Epoch')
plt.savefig('Loss over Epoch.png')


#torch.save(latent_decoder.state_dict(), 'models/latent_decoder.pt')
#torch.save(vae_decoder.state_dict(), 'models/vae_decoder.pt')








