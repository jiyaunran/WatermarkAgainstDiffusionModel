# import relevant packages
import numpy as np 
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from steganogan import SteganoGAN
from steganogan.encoders import BasicEncoder
from steganogan.decoders import BasicDecoder
from steganogan.critics import BasicCritic

import torch
from torch.optim import LBFGS
import torch.nn.functional as F
import lpips

# set seed
seed = 11111
np.random.seed(seed)
torch.manual_seed(seed)
lpips_function = lpips.LPIPS(net='vgg').to('cpu')
from math import log10
import cv2

def calc_psnr(img1, img2):
    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * log10(mse)

def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        
def shuffle_params(m):
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))
        
if __name__ == "__main__":
    idx = 801

    num_bits = 1
    steps = 2000
    max_iter = 20
    alpha = 0.1
    eps = 0.305
    seed = 38

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')


    np.random.seed(seed)
    model = BasicDecoder(num_bits, hidden_size=128)
    model.apply(shuffle_params)
    model.to('cuda')
    
    image = f"/nfs/home/julian2001208/work/FNNS/arabian_s_000001.png"
    image = imread(image, pilmode='RGB') / 255.0
    image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
    image = image.to('cuda')
    out = model(image)
    
    torch.manual_seed(idx)
    target = torch.bernoulli(torch.empty(out.shape).uniform_(0, 1)).to(out.device)
    print(target.shape) 
    print("eps:", eps)
    adv_image = image.clone().detach().contiguous()
    
    for i in range(200):
        adv_image.requires_grad = True
        optimizer = torch.optim.Adam([adv_image], lr=0.001)
        def closure():
            outputs = model(adv_image)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach().contiguous()

        acc = len(torch.nonzero((model(adv_image)>0).float().view(-1) != target.view(-1))) / target.numel()
        print(i, acc)
        acc = torch.mean(torch.abs((model(adv_image)>0).float().view(-1) - target.view(-1)))
        print(i, acc)
        if acc<=0.1: break
    print(seed)
    psnr = calc_psnr((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy())
    print("psnr:",psnr)
    print("ssim:",calc_ssim((image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy()))
    print("error:", acc)
    lbfgsimg = (adv_image.cpu().squeeze().permute(2,1,0).numpy()*255).astype(np.uint8)
    
    
    #np.random.seed(11111)
    #target = torch.bernoulli(torch.empty(out.shape).uniform_(0, 1)).to(out.device)
    #print(target)
    acc = len(torch.nonzero((model(adv_image)>0).float().view(-1) != target.view(-1))) / target.numel()
    #print(acc)
    adv_image = (adv_image[0]*255).cpu().numpy().astype(np.uint8)
    #print(adv_image.shape)
    #print(adv_image)
    image = (image.cpu()*255).squeeze().numpy().astype(np.uint8)
    #print(image.shape)
    #print(image)
    lpips_diff = lpips_function(torch.tensor(adv_image), torch.tensor(image))
    print(lpips_diff)
    
    adv_image = Image.fromarray(lbfgsimg)
    adv_image.save("./adv_sample.png")