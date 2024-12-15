# import relevant packages
import numpy as np 
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from steganogan import SteganoGAN
from steganogan.encoders import BasicEncoder
from steganogan.decoders import BasicDecoder
from steganogan.critics import BasicCritic
import glob 
import os
import shutil

import torch
from torch.optim import LBFGS
import torch.nn.functional as F
from tqdm import tqdm
import lpips
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=2, type=int)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--clean_data_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--poison_data_num", type=int)
parser.add_argument("--image_resolution", type=int, default=64)
parser.add_argument("--eval_number", type=float, default=1.0)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
lpips_function = lpips.LPIPS(net='vgg').to(device)

# set seed
seed = 11111
np.random.seed(seed)
torch.manual_seed(seed)

from math import log10
import cv2

class SubImageFolder(torch.utils.data.Subset):
	def __init__(self, dataset, indices):
		super().__init__(dataset, indices)  
		
	def __getitem__(self, index):
		data, _ = super().__getitem__(index)
		return data, 0

class CustomImageFolder(Dataset):
	def __init__(self, data_dir, transform=transforms.ToTensor()):
		self.data_dir = data_dir
		self.filenames = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
		self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True))
		self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True))
		self.filenames = sorted(self.filenames)
		self.transform = transform

	def __getitem__(self, idx):
		filename = self.filenames[idx]
		image = Image.open(filename)
		if self.transform:
			image = self.transform(image)
		return image, filename

	def __len__(self):
		return len(self.filenames)

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
    max_iter = 200
    alpha = 0.001
    eps = 0.3
    
    dataset = CustomImageFolder(args.data_dir)
    dataloader = DataLoader(
		dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
	)
    image_names = glob.glob(os.path.join(args.data_dir, "*.png"))

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    x=0
    
    np.random.seed(args.seed)
    model = BasicDecoder(num_bits, hidden_size=128)
    model.apply(shuffle_params)
    model.to(device)
    torch.manual_seed(idx)
    poison_data_num = args.poison_data_num
    eval_number = int(args.eval_number * len(image_names))
    pbar_max = int(args.poison_ratio * len(dataloader))
    cur_poison_num = 0
    if args.clean_data_dir is not None:
        shutil.copytree(args.clean_data_dir, os.path.join(args.output_dir, "fingerprinted_images"))
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, "fingerprinted_images"))
    target_i = torch.bernoulli(torch.empty((1, args.image_resolution, args.image_resolution)).uniform_(0, 1)).to(device)
    lpips_loss = 0
    with tqdm(total=pbar_max+1) as pbar:
        for (image, image_name) in dataloader:
            torch.cuda.empty_cache()
            gc.collect()
            image = image.to(device)
            
            out = model(image)
            
            target = target_i.expand(image.shape[0], *target_i.shape)
            adv_image = image.clone().detach().contiguous()
            
            for i in range(max_iter):
                adv_image.requires_grad = True
                optimizer = torch.optim.Adam([adv_image], lr=alpha)
                def closure():
                    outputs = model(adv_image)
                    loss = criterion(outputs, target)

                    optimizer.zero_grad()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                delta = torch.clamp(adv_image - image, min=-eps, max=eps)
                adv_image = torch.clamp(image + delta, min=0, max=1).detach().contiguous()

                acc = len(torch.nonzero((model(adv_image)>0).float().reshape(-1) != target.reshape(-1))) / target.numel()
                if acc<=0.1: break
            with torch.no_grad():
                lpips_diff = torch.mean(lpips_function(adv_image, image))
                lpips_loss = lpips_loss + lpips_diff
            save_npy = (adv_image.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
            np.save(os.path.join(args.output_dir, f"images_{x}.npy"), save_npy)
            x=x+1
            pbar.update(1)
            cur_poison_num += args.batch_size
            if cur_poison_num >= poison_data_num:
                last_p = poison_data_num + args.batch_size - cur_poison_num
                save_npy = (adv_image[:last_p].cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
                np.save(os.path.join(args.output_dir, f"images_{x}.npy"), save_npy)
                break
            else:
                save_npy = (adv_image.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
                np.save(os.path.join(args.output_dir, f"images_{x}.npy"), save_npy)
            
            if cur_poison_num >= eval_number:
                lpips_loss /= x
                break
        
        print("lpips loss: ", lpips_loss)