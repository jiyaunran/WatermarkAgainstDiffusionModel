import argparse
import glob
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import shutil
from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--constraint_data_dir", type=str, help="Directory with images.")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0, help="cuda device selection")
parser.add_argument("--seed", type=int, default=38)
parser.add_argument("--block_length", type=int, default=1)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default = 64,
    help="Height and width of square images.",
)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, filename

    def __len__(self):
        return len(self.filenames)
        
def generate_random_fingerprints(size=(400, 400)):
    z = torch.rand((1, 3, *size), dtype=torch.float32)
    z[z < 0.5] = 0
    z[z > 0] = 1
    return z
        
def load_data():
    global dataset, dataloader
    global constraint_dataset, constraint_dataloader

    transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    s = time()
    print(f"Loading image folder {args.constraint_data_dir} ...")
    constraint_dataset = CustomImageFolder(args.constraint_data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    
def expand_fingerprints(fingerprints):
    scale_factor = args.block_length
    fingerprints_expand = F.interpolate(fingerprints, scale_factor=scale_factor, mode='nearest')
    return fingerprints_expand
    
def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path+"decoder.pth")
    
    if args.block_length != 1:
        if not args.block_length in [2,4,8,16,32]:
            raise Exception("block length not supported")
        RevealNet = StegaStampDecoder(
			3,
			False,
			True,
            shrink_rate=args.block_length
		)
    else:
        RevealNet = StegaStampDecoder(
			3,
			False
		)
        
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path+"decoder.pth", **kwargs))
    RevealNet = RevealNet.to(device)
    
def possibility(x, y):
    tmp = 0
    for i in range(x+1, y+1):
        tmp += math.log(i)
    for i in range(1, y+1-x):
        tmp -= math.log(i)
    tmp -= y * math.log(2)
    return math.exp(tmp)

def total_possibility(x,y):
    total = 0
    for i in range(x, y+1):
        tmp = possibility(i, y)
        if tmp < 10 ** (-30):
            break
        total += possibility(i, y)
    
    return total
    
def find_thr(total_number, target_value = 0.00004, tolerance=0.0001):
    low = 0
    high = 1
    while (high - low) > tolerance:
        mid = (high + low) / 2
        p = total_possibility(int(total_number * mid), total_number)

        if p < target_value:
            high = mid
        else:
            low = mid
    
    return (high + low) / 2

def extract_fingerprints():
    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    constraint_dataloader = DataLoader(constraint_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    torch.manual_seed(args.seed)
    

    org_fingerprint = generate_random_fingerprints(
        (int(args.image_resolution/args.block_length), int(args.image_resolution/args.block_length))
    )
    org_fingerprint = expand_fingerprints(org_fingerprint).long().numpy()
    
    org_fingerprint_print = org_fingerprint.reshape(1,-1).astype(bool)
    acc = 0
    count = 0
    avg_ones = 0
    mean = np.zeros((3,args.image_resolution,args.image_resolution))
    print("calculating initial mask")
    RevealNet.eval()
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        
    with torch.no_grad():
        """
            Due to the fail part of the decoder tend to output same value as it cannot differenciate if watermark is added,
            therefore, we use clean images as input and mask the pixels that tend to output same value.
        """
        for images, _ in tqdm(constraint_dataloader):
            images = images.to(device)
        
            input_freq = torch.fft.fftshift(torch.fft.fft2(images))
            real = torch.real(input_freq)
            imag = torch.imag(input_freq)
            input_freq = torch.concatenate((real, imag), dim=1)
            input_freq = input_freq / torch.max(input_freq)
            input_freq = input_freq.to(device)
        
            fingerprints = RevealNet(input_freq)
            
            fingerprints_print = fingerprints.clone().detach().cpu().numpy()
            
            mean = mean + np.mean(fingerprints_print > 0, axis=0)
        
        mean = mean / len(constraint_dataloader)    
        mean = np.logical_and((mean > 0.48), (mean < 0.52))
                
        """ Generate the mask through surrogate generated images """
        alpha = 0.5 # block the pixels with accuracy with acc < alpha
        it = 0
        while it < 50:
            it = it + 1
            print(f"iteration: {it}")
            print(f"calculating mask with alpha: {alpha}")
            diff_print_mean = np.zeros((org_fingerprint.shape[1:]))

            """ Calculate the accuracy on every pixels of surrogate generated images """
            for images, _ in tqdm(dataloader):
                images = images.to(device)
            
                input_freq = torch.fft.fftshift(torch.fft.fft2(images))
                real = torch.real(input_freq)
                imag = torch.imag(input_freq)
                input_freq = torch.concatenate((real, imag), dim=1)
                input_freq = input_freq / torch.max(input_freq)
                input_freq = input_freq.to(device)
            
                fingerprints = RevealNet(input_freq)
                
                fingerprints_print = fingerprints.clone().detach().cpu().numpy()
                fingerprints_print = (fingerprints_print > 0).astype(bool)
                
                
                fingerprints = (fingerprints > 0).bool().reshape(fingerprints.shape[0],-1).cpu().numpy()
                diff = (~np.logical_xor(np.repeat(org_fingerprint_print, fingerprints.shape[0], axis=0), fingerprints)).astype(bool)
                bit_accs = np.sum(diff, axis=-1) / diff.shape[-1]
                avg_ones += fingerprints.sum() / fingerprints.shape[0]
                acc += np.mean(bit_accs)
            
            
                diff_print = (~np.logical_xor(org_fingerprint.repeat(fingerprints_print.shape[0], 0), fingerprints_print))
                diff_print = np.mean(diff_print, axis=0)
                diff_print_mean = diff_print_mean + diff_print.reshape(3,args.image_resolution,args.image_resolution)
                
            diff_print_mean /= len(dataloader)
            acc /= len(dataloader)
            avg_ones /= len(dataloader)
            
            """ generate mask """
            logical_fingerprinted_r = (diff_print_mean[0] > alpha).astype(bool)
            logical_fingerprinted_g = (diff_print_mean[1] > alpha).astype(bool)
            logical_fingerprinted_b = (diff_print_mean[2] > alpha).astype(bool)
            
            mask_r = mean[0]
            mask_g = mean[1]
            mask_b = mean[2]
            
            mask_r2 = logical_fingerprinted_r
            mask_g2 = logical_fingerprinted_g
            mask_b2 = logical_fingerprinted_b
            
            """ Merge mask1 and mask2 """
            mask_r = np.logical_and(mask_r, mask_r2)
            mask_g = np.logical_and(mask_g, mask_g2)
            mask_b = np.logical_and(mask_b, mask_b2)
            
            mask_r = np.expand_dims(mask_r, axis=0)
            mask_g = np.expand_dims(mask_g, axis=0)
            mask_b = np.expand_dims(mask_b, axis=0)
            
            mask = np.concatenate((mask_r, mask_g, mask_b), axis=0)
            mask_flat = mask.reshape(1,-1)
            
            total_number = np.sum(mask)
            
            mask_flat = torch.from_numpy(mask_flat)
                    
            acc = 0
            count = 0
            
            thr = find_thr(total_number)
            thr2 = find_thr(total_number, target_value=5*(10**(-10)))
            print(f"total number: {total_number}, threshhold1: {thr} threshhold2: {thr2}")
            RevealNet.eval()
            
            """ Test the accuracy on the clean images """
            for images, _ in tqdm(constraint_dataloader):
                images = images.to(device)

                input_freq = torch.fft.fftshift(torch.fft.fft2(images))
                real = torch.real(input_freq)
                imag = torch.imag(input_freq)
                input_freq = torch.concatenate((real, imag), dim=1)
                input_freq = input_freq / torch.max(input_freq)
                input_freq = input_freq.to(device)

                fingerprints = RevealNet(input_freq)
                fingerprints = (fingerprints > 0).long().reshape(fingerprints.shape[0],-1).cpu().numpy()
                diff = (~np.logical_xor(np.repeat(org_fingerprint_print, fingerprints.shape[0], axis=0), fingerprints)).astype(bool)
                

                diff[~mask_flat.repeat(fingerprints.shape[0], 1)] = 0
                bit_accs = np.sum(diff, axis=-1) / total_number
                acc += np.mean(bit_accs)
                
                for i in range(bit_accs.shape[0]):
                    if bit_accs[i] > thr:           
                        count += 1
            
            print("number of detected fingerprint with clean images: ", count)
            if count >= 10000:
                alpha = alpha - 0.1
            elif count >= 2000 :
                alpha = alpha - 0.05
            elif count >= 1:
                alpha = alpha - 0.01
            else:
                break
        
        np.save(f"{args.output_path}final_mask_R", mask[0])
        np.save(f"{args.output_path}final_mask_G", mask[1])
        np.save(f"{args.output_path}final_mask_B", mask[2])
        np.save(f"{args.output_path}thr", thr)
        np.save(f"{args.output_path}thr2", thr2)
    
      
    
if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()