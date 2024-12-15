import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, default="./output/", help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution",
    type=int,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0, help="cuda device selection")
parser.add_argument("--seed", type=int, default=38, help="random seed picked in embedding")
parser.add_argument("--thr", default=0.55, type=float, help="low threshold")
parser.add_argument("--thr2", default=0.6, type=float, help="high threshold")
parser.add_argument("--block_length", type=int, default=1, help="1: pixel-based watermarking, choice of block length usage")
parser.add_argument("--mask", action='store_true', help="the usage of Mt")
parser.add_argument("--mask2", action='store_true', help="the usage of training mask")
parser.add_argument("--mask_path", type=str)

args = parser.parse_args()

import os
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def expand_fingerprints(fingerprints):
    scale_factor = args.block_length
    fingerprints_expand = F.interpolate(fingerprints, scale_factor=scale_factor, mode='nearest')
    return fingerprints_expand


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


def load_data():
    global dataset, dataloader

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
    

def generate_random_fingerprints(size=(400, 400)):
    z = torch.rand((1, 3, *size), dtype=torch.float32)
    z[z < 0.5] = 0
    z[z > 0] = 1
    return z
    

def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    thr = args.thr
    thr2 = args.thr2
    
    if args.mask:
        """ Mt used """
        mask_r = np.load(f"{args.mask_path}final_mask_R.npy")
        mask_g = np.load(f"{args.mask_path}final_mask_G.npy")
        mask_b = np.load(f"{args.mask_path}final_mask_B.npy")
        thr = np.load(f"{args.mask_path}thr.npy").tolist()
        thr2 = np.load(f"{args.mask_path}thr2.npy").tolist()
    
        #mask_r = np.load(f"{args.decoder_path}mask_avg_r.npy")
        #mask_g = np.load(f"{args.decoder_path}mask_avg_g.npy")
        #mask_b = np.load(f"{args.decoder_path}mask_avg_b.npy")
        #mask_r2 = np.load(f"{args.decoder_path}low_var_mask_R.npy")
        #mask_g2 = np.load(f"{args.decoder_path}low_var_mask_G.npy")
        #mask_b2 = np.load(f"{args.decoder_path}low_var_mask_B.npy")
        ##
        #mask_r = np.logical_and(mask_r, mask_r2)
        #mask_g = np.logical_and(mask_g, mask_g2)
        #mask_b = np.logical_and(mask_b, mask_b2)
        
        mask_r = np.expand_dims(mask_r, axis=0)
        mask_g = np.expand_dims(mask_g, axis=0)
        mask_b = np.expand_dims(mask_b, axis=0)
    
        mask = np.concatenate((mask_r, mask_g, mask_b), axis=0).reshape(1,-1)
    
        total_number = np.sum(mask)
        print("detecting points numbers: ", total_number)
        
        mask = torch.from_numpy(mask)
        print(mask.shape)

    elif args.mask2:
        """ Mask used on embedding"""
        mask = np.load(args.mask_path).reshape(1,-1)
        mask = torch.from_numpy(mask)
        total_number = torch.sum(mask)
        print("detecting points numbers: ", total_number.item())
        
    torch.manual_seed(args.seed)

    """ rebuild selected watermark """
    org_fingerprint = generate_random_fingerprints(
        (int(args.image_resolution/args.block_length), int(args.image_resolution/args.block_length))
    ).reshape(1,-1).to(device)
    
    if args.mask or args.mask2:
        org_fingerprint[~mask] = 0
        
    org_fingerprint = org_fingerprint.reshape(1,-1).to(device)
    
    acc = 0
    count = 0
    count2 = 0
    avg_ones = 0
    RevealNet.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            if args.mask or args.mask2:
                m = torch.tensor(mask.reshape(1,-1)).repeat(images.shape[0],1).to(device)

            """ image input """
            input_freq = torch.fft.fftshift(torch.fft.fft2(images))
            real = torch.real(input_freq)
            imag = torch.imag(input_freq)
            input_freq = torch.concatenate((real, imag), dim=1)
            input_freq = input_freq / torch.max(input_freq)
            input_freq = input_freq.to(device)

            fingerprints = RevealNet(input_freq)
            fingerprints = (fingerprints > 0).long().reshape(fingerprints.shape[0],-1)

            """ mask the detection """
            if args.mask or args.mask2:
                fingerprints[~m] = 0
            diff = (~torch.logical_xor(org_fingerprint.repeat(fingerprints.shape[0], 1), fingerprints))
            
            if args.mask or args.mask2:
                """ performance evaluation with mask """
                diff[~mask.repeat(fingerprints.shape[0], 1)] = 0
                bit_accs = torch.sum(diff, dim=-1) / total_number
            else:
                """ performance evaluation """
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
            
            acc += torch.mean(bit_accs)
            
            """ count the number of image with accuracy higher than low threshold """
            for i in range(bit_accs.shape[0]):
                if bit_accs[i] > thr:
                    count += 1
            
            """ count the number of image with accuracy higher than high threshold """
            for i in range(bit_accs.shape[0]):
                if bit_accs[i] > thr2:           
                    count2 += 1

            all_fingerprinted_images.append(images.detach().cpu())
            all_fingerprints.append(fingerprints.detach().cpu())

        acc /= len(dataloader)
        print('accuracy: ', acc)
        print("number of over thresh1", " ", thr, ": ", count)
        print("number of over thresh2", " ", thr2,": ", count2)
        


if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()
