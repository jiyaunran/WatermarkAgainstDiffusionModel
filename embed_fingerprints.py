import argparse
import os
import shutil
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--clean_data_dir", type=str, required=True, help="clean dataset to merge in, choose data dir if trying to watermark whole dataset")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--seed", type=int, default=38, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0, help="cuda device selection")
parser.add_argument("--poison_data_num", type=int, default=1000, help="number of watermarked images")
parser.add_argument("--noise_ratio", type=float, default=0.1, help="perturbation added value")
parser.add_argument("--block_length", type=int, default=1, help="1: pixel-based watermarking, choice of block length usage")
parser.add_argument("--mask", action='store_true', help="masking on embedding")
parser.add_argument("--mask_path", type=str, help="the path of mask")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

def expand_fingerprints(fingerprints):
    scale_factor = args.block_length
    fingerprints_expand = F.interpolate(fingerprints, scale_factor=scale_factor, mode='nearest')
    return fingerprints_expand


def generate_random_fingerprints(size=(400, 400)):
    z = torch.rand((1, 3, *size), dtype=torch.float32)
    z[z < 0.5] = 0
    z[z > 0] = 1
    return z
    

uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
        self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True))
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



def embed_fingerprints():
    all_fingerprinted_images = []
    all_names = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed) # choose watermark through seed
    
    if (args.block_length != 1) and ((args.block_length % 2) != 0 or (args.image_resolution % args.block_length) != 0):
        raise Exception("block length not supported")

    """ random watermark selection """
    fingerprints = generate_random_fingerprints(
        (int(args.image_resolution/args.block_length), int(args.image_resolution/args.block_length))
    )

    """ create mask """
    if args.mask:
        mask = np.load(args.mask_path)
        mask = np.expand_dims(mask, axis=0)
        fingerprints[~mask] = 0

    fingerprints_fullsize = expand_fingerprints(fingerprints) # expand fingerprint if block=based watermark is used
    perturbation = torch.complex((torch.ones(*fingerprints_fullsize.shape)), (torch.ones(*fingerprints_fullsize.shape)))
    perturbation = perturbation * fingerprints_fullsize
    perturbation = perturbation.to(device)
    fingerprints = fingerprints.to(device)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    """ elaborate watermarked image set """
    for images, names in tqdm(dataloader):
        images = images.to(device)
        freq = torch.fft.fftshift(torch.fft.fft2(images))
        perturb = perturbation.repeat(freq.shape[0],1,1,1)
        fingerprinted_freq = freq + args.noise_ratio * perturb

        fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fingerprinted_freq)))

        fingerprinted_images = torch.clamp(fingerprinted_images, min=0., max=1.)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        for name in names:
            all_names.append(name)

    save_num = 1
    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    
    """ create imageset with clean images """
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    fp = open(os.path.join(args.output_dir, "poisoned_names.txt"), "w")
    shutil.copytree(args.clean_data_dir, os.path.join(args.output_dir, "fingerprinted_images"))

    
    """ inject watermarked images into imageset"""
    for idx in range(len(all_fingerprinted_images)):
        if save_num <= args.poison_data_num:
            name = all_names[idx]
            name = name.split('/')[-1]
            image = all_fingerprinted_images[idx]            
            fp.write(name)
            fp.write('\n')
            
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{name}"), padding=0)
        save_num += 1
    f.close()
    fp.close()
    

def main():
    load_data()
    embed_fingerprints()


if __name__ == "__main__":
    main()
