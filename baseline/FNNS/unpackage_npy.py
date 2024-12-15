import numpy as np
import glob
import os

import argparse 
from tqdm import tqdm
import PIL
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=512)
args = parser.parse_args()

if not os.path.exists(os.path.join(args.data_dir, "fingerprinted_images")):
    os.mkdir(os.path.join(args.data_dir, "fingerprinted_images"))

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
        
        
if __name__ == "__main__":
    dataset = CustomImageFolder(args.image_dir)
    dataloader = DataLoader(
		dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
	)
    
    image_names = glob.glob(os.path.join(args.data_dir, "*.png"))
    i = 0 
    for (_, image_name) in tqdm(dataloader):
        npy_file = np.load(os.path.join(args.data_dir, f"images_{i}.npy"))
        for j in range(npy_file.shape[0]):
            save_image = npy_file[j]           
            save_image = Image.fromarray(save_image)
            save_image.save(os.path.join(args.data_dir, "fingerprinted_images", image_name[j].split('/')[-1]))
        i =i + 1