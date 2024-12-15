import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Directory with image dataset.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results to.")
parser.add_argument("--image_resolution", type=int, default=128, required=True, help="Height and width of square images.",)
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0, help="cuda device selection")
parser.add_argument("--noise_ratio", type=float, default=3, help="perturbation added value")
parser.add_argument("--image_in", action='store_true', help="model input taking image or not0")
parser.add_argument("--block_length", type=int, default=1, help="1: pixel-based watermarking, choice of block length usage")
parser.add_argument("--jpg_noise", action="store_true", help="train with jpg compression")
parser.add_argument("--mask_train_epoch", type=int, default=18, help="masking training epochs number, choose 0 can avoid using masking")
parser.add_argument("--thr", type=float, default=0.55, help="start mask training accuracy")
parser.add_argument("--mix_train", action='store_true', help="mixed noise training")
args = parser.parse_args()


import glob
import os
from os.path import join
from time import time
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import numpy as np

import models

import io

device = torch.device("cuda")

LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
	os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
	os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
	os.makedirs(SAVED_IMAGES)

def expand_fingerprints(fingerprints):
    scale_factor = args.block_length
    fingerprints_expand = F.interpolate(fingerprints, scale_factor=scale_factor, mode='nearest')
    return fingerprints_expand

	

def generate_random_fingerprints(batch_size=4, size=(400, 400)):
	z = torch.rand((batch_size, 3, *size), dtype=torch.float32)
	z[z < 0.5] = 0
	z[z > 0] = 1
	return z


plot_points = (
	list(range(0, 1000, 100))
	+ list(range(1000, 3000, 200))
	+ list(range(3000, 100000, 500))
)

	
#################################################################################################  
def random_jpeg_compress(batch, probability=0.5, quality_min=40, quality_max=60):
	"""
	对 batch 中的部分图片随机应用 JPEG 压缩
	:param batch: 输入的张量 batch，形状为 (batch_size, C, H, W)
	:param probability: 每张图片应用压缩的概率
	:param quality_min: 最低 JPEG 质量
	:param quality_max: 最高 JPEG 质量
	:return: 处理后的 batch
	"""
	processed_images = []
    
	for img_tensor in batch:
        # 随机决定是否应用 JPEG 压缩
		if torch.rand(1).item() < probability:
			# 转回 PIL.Image
			img = transforms.ToPILImage()(img_tensor.cpu())
			
			# 随机选择压缩质量
			quality = torch.randint(quality_min, quality_max + 1, (1,)).item()
			
			# 应用 JPEG 压缩
			buffer = io.BytesIO()
			img.save(buffer, format="JPEG", quality=quality)
			buffer.seek(0)
			img = PIL.Image.open(buffer)
            
            # 转回 Tensor
			img_tensor = transforms.ToTensor()(img).to(device)
        
			# 将（压缩或未压缩的）图片加入列表
		processed_images.append(img_tensor)
		
    
    # 拼接回一个 batch
	return torch.stack(processed_images)

def random_augment(batch, probability=0.5, quality_min=40, quality_max=60):
    """
    对 batch 中的每张图片随机应用以下一种操作：
    - JPEG 压缩
    - Color Jitter
    - Gaussian Noise
    - Gaussian Blur

    :param batch: 输入的张量 batch，形状为 (batch_size, C, H, W)
    :param probability: 每张图片应用增强的概率
    :param quality_min: JPEG 压缩的最低质量
    :param quality_max: JPEG 压缩的最高质量
    :return: 处理后的 batch
    """
    processed_images = []
    device = batch.device  # 确保保持张量设备一致

    for img_tensor in batch:
        if torch.rand(1).item() < probability:
            # 随机选择一种增强方法
            augment_choice = random.choice(['jpeg', 'color_jitter', 'gaussian_noise', 'gaussian_blur'])

            if augment_choice == 'jpeg':
                # JPEG 压缩
                img = transforms.ToPILImage()(img_tensor.cpu())
                quality = torch.randint(quality_min, quality_max + 1, (1,)).item()
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                img = PIL.Image.open(buffer)
                img_tensor = transforms.ToTensor()(img).to(device)

            elif augment_choice == 'color_jitter':
                # Color Jitter
                color_jitter = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
                img = transforms.ToPILImage()(img_tensor.cpu())
                img = color_jitter(img)
                img_tensor = transforms.ToTensor()(img).to(device)

            elif augment_choice == 'gaussian_noise':
                # Gaussian Noise
                noise = torch.randn_like(img_tensor) * 0.1  # 调整噪声强度
                img_tensor = (img_tensor + noise).clamp(0, 1)

            elif augment_choice == 'gaussian_blur':
                # Gaussian Blur
                img = transforms.ToPILImage()(img_tensor.cpu())
                img = img.filter(PIL.ImageFilter.GaussianBlur(radius=2))  # 调整模糊半径
                img_tensor = transforms.ToTensor()(img).to(device)

        # 将处理后的图片加入列表
        processed_images.append(img_tensor)

    # 拼接回一个 batch
    return torch.stack(processed_images)


#################################################################################################        
class SubImageFolder(torch.utils.data.Subset):
	def __init__(self, dataset, indices):
		super().__init__(dataset, indices)  
		
	def __getitem__(self, index):
		data, _ = super().__getitem__(index)
		return data, 0

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
		return image, 0

	def __len__(self):
		return len(self.filenames)


def load_data():
	global train_dataset, valid_dataset, train_dataloader, valid_dataloader
	global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

	IMAGE_RESOLUTION = args.image_resolution
	IMAGE_CHANNELS = 3


	transform = transforms.Compose(
		[
			#transforms.Pad((8,8,8,8), fill=0, padding_mode="constant"),
			#transforms.RandomRotation(45),
			#transforms.RandomCrop(IMAGE_RESOLUTION),
			transforms.Resize(IMAGE_RESOLUTION),
			transforms.CenterCrop(IMAGE_RESOLUTION),
			transforms.ToTensor(),
		]
	)
    
	s = time()
	print(f"Loading image folder {args.data_dir} ...")
	dataset = CustomImageFolder(args.data_dir, transform=transform)
	indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
	train_indices = indices[:int(0.8 * len(indices))]
	valid_indices = indices[int(0.8 * len(indices)):]
	train_dataset = SubImageFolder(dataset, train_indices)
	valid_dataset = SubImageFolder(dataset, valid_indices)
	print(f"Finished. Loading took {time() - s:.2f}s")

def main():
	load_data()

	mask_train = (args.mask_train_epoch != 0)
	if mask_train:
		thr = args.thr

	if args.image_in:
		decoder = models.StegaStampDecoder(
			IMAGE_CHANNELS,
			True
		)
	elif args.block_length != 1:
		"""
			block-based model import
		"""
		if (args.block_length % 2) != 0 or (args.image_resolution % args.block_length) != 0:
			raise Exception("block length not supported")
		decoder = models.StegaStampDecoder(
			IMAGE_CHANNELS,
			False,
			True,
            shrink_rate=args.block_length
		)
	else:
		"""
			pixel-based model import
		"""
		decoder = models.StegaStampDecoder(
			IMAGE_CHANNELS,
			False
		)
	decoder = decoder.to(device)

	global_step = 0
	
	train_dataloader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
	)
	valid_dataloader = DataLoader(
		valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
	)
		
	decoder_optim = Adam(
		params=decoder.parameters(), lr=args.lr
	)       
    
	""" mask initial setting """
	start_mask = False
	if mask_train:
		diff_mask = torch.ones((3,args.image_resolution//args.block_length, args.image_resolution//args.block_length)).float().to(device)
	
	for i_epoch in range(args.num_epochs):
		""" training """
		for images, _ in tqdm(train_dataloader):
			global_step += 1
			decoder.train()

			""" random watermark select """
			batch_size = min(args.batch_size, images.size(0))
			org_fingerprints = generate_random_fingerprints(batch_size, (args.image_resolution//args.block_length, args.image_resolution//args.block_length)).to(device)

			""" masking """
			if start_mask:
				m = mask.repeat(org_fingerprints.shape[0], 0)
				detected_numbers = np.sum(m)
				org_fingerprints[~m] = 0	

			fingerprints = expand_fingerprints(org_fingerprints) # expand watermark if block-based watermark used

			clean_images = images.to(device)
			fingerprints = fingerprints.to(device).bool()
			""" image to frequency """
			clean_freq = torch.fft.fftshift(torch.fft.fft2(clean_images))

			""" add perturbation """
			clean_freq.real[fingerprints] = clean_freq.real[fingerprints] + args.noise_ratio
			clean_freq.imag[fingerprints] = clean_freq.imag[fingerprints] + args.noise_ratio

			""" frequency to image """
			fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(clean_freq)))

			""" quantization """
			fingerprinted_images = torch.clamp(fingerprinted_images, min=0, max=1) * 255
			fingerprinted_images = fingerprinted_images.round() / 255

			""" jpg compression """
			if args.jpg_noise:
				fingerprinted_images = random_jpeg_compress(fingerprinted_images, probability=1)
			elif args.mix_train:
				fingerprinted_images = random_augment(fingerprinted_images, probability=1)
			
			""" image to frequency """
			input_freq = torch.fft.fftshift(torch.fft.fft2(fingerprinted_images))

			"""split real part and imaginary part """
			real = torch.real(input_freq)
			imag = torch.imag(input_freq)

			""" change shape """
			input_freq = torch.concatenate((real, imag), dim=1)
			input_freq = input_freq.to(device)

			""" input images """
			input_freq = input_freq / torch.max(input_freq)
			fingerprints = fingerprints.float()
			residual = fingerprinted_images - clean_images
			decoder_output = decoder(input_freq)
			
			""" LOSS """
			criterion = nn.BCEWithLogitsLoss()

			""" masking the unused pixels(blocks) """
			if start_mask:
				mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(decoder_output.device)
				loss_decoder_output = decoder_output * mask_tensor
				loss_org_fingerprints = org_fingerprints * mask_tensor
				BCE_loss = criterion(loss_decoder_output.view(-1), loss_org_fingerprints.view(-1))
			else:
				BCE_loss = criterion(decoder_output.view(-1), org_fingerprints.view(-1))

			loss = BCE_loss

			""" optimization """
			decoder.zero_grad()
			loss.backward()
			decoder_optim.step()

			""" perfoemance evaluation """
			fingerprints_predicted = (decoder_output > 0).float()
			if start_mask:
				fingerprints_predicted[~m] = 0
				diff = (~torch.logical_xor(org_fingerprints, fingerprints_predicted))
				diff[~m] = 0
				bitwise_accuracy = torch.sum(diff) / detected_numbers
			else:
				bitwise_accuracy = 1.0 - torch.mean(
					torch.abs(org_fingerprints- fingerprints_predicted)
				)
            

			""" Logging """
			if global_step in plot_points:
				print("Bitwise accuracy {}".format(bitwise_accuracy))
				print("BCE Loss {}".format(BCE_loss))
				print(
					"residual_statistics: {}".format(
						{
							"min": residual.min(),
							"max": residual.max(),
							"mean_abs": residual.abs().mean(),
						}
					)
				)
				save_image(
					fingerprinted_images,
					SAVED_IMAGES + "/fin_img{}.png".format(global_step),
					normalize=True,
				)
				save_image(
					decoder_output,
					SAVED_IMAGES + "/out_{}.png".format(global_step),
					normalize=True,
				)
				save_image(
					fingerprints,
					SAVED_IMAGES + "/fin_{}.png".format(global_step),
					normalize=True,
				)


			""" checkpoint """
			if global_step % 1000 == 0:
				torch.save(
					decoder_optim.state_dict(),
					join(CHECKPOINTS_PATH,"optim.pth"),
				)
				torch.save(
					decoder.state_dict(),
					join(CHECKPOINTS_PATH,"decoder.pth"),
				)
				f = open(join(CHECKPOINTS_PATH,"variables.txt"), "w")
				f.write(str(global_step))
				f.close()
		
		""" validation """
		valid_acc = 0
		valid_loss = 0
		with torch.no_grad():
			for images, _ in tqdm(valid_dataloader):
				decoder.eval()

				""" random watermark select """
				batch_size = min(args.batch_size, images.size(0))
				org_fingerprints = generate_random_fingerprints(batch_size, (args.image_resolution//args.block_length, args.image_resolution//args.block_length)).to(device)

				if start_mask:
					m = torch.tensor(mask.repeat(org_fingerprints.shape[0], 0))
					detected_numbers = torch.sum(m)
					org_fingerprints[~m] = 0

				fingerprints = expand_fingerprints(org_fingerprints)
				
				clean_images = images.to(device)
				fingerprints = fingerprints.to(device).bool()
				""" image to frequency """
				clean_freq = torch.fft.fftshift(torch.fft.fft2(clean_images))
				
				""" add perturbation """
				clean_freq.real[fingerprints] = clean_freq.real[fingerprints] + args.noise_ratio
				clean_freq.imag[fingerprints] = clean_freq.imag[fingerprints] + args.noise_ratio
				""" frequency to image """
				fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(clean_freq)))
				""" quantization """
				fingerprinted_images = torch.clamp(fingerprinted_images, min=0, max=1) * 255
				fingerprinted_images = fingerprinted_images.round() / 255
				
				""" jpg compression """
				if args.jpg_noise:
					fingerprinted_images = random_jpeg_compress(fingerprinted_images, probability=1)
				elif args.mix_train:
					fingerprinted_images = random_augment(fingerprinted_images, probability=1)

				""" image to frequency """
				input_freq = torch.fft.fftshift(torch.fft.fft2(fingerprinted_images))

				""" split real part and imaginary part """
				real = torch.real(input_freq)
				imag = torch.imag(input_freq)
				
				""" change shape """
				input_freq = torch.concatenate((real, imag), dim=1)
				input_freq = input_freq.to(device)
				input_freq = input_freq / torch.max(input_freq)
				
				""" input image """
				fingerprints = fingerprints.float()
				residual = fingerprinted_images - clean_images
				decoder_output = decoder(input_freq)
				
				""" loss """
				criterion = nn.BCEWithLogitsLoss()
				BCE_loss = criterion(decoder_output.view(-1), org_fingerprints.view(-1))
				loss = BCE_loss			
				valid_loss += loss.detach().cpu()

				""" performance evaluation """
				fingerprints_predicted = (decoder_output > 0).float()
				if start_mask:
					fingerprints_predicted[~m] = 0
					diff = (~torch.logical_xor(org_fingerprints, fingerprints_predicted)) # b, c, h, w
					tmp = diff.float()
					diff_mask = diff_mask + torch.mean(tmp, dim=0) # c, h, w
					diff_mask[~mask] = 0
					diff[~m] = 0
					bitwise_accuracy = torch.sum(diff) / detected_numbers
				else:
					bitwise_accuracy = 1.0 - torch.mean(
						torch.abs(org_fingerprints- fingerprints_predicted)
					)
					
					if mask_train:
						diff = (~torch.logical_xor(org_fingerprints, fingerprints_predicted)).float() # b, c, h, w
						diff_mask = diff_mask + torch.mean(diff, dim=0) # c, h, w
			
				valid_acc += bitwise_accuracy
			
			valid_acc /= len(valid_dataloader)
			valid_loss /= len(valid_dataloader)

			""" mask generation """
			if i_epoch >= (args.num_epochs - args.mask_train_epoch):
				start_mask = True

			if mask_train:
				diff_mask = diff_mask / len(valid_dataloader)

			if start_mask:
				if thr == args.thr:
					thr = thr + 0.01
				else:
					if np.sum(mask) >= 300:
						print(detected_numbers, thr)
						thr = thr + 0.01

				print(diff_mask)
				mask = (diff_mask > thr).cpu().numpy()
				mask = np.expand_dims(mask, axis=0)
				np.save(f"{args.output_dir}mask.npy", mask)
				print("detected numbers: ", np.sum(mask))

			if mask_train:
				print(f"epoch: {i_epoch}, valid_loss: {valid_loss}, valid_acc: {valid_acc}, thr: {thr}")
			else:			
				print(f"epoch: {i_epoch}, valid_loss: {valid_loss}, valid_acc: {valid_acc}")
	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()


if __name__ == "__main__":
	main()
