import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Directory with image dataset.")
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results to.")
parser.add_argument("--fingerprint_length", type=int, default=100, help="Number of bits in the fingerprint.",)
parser.add_argument("--image_resolution", type=int, default=128, required=True, help="Height and width of square images.",)
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)
parser.add_argument("--l2_loss_await", help="Train without L2 loss for the first x iterations", type=int, default=1000,)
parser.add_argument("--l2_loss_weight", type=float, default=20, help="L2 loss weight for image fidelity.",)
parser.add_argument("--l2_loss_ramp", type=int, default=3000, help="Linearly increase L2 loss weight over x iterations.",)
parser.add_argument("--BCE_loss_weight", type=float, default=1, help="BCE loss weight for fingerprint reconstruction.",)
parser.add_argument("--use_regressor", action='store_true')
parser.add_argument("--regressor_ckpt", type=str, default='/home/jyran2001208/model/promote_poison/train_regression/Regressor3.pt')
parser.add_argument("--reg_loss_weight", type=float, default=1,)
parser.add_argument("--pretrained_epoch", type=int, default=100)
parser.add_argument("--budget", type=float, default=0.01)
parser.add_argument("--augmentation", type=str, default=None)
args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

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

import models

import lpips

device = torch.device("cuda")
lpips_function = lpips.LPIPS(net='vgg').to(device)


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


def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


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
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.fingerprint_length

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:
        if args.augmentation is not None:
            transform = transforms.Compose(
                [
                    transforms.Pad((8,8,8,8), fill=0, padding_mode="constant"),
                    transforms.RandomRotation(45),
                    transforms.RandomCrop(IMAGE_RESOLUTION),
                    transforms.Resize(IMAGE_RESOLUTION),
                    transforms.CenterCrop(IMAGE_RESOLUTION),
                    transforms.ToTensor(),
                ]
            )
            print('a')
        else:
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
    print(f"Finished. Loading took {time() - s:.2f}s")

def regression_loss(predict):
    loss = (1 - predict)
    
    return torch.sum(loss) / len(loss)

def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'lpips':
        loss = lpips_function(imgs, imgs_ori)
        loss = loss - args.budget
        loss = torch.clamp(loss, min=0)
        return torch.sum(loss)/len(loss)
    else:
        raise ValueError('Unknown loss type')

def main():
    EXP_NAME = f"stegastamp_{args.fingerprint_length}"

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_optim = Adam(encoder.parameters(), lr=args.lr)

    global_step = 0
    steps_since_l2_loss_activated = -1
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )
        
    if args.use_regressor:
        print("regressor used")
        assert os.path.exists(args.regressor_ckpt)
        
        regressor = torchvision.models.resnet50(num_classes=1).to(device)
        checkpoint = torch.load(args.regressor_ckpt)
        regressor.load_state_dict(checkpoint['model_state'])
        regressor.eval()
        
        reg_loss_weight = args.reg_loss_weight       
        
        # pretrain with regressor   
        
        #for epoch in range(args.pretrained_epoch):
        #    train_loss = 0
        #    train_loss_l2 = 0
        #    train_loss_reg = 0
        #    for images, _ in dataloader:
        #        batch_size = min(args.batch_size, images.size(0))
        #        fingerprints = generate_random_fingerprints(
        #            args.fingerprint_length, batch_size, (args.image_resolution, args.image_resolution)
        #        )
        #        
        #        l2_loss_weight = min(
        #            max(
        #                0,
        #                args.l2_loss_weight
        #                * (steps_since_l2_loss_activated - args.l2_loss_await)
        #                / args.l2_loss_ramp,
        #            ),
        #            args.l2_loss_weight,
        #        )
        #        
        #        clean_images = images.to(device)
        #        fingerprints = fingerprints.to(device)
        #        
        #        fingerprinted_pert = encoder(fingerprints, clean_images)
        #        fingerprinted_images = fingerprinted_pert + clean_images
        #        
        #        reg_score = regressor(fingerprinted_images)
        #        
        #        #criterion = nn.MSELoss()
        #        #l2_loss = criterion(fingerprinted_images, clean_images)
        #        l2_loss = image_loss(fingerprinted_images, clean_images, 'lpips')
        #        
        #        criterion = regression_loss
        #        reg_loss = criterion(reg_score)
        #        
        #        loss = reg_loss_weight * reg_loss + l2_loss_weight * l2_loss
        #        
        #        encoder.zero_grad()
        #        decoder.zero_grad()
        #        
        #        loss.backward()
#
        #        encoder_optim.step()
        #        
        #        train_loss += loss.detach().cpu()
        #        train_loss_l2 += l2_loss.detach().cpu()
        #        train_loss_reg += reg_loss.detach().cpu()
        #        
        #    train_loss /= len(dataloader)
        #    train_loss_l2 /= len(dataloader)
        #    train_loss_reg /= len(dataloader)
        #    
        #    print(f'pretrain epoch: {epoch}, total loss: {train_loss}, l2 loss: {train_loss_l2}, reg loss: {train_loss_reg}')
        #    
        #torch.save(
        #    encoder.state_dict(),
        #    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
        #)
            
    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )       

    for i_epoch in range(args.num_epochs):
        #dataloader = DataLoader(
        #    dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        #)
        for images, _ in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(
                args.fingerprint_length, batch_size, (args.image_resolution, args.image_resolution)
            )

            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            
            reg_loss_weight = min(
                max(
                    0,
                    args.reg_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.reg_loss_weight,
                ),
                args.reg_loss_weight,
            )
            
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)
            
            #fingerprinted_images = encoder(fingerprints, clean_images)

            fingerprinted_pert = encoder(fingerprints, clean_images)
            fingerprinted_images = fingerprinted_pert + clean_images
            
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            #criterion = nn.MSELoss()
            #l2_loss = criterion(fingerprinted_images, clean_images)
            l2_loss = image_loss(fingerprinted_images, clean_images, 'lpips')
            
            criterion = nn.BCEWithLogitsLoss()
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
            
            if args.use_regressor:
                predict = regressor(fingerprinted_images)
                reg_loss = regression_loss(predict)
                loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss + reg_loss_weight * reg_loss
            else:
                loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            #loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss
            
            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1

            # Logging
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                print("l2 loss {}".format(l2_loss))
                writer.add_scalar("loss", loss, global_step),
                writer.add_scalar("BCE_loss", BCE_loss, global_step),
                writer.add_scalar("l2_loss", l2_loss, global_step),
                writer.add_scalars(
                    "clean_statistics",
                    {"min": clean_images.min(), "max": clean_images.max()},
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                writer.add_image(
                    "clean_image", make_grid(clean_images, normalize=True), global_step
                )
                writer.add_image(
                    "residual",
                    make_grid(residual, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "image_with_fingerprint",
                    make_grid(fingerprinted_images, normalize=True),
                    global_step,
                )
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_weights/l2_loss_weight", l2_loss_weight, global_step
                )
                writer.add_scalar(
                    "loss_weights/BCE_loss_weight",
                    BCE_loss_weight,
                    global_step,
                )

            # checkpointing
            if global_step % 1000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
