import argparse
import os
import glob
import PIL
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--clean_data_dir", type=str, required=True, help="clean data directory to merge in")
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=38, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--poison_data_num", type=int, default=1000)


args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2, generator=torch.Generator().manual_seed(args.seed))
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

def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = 100

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []
    all_images = []
    all_names = []

    print("Fingerprinting the images...")
    #torch.manual_seed(args.seed)

    # generate identical fingerprints
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)
    fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    #torch.manual_seed(args.seed)

    bitwise_accuracy = 0
    save_num = 1

    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    fp = open(os.path.join(args.output_dir, "poisoned_names.txt"), "w")
    shutil.copytree(args.clean_data_dir, os.path.join(args.output_dir, "fingerprinted_images"))
    for images, names in tqdm(dataloader):

        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)

        fingerprinted_pert = HideNet(fingerprints[: images.size(0)], images)
        fingerprinted_images = images + fingerprinted_pert
        fingerprinted_images = torch.clamp(fingerprinted_images, min=0., max=1.)
        #all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        #all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())
        #all_images.append(images.detach().cpu())
        for name in names:
            all_names.append(name)

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()

        for i in range(images.shape[0]):
            if save_num <= args.poison_data_num:
                name = names[i]
                name = name.split('/')[-1]
                image = fingerprinted_images[i]            
                fp.write(name)
                fp.write('\n')
            else:
                break
            save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{name}"), padding=0)
            save_num += 1
            


    #all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    #all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    #all_images = torch.cat(all_images, dim=0).cpu()
    
    #os.system(f"cp -r /home/jyran2001208/data/known_data/30000_clean/* {os.path.join(args.output_dir, 'fingerprinted_images')}")
    
    #f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    #fp = open(os.path.join(args.output_dir, "poisoned_names.txt"), "w")
    #if not os.path.exists(os.path.join(args.output_dir,"fingerprinted_images")):
    #    os.mkdir(os.path.join(args.output_dir, "fingerprinted_images"))
    #shutil.copytree(args.clean_data_dir, os.path.join(args.output_dir, "fingerprinted_images"))
    #for idx in range(len(all_fingerprinted_images)):
    #    if save_num <= args.poison_data_num:
    #        name = all_names[idx]
    #        name = name.split('/')[-1]
    #        image = all_fingerprinted_images[idx]            
    #        fp.write(name)
    #        fp.write('\n')
    #    else:
    #        break
    #    _, filename = os.path.split(dataset.filenames[idx])
    #    filename = filename.split('.')[0] + ".png"
    #    save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{name}"), padding=0)
    #    save_num += 1
    f.close()
    fp.close()


def main():

    load_data()
    load_models()

    embed_fingerprints()


if __name__ == "__main__":
    main()
