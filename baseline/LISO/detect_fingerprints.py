import os
import torch
import numpy as np
import argparse
from liso import LISO
from liso.fnns import solve_lbfgs
from liso.encoders import BasicEncoder
from liso.decoders import BasicDecoder, DenseDecoder
from liso.critics import BasicCritic
from liso.utils import calc_psnr, calc_ssim, to_np_img
from tqdm import tqdm
from imageio import imread, imwrite
from utils import get_loader, get_path, get_loader_embed
from PIL import Image


parser = argparse.ArgumentParser("Learning Iterative Neural Optimizers for Image Steganography")
# task
parser.add_argument("--dataset", type=str, default="div2k")
parser.add_argument("--bits", type=int, default=1)
parser.add_argument("--jpeg", action="store_true")  # use jpeg instead of png

# training
parser.add_argument("--seed", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--random-crop", type=int, default=360)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--limit", type=int, default=800, help="number of training images")

# architecture
parser.add_argument("--hidden-size", type=int, default=32)
parser.add_argument("--dense-decoder", action="store_true")
parser.add_argument("--no-critic", action="store_true")

# LISO
parser.add_argument("--mse-weight", type=float, default=1.0)
parser.add_argument("--step-size", type=float, default=1.0)
parser.add_argument("--iters", type=int, default=15)

# inference
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--test-step-size", type=float, default=0.1)
parser.add_argument("--test-iters", type=int, default=150)

# steganalysis
parser.add_argument("--kenet-weight", type=float, default=0)  # aka SiaStegNet
parser.add_argument("--test-kenet-weight", type=float, default=0)
parser.add_argument("--xunet-weight", type=float, default=0)
parser.add_argument("--test-xunet-weight", type=float, default=0)

# evaluation
parser.add_argument("--lbfgs", action="store_true")
parser.add_argument("--eval-jpeg", action="store_true")
parser.add_argument("--constraint", type=float, default=None, help="pixel-wise perturbation constraint")
parser.add_argument("--check", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--poison_ratio", type=float, default=1)
parser.add_argument("--thr1", type=float, default=51.8)
parser.add_argument("--thr2", type=float, default=528)

args = parser.parse_args()
print(torch.get_num_threads())


if __name__ == "__main__":
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    validation = get_loader_embed(args)

    if args.eval and args.load is None:
        print("Creating a new model.")
        model = LISO(
            data_depth=args.bits,
            encoder=BasicEncoder,
            decoder=DenseDecoder if args.dense_decoder else BasicDecoder,
            critic=BasicCritic,
            hidden_size=args.hidden_size,
            iters=args.iters,
            lr=args.lr,
            opt=args.opt,
            jpeg=args.jpeg,
            kenet_weight=args.kenet_weight,
            xunet_weight=args.xunet_weight,
            no_critic=args.no_critic)
    if args.load is not None and os.path.isfile(args.load):
        print(f"Loading pretrained weight from {args.load}.")
        model = LISO.load(path=args.load)
    else:
        print("Creating a new model.")
        model = LISO(
            data_depth=args.bits,
            encoder=BasicEncoder,
            decoder=DenseDecoder if args.dense_decoder else BasicDecoder,
            critic=BasicCritic,
            hidden_size=args.hidden_size,
            iters=args.iters,
            lr=args.lr,
            opt=args.opt,
            jpeg=args.jpeg,
            kenet_weight=args.kenet_weight,
            xunet_weight=args.xunet_weight,
            no_critic=args.no_critic)

    if args.eval:
        model.encoder.iters = args.test_iters
        model.encoder.step_size = args.test_step_size
    else:
        model.encoder.step_size = args.step_size

    if args.eval and args.test_kenet_weight > 0:
        args.kenet_weight = args.test_kenet_weight
    model.encoder.set_kenet(args.kenet_weight)

    if args.eval and args.test_xunet_weight > 0:
        args.xunet_weight = args.test_xunet_weight
    model.encoder.set_xunet(args.xunet_weight)

    model.jpeg = args.jpeg or args.eval_jpeg
    model.mse_weight = args.mse_weight
    model.encoder.constraint = args.constraint
    
    gen_watermark = False



    img_names = [os.path.basename(x[0]).split(".")[0] for x in validation.dataset.imgs]

    times, steps, errors, ssims, psnrs = [], [], [], [], []
    if args.kenet_weight > 0:
        detect_kenets = []
    if args.xunet_weight > 0:
        detect_xunets = []
        
    pbar_max = len(validation)
    cur_poison_num = 0
    mean_acc = 0
    detect_n1 = 0
    detect_n2 = 0
    with tqdm(total=pbar_max) as pbar:
        for i, (cover, _) in enumerate(validation):
            cover = cover.cuda()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            
            if not gen_watermark:
                payload_i = model._random_data(cover[0].unsqueeze(0))[0]
                gen_watermark = True
            
            payload = payload_i.expand(cover.shape[0], *payload_i.shape)
            #print(cover.shape)
            with torch.no_grad():
                decoded = model._decode(cover)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

            with torch.no_grad():
                _errors = [float(1 - (x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()) * 100 for x in decoded]
                print(_errors)
                acc = 100 - _errors[0]
                if acc >= args.thr1:
                    detect_n1 += 1
                if acc >= args.thr2:
                    detect_n2 += 1
            mean_acc = mean_acc + acc 
            pbar.update(1)
        mean_acc /= len(validation)
        print('accuracy: ', mean_acc)
        print(f'detected mumber over thr1({args.thr1}): {detect_n1}')
        print(f'detected mumber over thr2({args.thr2}): {detect_n2}')

