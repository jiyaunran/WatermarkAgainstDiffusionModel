import numpy as np
import os
import torch
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
parser.add_argument("--batch_size", type=int, default=2)
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
parser.add_argument("--image_save_folder", type=str, required=True)
parser.add_argument("--check", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--poison_ratio", type=float, default=1)

args = parser.parse_args()
print(torch.get_num_threads())

if not os.path.exists(args.image_save_folder):
    os.mkdir(args.image_save_folder)

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
    print(f"{len(img_names)} images will be saved to {args.image_save_folder}.")

    times, steps, errors, ssims, psnrs = [], [], [], [], []
    if args.kenet_weight > 0:
        detect_kenets = []
    if args.xunet_weight > 0:
        detect_xunets = []
        
    pbar_max = len(validation)*args.poison_ratio
    cur_poison_num = 0
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
            
            with torch.no_grad():
                generated, payload, decoded, grads, ptbs = model._encode_decode(cover, payload=payload, quantize=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

            original_image = imread(validation.dataset.imgs[i][0], pilmode="RGB").astype(np.float32)
            _psnrs = [calc_psnr(
                original_image,
                to_np_img(x[0], dtype=np.float32)) for x in generated]
            with torch.no_grad():
                _errors = [float(1 - (x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()) * 100 for x in decoded]
                #print(_errors)

            # Order of priority: avoid steganalysis detection > minimize decoding error > maximize PSNR
            costs = np.array([-y if x == 0 else x for x, y in zip(_errors, _psnrs)])
            if args.kenet_weight > 0:
                _detect_kenets = model.encoder.kenet_detect(generated)
                costs += 100 * np.array(_detect_kenets).astype(np.float32)
            if args.xunet_weight > 0:
                _detect_xunets = model.encoder.xunet_detect(generated)
                costs += 100 * np.array(_detect_xunets).astype(np.float32)
            best_idx = np.argmin(costs)

            steps.append(best_idx)
            errors.append(_errors[best_idx])
            if args.kenet_weight > 0:
                detect_kenets.append(_detect_kenets[best_idx])
            if args.xunet_weight > 0:
                detect_xunets.append(_detect_xunets[best_idx])

            # save the best output and reload from disk
            generated = to_np_img(generated[best_idx][0])
            if args.jpeg or args.eval_jpeg:
                img_save_path = args.image_save_folder
                Image.fromarray(generated).save(args.img_save_path, format="jpeg", quality=80)
                generated = np.asarray(Image.open(img_save_path))
            else:
                imwrite(os.path.join(args.image_save_folder, f"{img_names[i]}.png"), generated)
            
            cur_poison_num += 1
            
            # LISO + L-BFGS
            if args.lbfgs:
                t, generated = solve_lbfgs(model.decoder, generated, payload)
                times[-1] += t

            #ssims.append(calc_ssim(original_image, generated.astype(np.float32)))
            #psnrs.append(calc_psnr(original_image, generated.astype(np.float32)))

            #log_str = f"{img_names[i]}, time: {times[-1]:0.2f}ms, steps: {steps[-1]}, error: {errors[-1]:0.2f}%, ssim: {ssims[-1]:0.3f}, psnr: {psnrs[-1]:0.2f}"
            #if args.kenet_weight > 0:
            #    log_str += f", avoid_kenet: {1 - detect_kenets[-1]}"
            #if args.xunet_weight > 0:
            #    log_str += f", avoid_xunet: {1 - detect_xunets[-1]}"
            
            if cur_poison_num >= pbar_max:
                break
        print(f"Error: {np.mean(errors):0.2f}%")
        #print(f"SSIM: {np.mean(ssims):0.3f}")
        #print(f"PSNR: {np.mean(psnrs):0.2f}")
        
        pbar.update(1)

