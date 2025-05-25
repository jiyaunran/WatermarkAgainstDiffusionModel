# FPW(Frequency-domain Pixel-by-Pixel Watermarking)
This repo is the official code for [https://openreview.net/forum?id=4hSqhUhF20&noteId=4hSqhUhF20](https://openreview.net/forum?id=eEJvT5LDwa&noteId=eEJvT5LDwa)

## Dependencies and Installation
- Python 3.9.18
- Pytroch 2.1.0+cu121
- Torchvision 0.16.0
- TensorboardX 2.6.2.2

## Extra Experiment
Due to the page limitations of ICIP, some experimental results could not be included in the main paper. We present them below for completeness.

### Spatial Domain v.s. FFT
We compare the watermark reproducibility and image distortion when embedding the watermark in the spatial domain versus the FFT domain. The experiment is conducted on the CelebA dataset, using two different watermark configurations with the same poisoning rate. The results are shown below:
![image](https://github.com/user-attachments/assets/bdcf78cb-5b45-4ee9-8b67-737b5495d07b)

### Robustness testing
We evaluate the robustness of the watermark against common image processing augmentations. The applied transformations include:
JPEG Compression (50% quality)
*Gaussian Noise (σ = 25 / 255)
*Gaussian Blur (2×2 kernel)
*Color Jitter (brightness factor sampled from [0.8, 1.2])

To investigate whether training with augmentations improves robustness, we introduce two variants:
*FPW<sub>jpg</sub>: trained with JPEG-compressed images.
*FPW<sub>mix</sub>: trained with all the augmentations applied with equal probability.

All experiments are conducted on the CelebA dataset.
The following results show the detection accuracy on watermarked images after augmentation:
![image](https://github.com/user-attachments/assets/65247aac-ede1-485b-85be-c094732e7e81)

We also evaluate performance under a poison ratio of 0.3, to simulate a more challenging, low-signal regime:
![image](https://github.com/user-attachments/assets/120e00ed-b27c-4e79-bbe9-f98a0db18498)


## Training
Choose a proper noise ratio can maintain both steathiness and effectiveness.
For exam, we use 3 on CelebA(64x64) and Cifar10(32x32)
```
python train.py --data_dir PATH_TO_DATASET \
--image_resolution YOUR_IMAGE_RESOLUTION \
--noise_ratio 3 \
--output_dir "./model/" \
--batch_size 256 \
--num_epochs 30 \
--block_length BLOCK_WATERMARK_LENGTH \
--cuda 0
```
## Embedding
For clean_data_dir, it gives a choice of merging a clean dataset with watermarked one. If there is no need, please leave it the same as PATH_TO_DATASET.
poison_data_num give an option of poison image number. If all poison is wanted, leave a big number will do the work.
```
python embed_fingerprints.py --data_dir PATH_TO_DATASET \
--clean_data_dir PATH_TO_CLEAN_DATASET \
--batch_size 256 \
--noise_ratio 3 \
--image_resolution YOUR_IMAGE_RESOLUTION \
--poison_data_num AMOUT_OFFINGERPRINTED_IMAGES \
--block_length BLOCK_WATERMARK_LENGTH \
--output_dir OUTPUT_PATH
```
## Generate Masking
Use a clean dataset to constraint generation of mask.
```
python gen_proper_cor.py --constraint_data_dir PATH_TO_DATASET \
--data_dir PATH_TO_SUR_GEN_DATASET \
--batch_size 256 \
--image_resolution YOUR_IMAGE_RESOLUTION \
--decoder_path MODEL_PATH \
--output_path OUTPUT_PATH \
--block_length BLOCK_WATERMARK_LENGTH \
--cuda 0
```
## Detection
If the masking method is used, use --mask 1 and type in the MASK_PATH 
```
python detect_fingerprints.py --data_dir PATH_TO_DATASET \
--image_resolution 64 \
--decoder_path MODEL_PATH \
--batch_size 256 \
--thr DETECTION THR 1 \
--thr2 DETECTION THR 2 \
--mask 1 \
--mask_path MASK_PATH \
--block_length BLOCK_WATERMARK_LENGTH \
--cuda 0
```

# Belows are baseline method conduction command
## FNNS
Embed Images with below command:
```
python embed_fingerprints.py --data_dir "/nfs/home/julian2001208/data/data/FFHQ/seen/" \
--clean_data_dir "/nfs/home/julian2001208/data/data/FFHQ/unseen/" \
--output_dir "/nfs/home/julian2001208/data/data/FNN_pn03_FFHQ/" \
--batch_size 128 \
--poison_ratio 0.3 \
--cuda 0
```

The embedded code will generate a compact npy files, use below coomman to unpackged it to png folder
```
python unpackage_npy.py --data_dir /nfs/home/julian2001208/data/data/FNN_pn03_FFHQ/ \
--image_dir /nfs/home/julian2001208/data/data/FFHQ/seen/ \
--batch_size 32
```
Below code can conduct the Detection
```
python detect_fingerprints.py --data_dir "/nfs/home/julian2001208/data/data/FNN_pn01_FFHQ/generated_images/fulltrain50000/gen_img/" \
--batch_size 16 --image_resolution 256 --cuda 3 --thr 0.5077 --thr2 0.512
```
## LISO
LISO require training en-decoder first, below command can conduct training, detail usage can refer https://github.com/cxy1997/LISO
```
python train_bits.py --bits 1 --dataset /nfs/home/julian2001208/data/data/CelebA/ \
--save_dir /nfs/home/julian2001208/work/LISO/test_CelebA/
```
Below code embed the image dataset
```
python embed_fingerprints.py --eval --bits 1 --dataset /nfs/home/julian2001208/data/data/CelebA/seen/ \
--load ./test/checkpoints/best.steg \
--image_save_folder /nfs/home/julian2001208/data/data/LISO_pdn01_CelebA/ \
--poison_ratio 0.1
```
Detection can be done by below command:
```
python detect_fingerprints.py --eval --bits 1 --dataset /nfs/home/julian2001208/data/data/LISO_pdn01_cifar10/generated_images/fulltrain50000/ \
--load ./test_cifar10/checkpoints/best.steg
```

## HiDDeN

The training code:
```
python train.py --data_dir "/nfs/home/julian2001208/data/data/LSUN/seen/" \
--image_resolution 256 \
--budget 0.05 \
--output_dir "./budget_005_LSUN/" \
--batch_size 32 \
--cuda 6
```

Embedding code:
```
python embed_fingerprints.py --data_dir "/nfs/home/julian2001208/data/LSUN/seen/" \
--clean_data_dir "/nfs/home/julian2001208/data/LSUN/seen/" \
--batch_size 32 \
--image_resolution 256 \
--identical_fingerprints \
--poison_data_num 99999999 \
--output_dir "/nfs/home/julian2001208/data/hidden_pr005_fullp_LSUN/" \
--encoder_path ./budget_005_LSUN/checkpoints/stegastamp_100_encoder.pth \
--cuda 6
```

Detection:
```
python detect_fingerprints.py --data_dir /nfs/home/julian2001208/data/hidden_pr005_pdn01_LSUN/generated_images/fulltrain50000/gen_img/ \
--image_resolution 256 \
--output_dir "./output/" \
--decoder_path "./budget_005_LSUN/checkpoints/stegastamp_100_decoder.pth" \
--batch_size 32 \
--thr 0.7 \
--cuda 7
```

# Belows are generative model conduction guideline
## DDPM-IP
Detail setting for each dataset can refer https://github.com/forever208/DDPM-IP
Training:
I've make the bash code to simplify the argparse
```
bash train.sh -d /nfs/home/julian2001208/data/reconst_freq_pr10_bl4_FFHQ_mask/fingerprinted_images/ -i 256 -b 3 -s 50000 \
-v /nfs/home/julian2001208/data/reconst_freq_pr10_bl4_FFHQ_mask/DDPM/fulltrain50000/  -c 6
```
Sampling:
```
bash sample.sh -m "/nfs/home/julian2001208/data/reconst_freq_pr4_bl4_mix_pdn01_CelebA/DDPM/fulltrain50000/ema_0.9999_050000.pt" \
-i 64 -b 200 -n 3000 -o /nfs/home/julian2001208/data/reconst_freq_pr4_bl4_mix_pdn01_CelebA/generated_images/fulltrain50000/ -c 5
```
The sampling code will generate a compact npy files, i use below unpackaging command turning it to png or jpg folder
```
python npz2png.py --input_path /nfs/home/julian2001208/data/clean_trained/generated_images/samples_3000x256x256x3.npz \
--output_path /nfs/home/julian2001208/data/clean_trained/generated_images/gen_img/
```
```
python npz2jpg.py --input_path /nfs/home/julian2001208/data/clean_trained/generated_images/samples_3000x256x256x3.npz \
--output_path /nfs/home/julian2001208/data/clean_trained/generated_images/gen_img/
```

## Diffusion-GAN
Code from https://github.com/Zhendong-Wang/Diffusion-GAN/tree/main
Training:
```
python train.py --outdir=training-runs \
--data="/nfs/home/julian2001208/data/data/CelebA/seen.zip" --gpus=1 \
--cfg auto --kimg 50000 --aug no --target 0.6 --noise_sd 0.05 --ts_dist priority
```
```
Sampling:
python generate.py --outdir=reconst_freq_pdn03/gen_img/ \
--seeds=1-3000 \
--network=/nfs/home/julian2001208/work/Diffusion-GAN/diffusion-stylegan2/reconst_freq_pdn03/00000-fingerprinted-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/best_model.pkl
```

## Evaluation
Below code can evaluate the image_loss from two images folder, but notice that it compare the images one by one through image names, so the image_name has to be aligned.
Also since it use name for alignment, data_path2 must contain all images in data_path1. The evaluation doesn't conduct measurement from all images but randomly pick 5000 images for measurement.
```
python evaluate_image_loss.py --data_path1 /nfs/home/julian2001208/data/LSUN/seen/  \
--data_path2 "/nfs/home/julian2001208/data/reconst_freq_pr10_fullp_LSUN_BED/fingerprinted_images/"  \
--image_resolution 256	 \
--cuda 0
```
