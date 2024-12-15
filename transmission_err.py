import os
import argparse
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from tqdm import tqdm

# 原始圖像目錄與處理後的目錄
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/nfs/home/julian2001208/data/reconst_freq_pr3_bl4_pdn01_CelebA/fingerprinted_images/", help="Directory with images")
parser.add_argument("--err_type", type=str, default='j', help="transmission error type, accept: j => jpeg compression, g => gaussian noise, c => color jitter, b => gaussian blur")
parser.add_argument("--jpeg_q", type=int, default=50, help="jpeg compression quality factor")
parser.add_argument("--noise_std", type=float, default=25.0, help="Standard deviation for Gaussian noise")
parser.add_argument("--jitter_strength", type=float, default=0.2, help="Strength of color jitter")
parser.add_argument("--blur_radius", type=float, default=2.0, help="Radius for Gaussian blur")
args = parser.parse_args()

input_dir = args.input_dir

if args.err_type == 'j':
    output_dir = args.input_dir[:-1] + "_jpg/"
elif args.err_type == 'g':
    output_dir = args.input_dir[:-1] + "_gn/"
elif args.err_type == 'c':
    output_dir = args.input_dir[:-1] + "_cj/"
elif args.err_type == 'b':
    output_dir = args.input_dir[:-1] + "_gb/"
else:
    raise Exception("transmission error type ", args.err_type, " not supported")
    
os.makedirs(output_dir, exist_ok=True)

# 處理所有圖像
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with Image.open(input_path) as img:
            # 確保是RGB模式（部分處理需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if args.err_type == 'j':
                # JPEG壓縮
                img.save(output_path, "JPEG", quality=args.jpeg_q)

            elif args.err_type == 'g':
                # 高斯噪聲
                np_img = np.array(img, dtype=np.float32)
                noise = np.random.normal(0, args.noise_std, np_img.shape)
                noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy_img)
                noisy_img.save(output_path)

            elif args.err_type == 'c':
                # 顏色抖動
                enhancer = ImageEnhance.Color(img)
                jittered_img = enhancer.enhance(1 + random.uniform(-args.jitter_strength, args.jitter_strength))
                jittered_img.save(output_path)
                
            elif args.err_type == 'b':
                # 高斯模糊
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
                blurred_img.save(output_path)

print(f"所有圖像已處理並保存到: {output_dir}")
