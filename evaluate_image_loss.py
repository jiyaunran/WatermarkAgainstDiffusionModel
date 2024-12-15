import argparse
import os
import glob
from PIL import Image
import torch
import lpips
from tqdm import tqdm
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--data_path1", type=str, required=True, help="First data directory")
parser.add_argument("--data_path2", type=str, required=True, help="Second data directory")
parser.add_argument("--image_resolution", type=int, required=True, help="Image resolution")
parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_function = lpips.LPIPS(net='vgg').to(device)

def calculate_psnr(img1, img2, max_pixel=255.0):
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同时，PSNR 是无穷大
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

if __name__ == "__main__":
    image_names = glob.glob(os.path.join(args.data_path1, "*.png"))
    
    l2_loss = 0
    linf_loss = 0
    lpips_loss = 0
    psnr_total = 0
    i = 0
    
    with torch.no_grad():
        for image_name in tqdm(image_names):
            # 加载原始图像
            org_image = Image.open(image_name)
            org_image = np.asarray(org_image).astype(np.float32)
            
            # 加载嵌入图像
            embed_name = os.path.join(args.data_path2, os.path.basename(image_name))
            embed_image = Image.open(embed_name)
            embed_image = np.asarray(embed_image).astype(np.float32)
            
            # 计算 L2 Loss
            l2_diff = np.sqrt(np.mean(np.square(embed_image - org_image)))
            
            # 计算 L∞ Loss
            linf_diff = np.max(np.abs(embed_image - org_image))
            
            # 计算 LPIPS Loss
            embed_image_tensor = torch.from_numpy(np.moveaxis(embed_image, -1, 0)).to(device).reshape(1, 3, *embed_image.shape[:2])
            org_image_tensor = torch.from_numpy(np.moveaxis(org_image, -1, 0)).to(device).reshape(1, 3, *org_image.shape[:2])
            lpips_diff = lpips_function(embed_image_tensor, org_image_tensor).item()
            
            # 计算 PSNR
            psnr = calculate_psnr(org_image, embed_image)
            
            # 累加损失
            l2_loss += l2_diff
            linf_loss += linf_diff
            lpips_loss += lpips_diff
            psnr_total += psnr
            
            i += 1
            if i >= 5000:
                break
    
    # 计算平均损失
    l2_loss /= i
    linf_loss /= i
    lpips_loss /= i
    psnr_total /= i
    
    # 打印结果
    print("L2 loss: ", l2_loss)
    print("L∞ loss: ", linf_loss)
    print("LPIPS loss: ", lpips_loss)
    print("Average PSNR: ", psnr_total)
