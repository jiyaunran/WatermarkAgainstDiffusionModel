import numpy as np
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

sample_name = args.input_path
sample_dir = sample_name.split('/')[:-1]
sample_dir = os.path.join('/', *sample_dir, 'gen_img')

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

images = np.load(sample_name)
print(images['arr_0'].shape)
images = images['arr_0']

for i in range(len(images)):
    img = Image.fromarray(images[i])
    name = os.path.join(args.output_path, f'{i}.png')
    img.save(name)
    print(name + ' saved')