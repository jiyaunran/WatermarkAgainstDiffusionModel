import os
from PIL import Image

analyzing_name = "AR"

images = [Image.open(f"{x}_{analyzing_name}.png").convert('RGB') for x in ["highest_result_comparison", "lowest_result_comparison", "highest_result_comparison_hidden", "lowest_result_comparison_hidden"]]

widths, heights = zip(*(i.size for i in images))

total_width = sum(widths) // 2
total_height = sum(heights) // 2

x_offset = 0

pack_image = Image.new('RGB', (total_width, total_height))

pack_image.paste(images[0], (0,0))
pack_image.paste(images[1], (widths[0],0))
pack_image.paste(images[2], (0,heights[0]))
pack_image.paste(images[3], (widths[0],heights[0]))

pack_image.save(f"{analyzing_name}_analyze.png")