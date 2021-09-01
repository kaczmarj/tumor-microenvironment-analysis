import glob
import os
from PIL import Image

patch_files = glob.glob("./Data2/*_146_146.png")
min_x = 35917
min_y = 23945
merged_image = Image.new("RGB", (42 * 146, 42 * 146), (250, 250, 250))

for patch_file in patch_files:
    basename = os.path.basename(patch_file)
    x, y, _, _ = basename.split("_")
    x, y = int(x) - min_x, int(y) - min_y
    patch_img = Image.open(patch_file)
    merged_image.paste(patch_img, (x, y))

merged_image.save("merged_image.png", "PNG")
