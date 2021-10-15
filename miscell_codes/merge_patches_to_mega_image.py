import glob
import os
from PIL import Image

region_of_interest = "ROI_1"
patch_files = glob.glob("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3739/{}/*_73_73.png".format(region_of_interest))
patch_size = 73

dict_for_sorting = {}

for file in patch_files:
    basename = os.path.basename(file)
    offset_x, offset_y, _, _ = basename.split('_')
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    if offset_x not in dict_for_sorting.keys():
        dict_for_sorting[offset_x] = [offset_y]
    else:
        dict_for_sorting[offset_x].append(offset_y)

offset_x = sorted(dict_for_sorting.keys())[0]
offset_y = sorted(dict_for_sorting[offset_x])[0]

multiple = len(dict_for_sorting.keys())

with open("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3739/{}/offset.txt".format(region_of_interest),"w") as f:
    f.write("{} {}".format(offset_x,offset_y))
    f.close()

merged_image = Image.new("RGB", (multiple * patch_size, multiple * patch_size), (250, 250, 250))

for patch_file in patch_files:
    basename = os.path.basename(patch_file)
    x, y, _, _ = basename.split("_")
    x, y = int(x) - offset_x, int(y) - offset_y
    patch_img = Image.open(patch_file)
    merged_image.paste(patch_img, (x, y))

merged_image.save("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3739/{}/merged_image.png".format(region_of_interest), "PNG")
