import os
import numpy as np
from skimage import io
import glob
import sys
from scipy.misc import imresize

text_2_rgb_id = {
    "k17p": [(227, 137, 26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (139, 69, 19)],
    "cd8": [(43, 185, 253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (255, 0, 255)],
    "cd16": [(255, 255, 255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (255, 255, 0)],
    "cd4": [(0, 255, 255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0)],
    "cd3": [(240, 13, 218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (255, 0, 0)],
    "cd163": [(13, 240, 18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (0, 255, 0)],
    "k17n": [(100, 80, 80), 6, (0, 0, 0, 0, 0, 0, 1, 0), (0, 128, 128)],
    "background": [(31, 50, 222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
    "background2": [(31, 50, 222), -1, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
}


def visualize_all_stains_in_single_mask(imgfilepath, argmax_filepath, out_dir):
    argmax_arr = np.load(argmax_filepath, allow_pickle=True)
    img = io.imread(imgfilepath)
    # #argmax_arr = imresize(argmax_arr,(int(img.shape[0]),int(img.shape[1])),interp='nearest', mode='F').astype(np.uint8);
    basefilename = os.path.splitext(os.path.basename(argmax_filepath))[0]
    img_mask = np.zeros((img.shape[0], img.shape[1], 3))
    for stain_name in text_2_rgb_id.keys():
        stain_indx = text_2_rgb_id[stain_name][1] + 1
        stain_rgb = text_2_rgb_id[stain_name][3]
        img_mask[np.where(argmax_arr == stain_indx)] = stain_rgb
    io.imsave(
        os.path.join(out_dir, basefilename + "pred.png"), img_mask.astype(np.uint8)
    )


im_dir = "./Data2"
conc_dir = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Automated_Merged_AE_UNET_WSI/Combined"

argmax_dir = "./Data2"
out_dir_processed = "./Data2"

if not os.path.isdir(out_dir_processed):
    os.mkdir(out_dir_processed)

im_files = glob.glob(os.path.join(im_dir, "*.png"))
# im_files = glob.glob(os.path.join(im_dir,'668_31000_27000_1000_1000.png'))


for im_filepath in im_files:
    print("im_filepath", im_filepath)
    # conc_filepath = glob.glob(os.path.join(conc_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))[0]
    argmax_filepath = glob.glob(
        os.path.join(argmax_dir, "*" + os.path.basename(im_filepath)[:-4] + "*.npy")
    )[0]
    # print('conc_filepath',conc_filepath)
    print("argmax_filepath", argmax_filepath)
    visualize_all_stains_in_single_mask(im_filepath, argmax_filepath, out_dir_processed)
