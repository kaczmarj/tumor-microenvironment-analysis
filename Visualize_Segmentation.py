import os
import numpy as np
from skimage import io;
import glob;
import sys;
from scipy.misc import imresize

text_2_rgb_id = {
    'k17p': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (139, 69, 19)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (255, 0, 255)],
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (255, 255, 0)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (255, 0, 0)],
    'cd163': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (0, 255, 0)],
    'k17n': [(100, 80, 80), 6, (0, 0, 0, 0, 0, 0, 1, 0), (0, 128, 128)],
    'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
    'background2': [(31,50,222), -1, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
}


def visualize_all_stains_in_single_mask(argmax_filepath, out_dir):
    argmax_arr = np.load(argmax_filepath,allow_pickle=True);
    basefilename = os.path.splitext(os.path.basename(argmax_filepath))[0];    
    img_mask = np.zeros((argmax_arr.shape[0],argmax_arr.shape[1],3))
    for stain_name in text_2_rgb_id.keys():
        stain_indx = text_2_rgb_id[stain_name][1]+1
        stain_rgb = text_2_rgb_id[stain_name][3]
        img_mask[np.where(argmax_arr==stain_indx)] = stain_rgb
    io.imsave(os.path.join(out_dir, basefilename +'_pred.png'), img_mask.astype(np.uint8));


argmax_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/Div_Data/'
out_dir_processed = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/Div_Data/'

argmax_files = glob.glob(argmax_dir+'*.npy')


for argmax_filepath in argmax_files:
    print('argmax_filepath',argmax_filepath)
    visualize_all_stains_in_single_mask(argmax_filepath, out_dir_processed)


