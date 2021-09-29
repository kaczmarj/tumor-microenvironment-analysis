import os
import glob
import pandas as pd

slide_name = '668'
input_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT/{}'.format(slide_name)

all_filenames = glob.glob(input_dir+'*_cells.csv')
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(os.path.join(input_dir,"combined_cells.csv"), index=False)

all_filenames_patch = glob.glob(input_dir+'*_patches.csv')
combined_csv_patch = pd.concat([pd.read_csv(f) for f in all_filenames_patch ])
combined_csv_patch.to_csv(os.path.join(input_dir,"combined_patches.csv"), index=False)