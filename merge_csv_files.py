import os
import glob
import pandas as pd

input_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/Patches/'

all_filenames = glob.glob(input_dir+'*.csv')

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv( "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/combined_patches.csv", index=False)