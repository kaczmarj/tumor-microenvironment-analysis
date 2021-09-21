import numpy as np
import matplotlib.pyplot as plt

lymph_file = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/cdf_values_cd8.npz'
cd16_file = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/cdf_values_cd16.npz'
cd163_file = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/cdf_values_cd163.npz'
output = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result'

def plot_cdfs(arr,output_dir,cell_type):
    plt.figure()
    plt.plot(arr['bins_pos'][1:], arr['cdf_pos'], label="biomarker positive")
    plt.plot(arr['bins_pos'][1:], arr['cdf_neg'], label="biomarker negative")
    plt.legend()
    plt.xlabel("Distance (micrometer)")
    plt.title("CDFs of distances from {} to M+/M-.".format(cell_type))
    plt.savefig('{}/{}.png'.format(output_dir,cell_type))

lymph = np.load(lymph_file)
cd16 = np.load(cd16_file)
cd163 = np.load(cd163_file)

plot_cdfs(lymph,output,'Lymph')
plot_cdfs(cd16, output, 'CD16')
plot_cdfs(cd163, output, 'CD163')