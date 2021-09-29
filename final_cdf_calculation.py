import glob
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

input_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/*/CDF/'
output_dir = './temp'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

bin_range = 20
number_of_bin = 5
bin_range_dict = {i: 1 for i in range(number_of_bin)}
label_for_draw = ["{}-{}".format(i*bin_range,(i+1)*bin_range) for i in range(number_of_bin)]
seq_color = cm.get_cmap('inferno', number_of_bin)
color_list = [seq_color(i) for i in range(number_of_bin)]

def add_two_dict(dict1, dict2):
    for key in dict1.keys():
        dict1[key] += dict2[key]
    return dict1


def show_values_on_bars(axs, max_freq):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + (max_freq / 100)
            value = '{}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize='xx-small') 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def draw_hist_using_dict(target_dict, dest_dir, png_file_name, log_scale = False):
    plt.clf()
    fig, ax = plt.subplots()
    x = label_for_draw
    y = target_dict.values()
    max_freq = max(y)
    total_freq = sum(y)
    if log_scale == True:
        y = [math.log10(i) for i in y]
    ax.bar(x,y, color=color_list, edgecolor='Black')
    show_values_on_bars(ax, max_freq)
    plt.title("{}".format(png_file_name))
    plt.xlabel("Distances in Micrometer")
    if log_scale == True:
        plt.ylabel("Frequency in log10 scale")
    else:
        plt.ylabel("Frequency")
    plt.plot([], [], ' ', label="Total Num :{}".format(total_freq))
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.savefig(os.path.join(dest_dir, png_file_name+'.png'), dpi=300, format='png', bbox_inches='tight')


def draw_cdf_using_dict(neg_dict, pos_dict, dest_dir, png_file_name):
    plt.clf()
    x = label_for_draw
    y = np.asarray(list(neg_dict.values()))
    y = y / y.sum()
    y = np.cumsum(y)
    plt.plot(x,y, label='CDF for M-')
    y = np.asarray(list(pos_dict.values()))
    y = y / y.sum()
    y = np.cumsum(y)
    plt.plot(x,y, label='CDF for M+')
    plt.title("{}".format(png_file_name))
    plt.xlabel("Distances in Micrometer")
    plt.ylabel("Cumulative Distribution")
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.yticks(np.arange(0,1,0.1))
    plt.savefig(os.path.join(dest_dir, png_file_name+'.png'), dpi=300, format='png', bbox_inches='tight')

def normalize_cell_area(pixel_hist, cell_type):
    pixel_hist_copy = pixel_hist.copy()
    if cell_type == 'Lymph':
        denom = ((8*8)*3.1416)/0.34622
    else:
        denom = ((16*16)*3.1416)/0.34622
    for key in pixel_hist.keys():
        pixel_hist_copy[key] = int(pixel_hist_copy[key] / denom)
    return pixel_hist_copy

cell_types = ['Lymph', 'cd16', 'cd163']

for cell_type in cell_types:
    cell_dir = input_dir+'{}.pkl'.format(cell_type)
    cell_files = glob.glob(cell_dir)
    print(len(cell_files))
    total_neg_cell_point_hist = bin_range_dict.copy()
    total_pos_cell_point_hist = bin_range_dict.copy()
    total_neg_normalized_cell_point_hist = bin_range_dict.copy()
    total_pos_normalized_cell_point_hist = bin_range_dict.copy()
    for i in cell_files:
        data = pickle.load(open(i,"rb"))
        total_neg_cell_point_hist = add_two_dict(total_neg_cell_point_hist, data['neg_cell_point_hist'])
        total_pos_cell_point_hist = add_two_dict(total_pos_cell_point_hist, data['pos_cell_point_hist'])
        total_neg_normalized_cell_point_hist = add_two_dict(total_neg_normalized_cell_point_hist, data['neg_normalized_cell_point_hist'])
        total_pos_normalized_cell_point_hist = add_two_dict(total_pos_normalized_cell_point_hist, data['pos_normalized_cell_point_hist'])

    total_neg_cell_point_hist = normalize_cell_area(total_neg_cell_point_hist,cell_type)
    total_pos_cell_point_hist = normalize_cell_area(total_pos_cell_point_hist,cell_type)

    draw_hist_using_dict(total_neg_cell_point_hist, output_dir, '{}_Dist_From_M-'.format(cell_type))
    draw_hist_using_dict(total_pos_cell_point_hist, output_dir, '{}_Dist_From_M+'.format(cell_type))
    draw_hist_using_dict(total_neg_normalized_cell_point_hist, output_dir, '{}_Normalized_Dist_From_M-'.format(cell_type))
    draw_hist_using_dict(total_pos_normalized_cell_point_hist, output_dir, '{}_Normalized_Dist_From_M+'.format(cell_type))

    draw_cdf_using_dict(total_neg_cell_point_hist, total_pos_cell_point_hist, output_dir, 'CDF_{}'.format(cell_type))
    draw_cdf_using_dict(total_neg_normalized_cell_point_hist, total_pos_normalized_cell_point_hist, output_dir, 'Normalized_CDF_{}'.format(cell_type))