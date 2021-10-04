import glob
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm


# slides_list = ['583', '668', '887', '925', '930'] # High K17 List
slides_list = ['3372', '3739', '1092A', '1197A', '1282B']

input_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT'
dest_dir = './temp'
cell_types = ['Lymph', 'cd16', 'cd163']

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

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


def draw_cdf(neg_dict, pos_dict, dest_dir, png_file_name):
    plt.clf()
    x = label_for_draw
    y = np.asarray(list(neg_dict.values()))
    y = y / y.sum()
    y = np.cumsum(y)
    plt.plot(x,y, label='CDF for K17-')
    y = np.asarray(list(pos_dict.values()))
    y = y / y.sum()
    y = np.cumsum(y)
    plt.plot(x,y, label='CDF for K17+')
    plt.title("{}".format(png_file_name))
    plt.xlabel("Distances in Micrometer")
    plt.ylabel("Cumulative Distribution")
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.yticks(np.arange(0,1,0.1))
    plt.savefig(os.path.join(dest_dir,'{}_cdf.png'.format(png_file_name)), dpi=300, format='png', bbox_inches='tight')


def draw_hist(neg_hist, pos_hist, cell_type, normalization, dest_dir):
    plt.clf()
    labels = label_for_draw
    neg_y = neg_hist.values()
    pos_y = pos_hist.values()
    max_neg = max(neg_y)
    max_pos = max(pos_y)
    max_freq = max(max_neg, max_pos)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    neg_rects = ax.bar(x - width/2, neg_y, width, label='K17-', edgecolor='Black')
    pos_rects = ax.bar(x + width/2, pos_y, width, label='K17+', edgecolor='Black')
    show_values_on_bars(ax, max_freq)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Distances in Microns')
    ax.set_title('{} distributon {}'.format(cell_type, normalization))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(dest_dir,'{}_{}_hist.png'.format(cell_type, normalization)), dpi=300, format='png', bbox_inches='tight')


def normalize_by_avg_cell_area(neg_hist, pos_hist, cell_type):
    neg_hist_normalized = neg_hist.copy()
    pos_hist_normalized = pos_hist.copy()
    if cell_type == 'Lymph':
        denom = ((8*8)*3.1416)/0.34622
    else:
        denom = ((16*16)*3.1416)/0.34622

    for key in neg_hist.keys():
        neg_hist_normalized[key] = int(neg_hist_normalized[key] / denom)
        pos_hist_normalized[key] = int(pos_hist_normalized[key] / denom)
    return neg_hist_normalized, pos_hist_normalized

def normalize_by_environment_influence(neg_hist, pos_hist, neg_patch_hist, pos_patch_hist):
    neg_hist_normalized = {k: int(neg_hist[k] / neg_patch_hist[k]) for k in neg_patch_hist}
    pos_hist_normalized = {k: int(pos_hist[k] / pos_patch_hist[k]) for k in pos_patch_hist}
    return neg_hist_normalized, pos_hist_normalized


for cell_type in cell_types:
    total_neg_cell_point_hist = bin_range_dict.copy()
    total_pos_cell_point_hist = bin_range_dict.copy()
    total_pos_patch_hist = bin_range_dict.copy()
    total_neg_patch_hist = bin_range_dict.copy()
    for slide in slides_list:
        cell_filename = input_dir+'/'+slide+'/Hist_CDF/'+cell_type+'.pkl'
        patch_filename = input_dir+'/'+slide+'/Hist_CDF/patch.pkl'
        cell_data = pickle.load(open(cell_filename,"rb"))
        patch_data = pickle.load(open(patch_filename,"rb"))
        total_neg_cell_point_hist = add_two_dict(total_neg_cell_point_hist, cell_data['neg_cell_point_hist'])
        total_pos_cell_point_hist = add_two_dict(total_pos_cell_point_hist, cell_data['pos_cell_point_hist'])
        total_neg_patch_hist = add_two_dict(total_neg_patch_hist, patch_data['neg_patch_hist'])
        total_pos_patch_hist = add_two_dict(total_pos_patch_hist, patch_data['pos_patch_hist'])
    
    # Normalization by Avg Cell Area
    neg_hist_normalized_by_area, pos_hist_normalized_by_area = normalize_by_avg_cell_area(total_neg_cell_point_hist, total_pos_cell_point_hist, cell_type)
    draw_hist(neg_hist_normalized_by_area, pos_hist_normalized_by_area, cell_type, 'normalized_by_Avg_Area', dest_dir)
    draw_cdf(neg_hist_normalized_by_area, pos_hist_normalized_by_area, dest_dir, '{}_normalized_by_Avg_Area'.format(cell_type))

    # Normalization by Environment Influence
    neg_hist_normalized_by_env, pos_hist_normalized_by_env = normalize_by_environment_influence(total_neg_cell_point_hist, total_pos_cell_point_hist, total_neg_patch_hist, total_pos_patch_hist)
    draw_hist(neg_hist_normalized_by_env, pos_hist_normalized_by_env, cell_type, 'normalized_by_Influence_Area', dest_dir)
    draw_cdf(neg_hist_normalized_by_env, pos_hist_normalized_by_env, dest_dir, '{}_normalized_by_Influence_Area'.format(cell_type))