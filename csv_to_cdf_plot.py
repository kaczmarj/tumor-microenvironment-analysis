import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import pickle
import glob

# Parameters for Path
slide_name = "668"
input_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT/{}'.format(slide_name)
points_csv = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT/{}/combined_cells.csv".format(slide_name)
patch_csv = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT/{}/combined_patches.csv".format(slide_name)
output_dir = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/KYT/{}/Hist_CDF".format(slide_name)
# output_dir = './temp/'

# Combine Cells CSV files
all_filenames = glob.glob(input_dir+'/*_cells.csv')
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(points_csv, index=False)

# Combine Patch CSV files
all_filenames_patch = glob.glob(input_dir+'/*_patches.csv')
combined_csv_patch = pd.concat([pd.read_csv(f) for f in all_filenames_patch ])
combined_csv_patch.to_csv(patch_csv, index=False)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Parameters for Histrogram
mpp = 0.34622
bin_range = 20
number_of_bin = 5
bin_range_dict = {i: 1 for i in range(number_of_bin)}
label_for_draw = ["{}-{}".format(i*bin_range,(i+1)*bin_range) for i in range(number_of_bin)]

seq_color = cm.get_cmap('inferno', number_of_bin)
color_list = [seq_color(i) for i in range(number_of_bin)]

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


def histogram_calc_patch_points(patch_points_pos, patch_points_neg):
    hist_pos = bin_range_dict.copy()
    hist_neg = bin_range_dict.copy()
    patch_points_pos = np.asarray(patch_points_pos)
    patch_points_neg = np.asarray(patch_points_neg)
    for pos, neg in zip(patch_points_pos, patch_points_neg):
        if pos>0 and pos<neg:
            if int(pos/bin_range)>number_of_bin-1:
                continue
            else:
                hist_pos[int(pos/bin_range)] += 1

        elif neg>0 and neg<pos:
            if int(neg/bin_range)>number_of_bin-1:
                continue
            else:
                hist_neg[int(neg/bin_range)] += 1

        elif pos>0 and neg>0 and pos == neg:
            if int(neg/bin_range)>number_of_bin-1:
                continue                
            else:
                hist_neg[int(neg/bin_range)] += 1
                hist_pos[int(pos/bin_range)] += 1
    return hist_pos, hist_neg

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


def histogram_calc_cell_points(neg_cell_points, pos_cell_points, cell_type, dest_dir):
    if cell_type == 'cd8':
        cell_type = 'Lymph'

    neg_cell_points = np.asarray(neg_cell_points)
    pos_cell_points = np.asarray(pos_cell_points)
    neg_cell_point_hist = bin_range_dict.copy()
    pos_cell_point_hist = bin_range_dict.copy()

    for neg_val in neg_cell_points:
        if int(neg_val/bin_range)>number_of_bin-1:
            continue
        else:
            neg_cell_point_hist[int(neg_val/bin_range)] += 1

    for pos_val in pos_cell_points:
        if int(pos_val/bin_range)>number_of_bin-1:
            continue
        else:
            pos_cell_point_hist[int(pos_val/bin_range)] += 1

    return neg_cell_point_hist, pos_cell_point_hist



def get_and_draw_histograms_cdfs(points_csv, patch_csv, dest_dir, mpp = 0.34622):
    # Read Cell Points and Patch Points from CSV
    cell_points = pd.read_csv(points_csv)
    patch_points = pd.read_csv(patch_csv)

    # Transform distance from pixel distance to micron distance
    cell_points.loc[:, "dist_to_marker_neg"] *= mpp
    cell_points.loc[:, "dist_to_marker_pos"] *= mpp
    patch_points.loc[:, "dist_to_marker_neg"] *= mpp
    patch_points.loc[:, "dist_to_marker_pos"] *= mpp
    cell_types = ["cd8", "cd16", "cd163"]
    pos_patch_hist, neg_patch_hist = histogram_calc_patch_points(patch_points.loc[:, "dist_to_marker_pos"], patch_points.loc[:, "dist_to_marker_neg"])
    draw_hist(neg_patch_hist, pos_patch_hist, 'Patch', 'Distribution', dest_dir)
    patch_hist_dict = {'neg_patch_hist' : neg_patch_hist, 'pos_patch_hist': pos_patch_hist}
    pickle.dump(patch_hist_dict, open(os.path.join(dest_dir,'patch.pkl'), "wb"))

    for cell_type in cell_types:
        neg_cell_point_hist, pos_cell_point_hist = histogram_calc_cell_points(cell_points.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_neg"].dropna(), cell_points.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_pos"].dropna(), cell_type, output_dir)
        if cell_type == 'cd8':
            cell_type = 'Lymph'
        cell_hist_dict = {'neg_cell_point_hist' : neg_cell_point_hist, 'pos_cell_point_hist': pos_cell_point_hist}
        pickle.dump(cell_hist_dict,  open(os.path.join(dest_dir,'{}.pkl'.format(cell_type)), "wb"))

        # Normalization by Avg Cell Area
        neg_hist_normalized_by_area, pos_hist_normalized_by_area = normalize_by_avg_cell_area(neg_cell_point_hist, pos_cell_point_hist, cell_type)
        draw_hist(neg_hist_normalized_by_area, pos_hist_normalized_by_area, cell_type, 'normalized_by_Avg_Area', dest_dir)
        draw_cdf(neg_hist_normalized_by_area, pos_hist_normalized_by_area, dest_dir, '{}_normalized_by_Avg_Area'.format(cell_type))

        # Normalization by Environment Influence
        neg_hist_normalized_by_env, pos_hist_normalized_by_env = normalize_by_environment_influence(neg_cell_point_hist, pos_cell_point_hist, neg_patch_hist, pos_patch_hist)
        draw_hist(neg_hist_normalized_by_env, pos_hist_normalized_by_env, cell_type, 'normalized_by_Influence_Area', dest_dir)
        draw_cdf(neg_hist_normalized_by_env, pos_hist_normalized_by_env, dest_dir, '{}_normalized_by_Influence_Area'.format(cell_type))

get_and_draw_histograms_cdfs(points_csv, patch_csv, output_dir, mpp = mpp)