"""Plot overlays of tumor boundaries, cells, and distances on top of whole slide."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math

points_csv = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/combined_csv.csv"
patch_csv = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/combined_patches.csv"
output_dir = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/Tumor_Micro_Result/"
mpp = 0.34622
bin_range = 10
number_of_bin = 11
bin_range_dict = {i: 1 for i in range(number_of_bin)}
label_for_draw = ["{}-{}".format(i*bin_range,(i+1)*bin_range) for i in range(number_of_bin)]
label_for_draw[number_of_bin-1] = "{}-INF".format((number_of_bin-1) * bin_range)

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


def draw_cdf_using_dict(target_dict, dest_dir, png_file_name):
    plt.clf()
    x = label_for_draw
    y = np.asarray(list(target_dict.values()))
    y = y / y.sum()
    y = np.cumsum(y)
    plt.plot(x,y)
    plt.title("{}".format(png_file_name))
    plt.xlabel("Distances in Micrometer")
    plt.ylabel("Cumulative Frequency")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(dest_dir, png_file_name+'.png'), dpi=300, format='png', bbox_inches='tight')


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


def histogram_calc_patch_points(patch_points_pos, patch_points_neg):
    hist_pos = bin_range_dict.copy()
    hist_neg = bin_range_dict.copy()
    patch_points_pos = np.asarray(patch_points_pos)
    patch_points_neg = np.asarray(patch_points_neg)
    for pos, neg in zip(patch_points_pos, patch_points_neg):
        if pos>0 and pos<neg:
            if int(pos/bin_range)>number_of_bin-1:
                hist_pos[number_of_bin-1] += 1
            else:
                hist_pos[int(pos/bin_range)] += 1

        elif neg>0 and neg<pos:
            if int(neg/bin_range)>number_of_bin-1:
                hist_neg[number_of_bin-1] += 1
            else:
                hist_neg[int(neg/bin_range)] += 1

        elif pos>0 and neg>0 and pos == neg:
            if int(neg/bin_range)>number_of_bin-1:
                hist_neg[number_of_bin-1] += 1
                hist_pos[number_of_bin-1] += 1                
            else:
                hist_neg[int(neg/bin_range)] += 1
                hist_pos[int(pos/bin_range)] += 1
    draw_hist_using_dict(hist_pos,'./temp','Patches_Impacted_by_M+')
    draw_hist_using_dict(hist_neg,'./temp','Patches_Impacted_by_M-')
    return hist_pos, hist_neg


def histogram_calc_cell_points(values, patch_hist, cell_type, marker_type):
    if cell_type == 'cd8':
        cell_type = 'Lymph'

    cell_point_hist = bin_range_dict.copy()
    values = np.asarray(values)
    for val in values:
        if int(val/bin_range)>number_of_bin-1:
            cell_point_hist[number_of_bin-1] += 1
        else:
            cell_point_hist[int(val/bin_range)] += 1
    draw_hist_using_dict(cell_point_hist,'./temp','{}_dist_from_marker_{}'.format(cell_type,marker_type))
    draw_cdf_using_dict(cell_point_hist,'./temp','cumulative_{}_dist_from_marker_{}'.format(cell_type,marker_type))
    normalized_cell_point_hist = {k: int(cell_point_hist[k] / patch_hist[k]) for k in patch_hist}
    draw_hist_using_dict(normalized_cell_point_hist,'./temp','normalized_{}_dist_from_marker_{}'.format(cell_type,marker_type))
    draw_cdf_using_dict(normalized_cell_point_hist,'./temp','normalized_cumulative_{}_dist_from_marker_{}'.format(cell_type,marker_type))


def get_cdf(points_csv: Path, patch_csv: Path, mpp: float = 0.34622):
    cell_points = pd.read_csv(points_csv)
    patch_points = pd.read_csv(patch_csv)
    cell_points.loc[:, "dist_to_marker_neg"] *= mpp
    cell_points.loc[:, "dist_to_marker_pos"] *= mpp
    patch_points.loc[:, "dist_to_marker_neg"] *= mpp
    patch_points.loc[:, "dist_to_marker_pos"] *= mpp
    cell_types = ["cd8", "cd16", "cd163"]
    hist_pos, hist_neg = histogram_calc_patch_points(patch_points.loc[:, "dist_to_marker_pos"], patch_points.loc[:, "dist_to_marker_neg"])
    for cell_type in cell_types:
        histogram_calc_cell_points(cell_points.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_neg"].dropna(), hist_neg, cell_type, "M-")
        histogram_calc_cell_points(cell_points.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_pos"].dropna(), hist_pos, cell_type, "M+")


get_cdf(points_csv=points_csv, patch_csv=patch_csv, mpp = mpp)