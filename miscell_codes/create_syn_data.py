import glob
import numpy as np
import tumor_microenv as tm
import os
import cv2
from shapely.geometry import base as _base_geometry
import typing as ty
import enum
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import nearest_points
import pickle
from pathlib import Path
import random
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.geometry import MultiPolygon
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import pickle
import math

input_dir = Path("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3372/ROI_1")
output_dir = Path("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3372/ROI_1/Analysis")
count_dir = Path("/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/3372/ROI_1/Analysis/real/")
patch_npy_files = input_dir.glob("*.npy")
merged_image_path = input_dir / "merged_image.png"
offset_path = input_dir / "offset.txt"
mpp = 0.34622
tumor_microenv = 100
mpp = 0.34622
xoff, yoff = map(int, offset_path.read_text().split())
color_negative = (0, 128, 128)
color_positive = (139, 69, 19)
patch_size = 73
influence_point_path = str(output_dir / f"{xoff}-{yoff}_patches.csv")

merged_image = cv2.imread(str(merged_image_path))
merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

canvas = np.zeros(merged_image.shape,dtype=np.uint8)
canvas += 255

patches, _ = tm.LoaderV1(patch_npy_files, [], background=0, marker_positive=1, marker_negative=7, tumor_threshold=.15)()
canvas = tm.cv2_add_patch_exteriors(canvas, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=5, color_negative = color_negative, color_positive = color_positive)
# merged_image = tm.cv2_add_patch_exteriors(merged_image, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=5, color_negative = color_negative, color_positive = color_positive)
# cv2.imwrite(str(output_dir / f'original_image_with_border.png'), cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite(str(output_dir / f'image_with_border.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


influence_points = pd.read_csv(influence_point_path)
influence_points.loc[:, "dist_to_marker_neg"] *= mpp
influence_points.loc[:, "dist_to_marker_pos"] *= mpp
rand_range = len(influence_points)

cells: tm.Cells = []

stain_to_color = {
    "cd8": (255, 0, 255),
    "cd16": (255, 255, 0),
    "cd4": (0, 0, 0),
    "cd3": (0, 0, 255),
    "cd163": (0, 255, 0),
}


def get_count(count_dir, cell_type):
    file_name = str(count_dir / cell_type)
    dist = pickle.load(open(file_name, 'rb'))
    neg_hist_normalized = dist['neg_cell_point_hist']
    pos_hist_normalized = dist['pos_cell_point_hist']
    if cell_type == 'Lymph.pkl':
        denom = (4/mpp)*(4/mpp)*3.1416
    else:
        denom = (8/mpp)*(8/mpp)*3.1416
    for key in neg_hist_normalized.keys():
        neg_hist_normalized[key] = int(math.ceil(neg_hist_normalized[key] / denom))
        pos_hist_normalized[key] = int(math.ceil(pos_hist_normalized[key] / denom))
    total_count = sum(neg_hist_normalized.values()) + sum(pos_hist_normalized.values())
    return total_count

lymph_count = get_count(count_dir, 'Lymph.pkl')
cd16_count = get_count(count_dir, 'cd16.pkl')
cd163_count = get_count(count_dir, 'cd163.pkl')


def generate_random_cells(canvas, number_of_cells, cell_radius, cell_type):
    cell_count = 0
    while cell_count<number_of_cells:
        rand_index = np.random.randint(0,rand_range)
        pos_dist = influence_points['dist_to_marker_pos'][rand_index]
        neg_dist = influence_points['dist_to_marker_neg'][rand_index]
        if (pos_dist>= 25 and pos_dist <= 75) or (neg_dist >= 25 and neg_dist <= 75):
            random_point = loads(influence_points['point'][rand_index])
            rand_x = np.random.randint(0,patch_size//2)
            rand_y = np.random.randint(0,patch_size//2)
            if np.random.randint(0,2) == 1:
                rand_x = -rand_x
            if np.random.randint(0,2) == 1:
                rand_y = -rand_y
            random_point = translate(random_point, xoff=rand_x, yoff=rand_y)
            circle = random_point.buffer(cell_radius)
            polygon = Polygon(circle.exterior.coords)
            cells.append(tm.Cell(polygon=polygon, cell_type=cell_type, uuid=0,))
            random_point = translate(random_point, xoff=-xoff, yoff=-yoff)
            center_x, center_y = random_point.coords.xy
            center_x, center_y = int(center_x[0]), int(center_y[0])
            cv2.circle(canvas, center=(center_x, center_y), radius=cell_radius, color=stain_to_color[cell_type], thickness=-1)
            cell_count += 1


lymph_cell_radius = int(4/mpp)
macrophages_cell_radius = int(8/mpp)
generate_random_cells(canvas, lymph_count, lymph_cell_radius, 'cd8')
generate_random_cells(canvas, cd16_count, macrophages_cell_radius, 'cd16')
generate_random_cells(canvas, cd163_count, macrophages_cell_radius, 'cd163')

with open(output_dir / "random_point_count.txt", "w") as f:
    f.write("Lymph - {}\n".format(lymph_count))
    f.write("CD16 - {}\n".format(cd16_count))
    f.write("CD163 - {}".format(cd163_count))
    f.close()

tm.run_spatial_analysis(patches=patches, cells=cells, microenv_distances=[tumor_microenv], mpp=mpp, output_path=output_dir / f"{xoff}-{yoff}_virtual_cells.csv", output_patch_path=output_dir / f"{xoff}-{yoff}_patches.csv", progress_bar=True)


brown_gradient = [(118, 92, 71), (146, 115, 89), (196, 156, 124), (246, 199, 160), (255, 213, 178)]
cyan_gradient = [(5, 61, 139), (17, 100, 176), (41, 139, 200), (63, 183, 219), (92, 213, 232)]
red_gradient = [(241, 29, 40), (253, 58, 45), (254, 97, 44), (255, 135, 44), (255, 161, 44)]
line_thickness = 2

for index, row in influence_points.iterrows():
    point = loads(row[0])
    point = translate(point, -xoff, -yoff)
    midpoint_x, midpoint_y = point.coords.xy
    midpoint = (int(midpoint_x[0]),int(midpoint_y[0]))
    point = translate(point, -(patch_size//2), -(patch_size//2))
    neg_distance = float(row[4])
    pos_distance = float(row[5])
    neg_line = loads(row[6])
    neg_line = translate(neg_line, -xoff, -yoff)
    pos_line = loads(row[7])
    pos_line = translate(pos_line, -xoff, -yoff)
    x, y = point.coords.xy
    x, y = int(x[0]), int(y[0])
    start_point = (x, y)
    end_point = (x + patch_size, y + patch_size)
    neg_nearest_point_x, neg_nearest_point_y = neg_line.coords.xy
    neg_nearest_point = (int(neg_nearest_point_x[1]), int(neg_nearest_point_y[1]))
    pos_nearest_point_x, pos_nearest_point_y = pos_line.coords.xy
    pos_nearest_point = (int(pos_nearest_point_x[1]), int(pos_nearest_point_y[1]))
    if pos_distance == neg_distance and pos_distance <= 100:
        color_idx = int(pos_distance/25)
        cv2.rectangle(canvas, start_point, end_point, red_gradient[color_idx], line_thickness)
    elif pos_distance < neg_distance and pos_distance <= 100:
        color_idx = int(pos_distance/25)
        cv2.rectangle(canvas, start_point, end_point, brown_gradient[color_idx], line_thickness)
    elif neg_distance < pos_distance and neg_distance <= 100:
        color_idx = int(neg_distance/25)
        cv2.rectangle(canvas, start_point, end_point, cyan_gradient[color_idx], line_thickness)

cv2.imwrite(str(output_dir / f'image_with_border_influence_area_rand_points.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
