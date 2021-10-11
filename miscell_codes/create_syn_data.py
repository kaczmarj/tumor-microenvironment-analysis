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

input_dir = Path("/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/N9430-B11/ROI_1")
patch_npy_files = input_dir.glob("*.npy")
tumor_microenv = 100
mpp = 0.34622
xoff, yoff = 35917, 23945
color_negative = (0, 128, 128)
color_positive = (139, 69, 19)
patch_size = 73


merged_image_path = input_dir / "merged_image.png"
merged_image = cv2.imread(str(merged_image_path))
merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

canvas = np.zeros(merged_image.shape,dtype=np.uint8)
canvas += 255

patches, _ = tm.LoaderV1(patch_npy_files, [], background=0, marker_positive=1, marker_negative=7,)()
canvas = tm.cv2_add_patch_exteriors(canvas, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=5, color_negative = color_negative, color_positive = color_positive)


tumor_patches = [p.polygon for p in patches if p.patch_type == tm.PatchType.TUMOR]
tumor_geom = unary_union(tumor_patches)
tumor_exterior = tm._get_exterior_of_geom(tumor_geom)
tumor_polygon = []
for line in tumor_exterior:
    new_poly = Polygon(line.coords)
    tumor_polygon.append(new_poly)

multi_tumor_polygon = MultiPolygon(tumor_polygon)

cells: tm.Cells = []

stain_to_color = {
    "cd8": (255, 0, 255),
    "cd16": (255, 255, 0),
    "cd4": (0, 0, 0),
    "cd3": (0, 0, 255),
    "cd163": (0, 255, 0),
}


def generate_random_cells(canvas, number_of_cells, cell_radius, cell_type):
    height, width = canvas.shape[:-1]
    cell_count = 0
    for i in range(number_of_cells):
        rand_x, rand_y = random.randint(0, height), random.randint(0, width)
        random_point = Point(rand_x + xoff, rand_y + yoff)
        circle = random_point.buffer(cell_radius)
        polygon = Polygon(circle.exterior.coords)
        if not multi_tumor_polygon.buffer(10).intersects(polygon):
            cv2.circle(canvas, center=(rand_x, rand_y), radius=cell_radius, color=stain_to_color[cell_type], thickness=-1)
            cells.append(tm.Cell(polygon=polygon, cell_type=cell_type, uuid=0,))
            cell_count += 1
    return cell_count


lymph_cell_radius = int(4/mpp)
macrophages_cell_radius = int(8/mpp)
lymph_count = generate_random_cells(canvas, 300, lymph_cell_radius, 'cd8')
cd16_count = generate_random_cells(canvas, 300, macrophages_cell_radius, 'cd16')
cd163_count = generate_random_cells(canvas, 300, macrophages_cell_radius, 'cd163')

print(lymph_count, cd16_count, cd163_count)

# cv2.imwrite('./temp/border_with_random_points.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


influence_point_path = Path("./temp/35917-23945_patches.csv")
output_dir = Path("./temp")

brown_gradient = [(118, 92, 71), (146, 115, 89), (196, 156, 124), (246, 199, 160), (255, 213, 178)]
cyan_gradient = [(5, 61, 139), (17, 100, 176), (41, 139, 200), (63, 183, 219), (92, 213, 232)]
red_gradient = [(241, 29, 40), (253, 58, 45), (254, 97, 44), (255, 135, 44), (255, 161, 44)]
line_thickness = 2

influence_points = pd.read_csv(influence_point_path)
influence_points.loc[:, "dist_to_marker_neg"] *= mpp
influence_points.loc[:, "dist_to_marker_pos"] *= mpp

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
