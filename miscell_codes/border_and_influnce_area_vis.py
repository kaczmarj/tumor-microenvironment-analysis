import tumor_microenv as tm
from pathlib import Path
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.wkt import loads
from shapely.affinity import translate
from shapely.geometry import LineString

input_dir = Path("/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/N9430-B11/ROI_1")
output_dir = Path("./temp")
influence_point_path = Path("./temp/35917-23945_patches.csv")
tumor_microenv = 100
mpp = 0.34622
xoff, yoff = 35917, 23945
color_negative = (0, 128, 128)
color_positive = (139, 69, 19)
patch_size = 73

patch_npy_files = input_dir.glob("*.npy")
cell_json_files = input_dir.glob("*.json")


# patches, cells = tm.LoaderV1(patch_npy_files, cell_json_files, background=0, marker_positive=1, marker_negative=7,)()
# tm.run_spatial_analysis(patches=patches, cells=cells, microenv_distances=[tumor_microenv], mpp=mpp, output_path=Path('./temp') / f"{xoff}-{yoff}_cells.csv", output_patch_path=Path('./temp') / f"{xoff}-{yoff}_patches.csv", progress_bar=True)


merged_image_path = input_dir / "merged_image.png"
merged_image = cv2.imread(str(merged_image_path))
merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

canvas = np.zeros(merged_image.shape,dtype=np.uint8)
canvas += 255

patches, _ = tm.LoaderV1(patch_npy_files, cell_json_files, background=0, marker_positive=1, marker_negative=7,)()
# merged_image_with_border = tm.cv2_add_patch_exteriors(merged_image, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=10, color_negative = color_negative, color_positive = color_positive)
canvas = tm.cv2_add_patch_exteriors(canvas, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=10, color_negative = color_negative, color_positive = color_positive)

# cv2.imwrite(str(output_dir / f'real_image_with_tumor_boundary.png'), cv2.cvtColor(merged_image_with_border, cv2.COLOR_RGB2BGR))


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
        canvas = cv2.line(canvas, midpoint, pos_nearest_point, color_positive, thickness=line_thickness)
        canvas = cv2.line(canvas, midpoint, neg_nearest_point, color_negative, thickness=line_thickness)
    elif pos_distance < neg_distance and pos_distance <= 100:
        color_idx = int(pos_distance/25)
        cv2.rectangle(canvas, start_point, end_point, brown_gradient[color_idx], line_thickness)
        canvas = cv2.line(canvas, midpoint, pos_nearest_point, color_positive, thickness=line_thickness)
    elif neg_distance < pos_distance and neg_distance <= 100:
        color_idx = int(neg_distance/25)
        cv2.rectangle(canvas, start_point, end_point, cyan_gradient[color_idx], line_thickness)
        canvas = cv2.line(canvas, midpoint, neg_nearest_point, color_negative, thickness=line_thickness)

cv2.imwrite(str(output_dir / f'image_with_border_and_influence_area.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))