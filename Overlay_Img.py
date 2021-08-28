from pathlib import Path
import tumor_microenv as tm
import cv2
import itertools
import random
import matplotlib.pyplot as plt
import shapely.wkt

data_root = Path("/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Micro_Env_Data/Data")
patch_paths = list(data_root.glob("*.npy"))
json_paths = list(data_root.glob("*.json"))

loader = tm.LoaderV1(
    patch_paths=patch_paths, 
    cells_json=json_paths, 
    background=0, 
    marker_positive=1, 
    marker_negative=7,
    marker_neg_thresh=0.3,
)
patches, cells = loader()

# cells = [c for c in cells if c.cell_type in {"cd3", "cd4", "cd8", "cd16", "cd163"}]

# tm.run_spatial_analysis(
#     patches, 
#     cells, 
#     microenv_distances=[100], 
#     mpp=0.34622,
#     output_path="output.csv")

def gen_random_point_per_cell(points_data):
    for uuid, g in itertools.groupby(points_data, lambda p: p.cell_uuid):
        yield random.choice(list(g))

points_data = tm.read_point_csv("output.csv")
random_points_per_cell = list(gen_random_point_per_cell(points_data))

stain_to_color = {'cd8': (255, 0, 255), 'cd16': (0, 255, 255), 'cd4': (0, 0, 0), 'cd3': (0, 0, 255), 'cd163': (0, 255, 0)}


image = cv2.imread('merged_image.png')


for patch in patches:
    x = int(patch.polygon.exterior.xy[0][3]) - 35917
    y = int(patch.polygon.exterior.xy[1][3]) - 23945
    top_left, bottom_right = (x,y), (x+146,y+146)
    thickness = 5
    if patch.biomarker_status.value == 1:
        color = (19, 69, 139) #(139, 69, 19) 
        image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    elif patch.biomarker_status.value == 2:
        color = (128, 128, 0)  #(0, 128, 128)
        image = cv2.rectangle(image, top_left, bottom_right, color, thickness)

for point_data in random_points_per_cell:
    point = shapely.wkt.loads(point_data.point)
    point = (int(point.x) - 35917, int(point.y) - 23945)
    line_to_pos = shapely.wkt.loads(point_data.line_to_marker_pos)
    line_to_pos_start = (int(line_to_pos.coords.xy[0][0]) - 35917, int(line_to_pos.coords.xy[1][0]) - 23945)
    line_to_pos_end = (int(line_to_pos.coords.xy[0][1]) - 35917, int(line_to_pos.coords.xy[1][1]) - 23945)
    line_to_neg = shapely.wkt.loads(point_data.line_to_marker_neg)
    line_to_neg_start = (int(line_to_neg.coords.xy[0][0]) - 35917, int(line_to_neg.coords.xy[1][0]) - 23945)
    line_to_neg_end = (int(line_to_neg.coords.xy[0][1]) - 35917, int(line_to_neg.coords.xy[1][1]) - 23945)
    point_color = stain_to_color[point_data.cell_type]
    pos_line_color = (19, 69, 139)
    neg_line_color = (128, 128, 0)
    image = cv2.circle(image, point, 3, point_color, -1)
    image = cv2.line(image, line_to_pos_start, line_to_pos_end, pos_line_color, 1)
    image = cv2.line(image, line_to_neg_start, line_to_neg_end, neg_line_color, 1)
    print(i)

cv2.imwrite('overlayed_full.png',image)