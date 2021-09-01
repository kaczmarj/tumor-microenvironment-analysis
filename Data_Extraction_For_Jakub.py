import glob
import numpy as np
import os
import libpysal
from esda.getisord import G_Local
import openslide
import datetime
from bson import json_util
import json
import pandas as pd
from shapely.geometry import Polygon
from shutil import copyfile
from scipy.misc import imresize

# Following codes are necessary for including patches that are annotated by Prof. Shroyer as Tumor.
annotation_dir = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/annot_jakub"
patch_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi_patch_145_145/N9430-B11-multires.tif/"
manifest_file = os.path.join(annotation_dir, "manifest.csv")
slide_name = "N9430-B11"
clinicaltrialsubjectid = "N9430"
imageid = "B11"
manifest = pd.read_csv(manifest_file)
jakub_roi = []

# Path parameters
segmentation_input_directory = (
    "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result/WSI/WSI_145_145/Anchor_UNET"
)
wsi_input_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi/SBU"
wsi_filename = os.path.join(wsi_input_directory, "{}-multires.tif".format(slide_name))
predicted_segmentation_files = glob.glob(
    os.path.join(
        segmentation_input_directory, "{}-multires.tif/*.npy".format(slide_name)
    )
)
patch_size = 146
stain_dict = {"k17p": 1, "cd8": 2, "cd16": 3, "cd4": 4, "cd3": 5, "cd163": 6, "k17n": 7}


# WSI parameters
oslide = openslide.OpenSlide(wsi_filename)
slide_width = int(oslide.dimensions[0])
slide_height = int(oslide.dimensions[1])
patch_width, patch_height = patch_size, patch_size
patch_width = patch_width / slide_width
patch_height = patch_height / slide_height
patch_area = patch_width * slide_width * patch_height * slide_height

json_file_path = manifest.loc[
    (manifest["clinicaltrialsubjectid"] == clinicaltrialsubjectid)
    & (manifest["imageid"] == imageid)
]["path"].values[0]

with open(
    os.path.join(annotation_dir, json_file_path)
) as f:  # Loading JSON from manifest file. JSONs contain all annotations
    json_file = json.load(f)

for annot in json_file:
    if annot["properties"]["annotations"]["notes"] == "Jakub-ROI-6000px":
        box_coordinate = annot["geometries"]["features"][0]["geometry"]["coordinates"][
            0
        ][0]
        box_x, box_y = box_coordinate
        x_add, y_add = 6000 / slide_width, 6000 / slide_height
        jakub_roi.append(
            [
                [box_x, box_y],
                [box_x + x_add, box_y],
                [box_x + x_add, box_y + y_add],
                [box_x, box_y + y_add],
            ]
        )

jakub_polygon = []

for roi in jakub_roi:
    roi_tuple = [(i[0], i[1]) for i in roi]
    jakub_polygon.append(Polygon(roi_tuple))


# This function checks whether a rectangle intersects a polygon list. We have considered shroyer tumor list and shroyer not tumor list
def check_intersection(polygon_list, rectangle):
    for polygons in polygon_list:
        if polygons.intersects(rectangle):
            return True
    return False


for predicted_segmentation_file in predicted_segmentation_files:
    base_name = os.path.basename(predicted_segmentation_file)
    print(base_name)
    x, y, patch_original_size, patch_resized_size, _ = base_name.split("_")
    x, y, patch_resized_size = int(x), int(y), int(patch_resized_size)
    rectangle_coordinates = [
        (x / slide_width, y / slide_height),
        ((x + patch_resized_size) / slide_width, y / slide_height),
        (
            (x + patch_resized_size) / slide_width,
            (y + patch_resized_size) / slide_height,
        ),
        ((x) / slide_width, (y + patch_resized_size) / slide_height),
    ]
    rectangle_polygon = Polygon(rectangle_coordinates)
    if check_intersection(jakub_polygon, rectangle_polygon):
        predicted_segmentation = np.load(predicted_segmentation_file, allow_pickle=True)
        argmax_arr = imresize(
            predicted_segmentation, (146, 146), interp="nearest", mode="F"
        ).astype(np.uint8)
        # copyfile(predicted_segmentation_file,'./Data/{}'.format(base_name))
        np.save(
            "./Data2/{}_{}_{}_{}.npy".format(
                x, y, patch_original_size, patch_resized_size
            ),
            argmax_arr,
        )
        copyfile(
            os.path.join(
                patch_directory,
                "{}_{}_{}_{}.png".format(x, y, patch_original_size, patch_resized_size),
            ),
            "./Data2/{}_{}_{}_{}.png".format(
                x, y, patch_original_size, patch_resized_size
            ),
        )
