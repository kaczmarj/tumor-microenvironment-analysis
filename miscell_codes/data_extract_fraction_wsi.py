import glob
import numpy as np
import os
import openslide
import datetime
from bson import json_util
import json
import pandas as pd
from shapely.geometry import Polygon
from shutil import copyfile
from scipy.misc import imresize
import cv2

# Path Parameters for Annotation
annotation_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/annot/KYT_ANNOT/'
patch_directory = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi_patch_146_146/KYT/1092A-multires.tif/'
manifest_file = os.path.join(annotation_dir, 'manifest.csv')
slide_name = '1092A'
clinicaltrialsubjectid = '1092A-IHC'
imageid = '1092A-IHC:kytnew'
manifest = pd.read_csv(manifest_file)
jakub_roi = []

# Path parameters for Segmentation
segmentation_input_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/WSI/wsi_patch_146_146/Anchor_UNET/KYT/"
wsi_input_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi/KYT/"
wsi_filename = os.path.join(wsi_input_directory,'{}-multires.tif'.format(slide_name))
predicted_segmentation_files = glob.glob(os.path.join(segmentation_input_directory,"{}-multires.tif/*.npy".format(slide_name)))
patch_size = 146
stain_dict = {'k17p': 1, 'cd8': 2, 'cd16': 3, 'cd4': 4, 'cd3': 5, 'cd163': 6, 'k17n': 7}


# WSI parameters
oslide = openslide.OpenSlide(wsi_filename)
slide_width = int(oslide.dimensions[0])
slide_height = int(oslide.dimensions[1])
patch_width, patch_height = patch_size, patch_size
patch_width = patch_width / slide_width
patch_height = patch_height / slide_height
patch_area = patch_width * slide_width * patch_height * slide_height

json_file_path = manifest.loc[(manifest['clinicaltrialsubjectid'] == clinicaltrialsubjectid) & (manifest['imageid'] == imageid)]['path'].values[0]
patch_size = 146


with open(os.path.join(annotation_dir,json_file_path)) as f: #Loading JSON from manifest file. JSONs contain all annotations
    json_file = json.load(f)

for annot in json_file:
    if annot["properties"]["annotations"]["notes"] == "Jakub-ROI-6000px":
        box_coordinate = annot["geometries"]["features"][0]["geometry"]["coordinates"][0][0]
        box_x, box_y = box_coordinate
        x_add, y_add = 6000 / slide_width, 6000 / slide_height
        jakub_roi.append([[box_x, box_y],[box_x+x_add, box_y], [box_x+x_add, box_y+y_add], [box_x, box_y+y_add]])

jakub_polygon = []

for roi in jakub_roi:
    roi_tuple = [(i[0],i[1]) for i in roi]
    jakub_polygon.append(Polygon(roi_tuple))


# This function checks whether a rectangle intersects a polygon list. We have considered shroyer tumor list and shroyer not tumor list
def check_intersection(polygon_list, rectangle):
    for polygons in polygon_list:
        if polygons.intersects(rectangle):
            return True
    return False


def json_save(pred, json_file_name):
    with open(json_file_name, 'w') as f:
        for stain in stain_dict.keys():
            img_mask = np.zeros((pred.shape[0],pred.shape[1])).astype('uint8')
            stain_idx = stain_dict[stain]
            img_mask[np.where(pred==stain_idx)] = 255
            img_contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(img_contours)):
                dict_polygon = {}
                dict_polygon['stain_class'] = stain
                dict_polygon['size'] = cv2.contourArea(img_contours[i])
                polygon_coordinates = []
                for coord in img_contours[i]:
                    polygon_coordinates.append([str(coord[0][0]+x), str(coord[0][1]+y)])
                dict_polygon['coordinates'] = polygon_coordinates
                json.dump(dict_polygon, f, default=json_util.default)
                f.write('\n')
        f.close()


def divide_patches(pred_file, pred_file_npy, img_file, dest_path):
    base_file = os.path.basename(pred_file)
    x, y, _, _, _ = base_file.split('_')
    x, y = int(x), int(y)
    img = cv2.imread(img_file)
    height, width = pred_file_npy.shape[0], pred_file_npy.shape[1]
    first_start, first_end = (x, y), (x+height//2, y+width//2)
    second_start, second_end = (x, y+width//2), (x+height//2, y+width)
    third_start, third_end = (x+height//2,y), (x+height, y+width//2)
    fourth_start, fourth_end = (x+height//2, y+width//2), (x+height, y+width)

    # First Quarter Image, Prediction and JSON
    first_img = img[first_start[0]-x:first_end[0]-x, first_start[1]-y:first_end[1]-y]
    first_img_file_name = os.path.join(dest_path,'{}_{}_{}_{}.png'.format(first_start[0],first_start[1],height//2,width//2))
    cv2.imwrite(first_img_file_name,first_img)

    first_pred = pred_file_npy[first_start[0]-x:first_end[0]-x, first_start[1]-y:first_end[1]-y]
    first_file_name = os.path.join(dest_path,'{}_{}_{}_{}.npy'.format(first_start[0],first_start[1],height//2,width//2))
    np.save(first_file_name,first_pred)

    first_pred_json = os.path.join(dest_path,'{}_{}_{}_{}.json'.format(first_start[0],first_start[1],height//2,width//2))
    json_save(first_pred, first_pred_json)
    
    # Second Quarter Image, Prediction and JSON
    second_img = img[second_start[0]-x:second_end[0]-x, second_start[1]-y:second_end[1]-y]
    second_img_file_name = os.path.join(dest_path,'{}_{}_{}_{}.png'.format(third_start[0],third_start[1],height//2,width//2))
    cv2.imwrite(second_img_file_name,second_img)

    second_pred = pred_file_npy[second_start[0]-x:second_end[0]-x, second_start[1]-y:second_end[1]-y]
    second_file_name = os.path.join(dest_path,'{}_{}_{}_{}.npy'.format(third_start[0],third_start[1],height//2,width//2))
    np.save(second_file_name,second_pred)

    second_pred_json = os.path.join(dest_path,'{}_{}_{}_{}.json'.format(third_start[0],third_start[1],height//2,width//2))
    json_save(second_pred, second_pred_json)

    # Third Quarter Image, Prediction and JSON
    third_img = img[third_start[0]-x:third_end[0]-x, third_start[1]-y:third_end[1]-y]
    third_img_file_name = os.path.join(dest_path,'{}_{}_{}_{}.png'.format(second_start[0],second_start[1],height//2,width//2))
    cv2.imwrite(third_img_file_name,third_img)

    third_pred = pred_file_npy[third_start[0]-x:third_end[0]-x, third_start[1]-y:third_end[1]-y]
    third_file_name = os.path.join(dest_path,'{}_{}_{}_{}.npy'.format(second_start[0],second_start[1],height//2,width//2))
    np.save(third_file_name,third_pred)

    third_pred_json = os.path.join(dest_path,'{}_{}_{}_{}.json'.format(second_start[0],second_start[1],height//2,width//2))
    json_save(third_pred, third_pred_json)

    # Fourth Quarter Image, Prediction and JSON
    fourth_img = img[fourth_start[0]-x:fourth_end[0]-x, fourth_start[1]-y:fourth_end[1]-y]
    fourth_img_file_name = os.path.join(dest_path,'{}_{}_{}_{}.png'.format(fourth_start[0],fourth_start[1],height//2,width//2))
    cv2.imwrite(fourth_img_file_name,fourth_img)

    fourth_pred = pred_file_npy[fourth_start[0]-x:fourth_end[0]-x, fourth_start[1]-y:fourth_end[1]-y]
    fourth_file_name = os.path.join(dest_path,'{}_{}_{}_{}.npy'.format(fourth_start[0],fourth_start[1],height//2,width//2))
    np.save(fourth_file_name,fourth_pred)

    fourth_pred_json = os.path.join(dest_path,'{}_{}_{}_{}.json'.format(fourth_start[0],fourth_start[1],height//2,width//2))
    json_save(fourth_pred, fourth_pred_json)


dest_dir = "/data02/shared/mahmudul/Sbu_Kyt_Pdac_merged/Codes/Tumor_Mirco_Env_Data/low_k17/{}".format(slide_name)

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)


roi_num = 1

for single_poly in jakub_polygon:
    dest_path = os.path.join(dest_dir,"ROI_{}".format(roi_num))
    roi_num += 1
    single_poly = [single_poly]
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)   
    for predicted_segmentation_file in predicted_segmentation_files:
        base_name = os.path.basename(predicted_segmentation_file)
        x, y, patch_original_size, patch_resized_size, _ = base_name.split('_')
        x, y, patch_resized_size = int(x), int(y), int(patch_resized_size)
        rectangle_coordinates = [(x/slide_width, y/slide_height), ((x+patch_resized_size)/slide_width, y/slide_height), ((x+patch_resized_size)/slide_width, (y+patch_resized_size)/slide_height), ((x)/slide_width,(y+patch_resized_size)/slide_height)]
        rectangle_polygon = Polygon(rectangle_coordinates)
        if check_intersection(single_poly, rectangle_polygon):
            predicted_segmentation = np.load(predicted_segmentation_file, allow_pickle=True)
            argmax_arr = imresize(predicted_segmentation,(patch_size,patch_size),interp='nearest', mode='F').astype(np.uint8)
            png_filename = os.path.join(patch_directory,"{}_{}_{}_{}.png".format(x,y,patch_original_size,patch_resized_size))
            divide_patches(predicted_segmentation_file, argmax_arr, png_filename, dest_path)
        