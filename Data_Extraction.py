import glob
import numpy as np
import os
import openslide
import json
import pandas as pd
from shapely.geometry import Polygon
from shutil import copyfile
from scipy.misc import imresize
import cv2
from bson import json_util

# Following codes are necessary for including patches that are annotated by Prof. Shroyer as Tumor.
annotation_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/annot/SBU_Annot'
patch_directory = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi_patch_146_146/N9430-B11-multires.tif/'
manifest_file = os.path.join(annotation_dir, 'manifest.csv')
slide_name = 'N9430-B11'
clinicaltrialsubjectid = 'N9430'
imageid = 'B11'
manifest = pd.read_csv(manifest_file)
save_dir = '/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/data_for_tum_micro_2/{}'.format(slide_name)

# Path parameters
segmentation_input_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Result_Jakub/WSI/wsi_patch_146_146/Anchor_UNET"
wsi_input_directory = "/data00/shared/mahmudul/Sbu_Kyt_Pdac_merged/Input_Data/wsi/SBU"
wsi_filename = os.path.join(wsi_input_directory,'{}-multires.tif'.format(slide_name))
predicted_segmentation_files = glob.glob(os.path.join(segmentation_input_directory,"{}-multires.tif/*.npy".format(slide_name)))
patch_size = 146
#stain_dict = {'k17p': 1, 'cd8': 2, 'cd16': 3, 'cd4': 4, 'cd3': 5, 'cd163': 6, 'k17n': 7}
stain_dict = {'k17p': 1, 'cd8': 2, 'cd16': 3, 'cd163': 6, 'k17n': 7}
polygon_count_dict = {'k17p': 0, 'cd8': 0, 'cd16': 0, 'cd163': 0, 'k17n': 0}

# WSI parameters
oslide = openslide.OpenSlide(wsi_filename)
slide_width = int(oslide.dimensions[0])
slide_height = int(oslide.dimensions[1])
patch_width, patch_height = patch_size, patch_size
patch_width = patch_width / slide_width
patch_height = patch_height / slide_height
patch_area = patch_width * slide_width * patch_height * slide_height

json_file_path = manifest.loc[(manifest['clinicaltrialsubjectid'] == clinicaltrialsubjectid) & (manifest['imageid'] == imageid)]['path'].values[0]

with open(os.path.join(annotation_dir,json_file_path)) as f: #Loading JSON from manifest file. JSONs contain all annotations
    json_file = json.load(f)

shroyer_tumor_area = []
shroyer_not_tumor_area = []


for annot in json_file:
    if annot["properties"]["annotations"]["notes"] == "Shroyer Tumor":
        shroyer_tumor_area.append(annot["geometries"]["features"][0]["geometry"]["coordinates"][0])
    elif annot["properties"]["annotations"]["notes"] == "Shroyer not tumor":
        shroyer_not_tumor_area.append(annot["geometries"]["features"][0]["geometry"]["coordinates"][0])

tumor_polygons = []
not_tumor_polygons = []

for tumor in shroyer_tumor_area:
    tumor_tuple = [(i[0],i[1]) for i in tumor]
    tumor_polygons.append(Polygon(tumor_tuple))


for not_tumor in shroyer_not_tumor_area:
    not_tumor_tuple = [(i[0],i[1]) for i in not_tumor]
    not_tumor_polygons.append(Polygon(not_tumor_tuple))


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
                if dict_polygon['size'] > 5:
                    polygon_count_dict[stain] += 1
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


if(not os.path.isdir(save_dir)):
    os.mkdir(save_dir)

for predicted_segmentation_file in predicted_segmentation_files:
    base_name = os.path.basename(predicted_segmentation_file)
    print(base_name)
    x, y, patch_original_size, patch_resized_size, _ = base_name.split('_')
    x, y, patch_resized_size = int(x), int(y), int(patch_resized_size)
    rectangle_coordinates = [(x/slide_width, y/slide_height), ((x+patch_resized_size)/slide_width, y/slide_height), ((x+patch_resized_size)/slide_width, (y+patch_resized_size)/slide_height), ((x)/slide_width,(y+patch_resized_size)/slide_height)]
    rectangle_polygon = Polygon(rectangle_coordinates)
    if check_intersection(tumor_polygons, rectangle_polygon) and not check_intersection(not_tumor_polygons, rectangle_polygon):
        predicted_segmentation = np.load(predicted_segmentation_file, allow_pickle=True)
        argmax_arr = imresize(predicted_segmentation,(146,146),interp='nearest', mode='F').astype(np.uint8)
        png_filename = os.path.join(patch_directory,"{}_{}_{}_{}.png".format(x,y,patch_original_size,patch_resized_size))
        divide_patches(predicted_segmentation_file, argmax_arr, png_filename, save_dir)


with open(os.path.join(save_dir,'cell_count.txt'),'w') as f:
    f.write(str(polygon_count_dict))
    f.close()