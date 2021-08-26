import glob
import os
import numpy as np
import cv2
from bson import json_util
import json

seg_files = glob.glob('./Data2/*.npy')

stain_dict = {'k17p': 1, 'cd8': 2, 'cd16': 3, 'cd4': 4, 'cd3': 5, 'cd163': 6, 'k17n': 7}
ratio_threshold = 30
total_pixel_area = 146*146
json_file = './Data2/Tumor_Micro.json'
    
for seg_file in seg_files:
    base_name = os.path.basename(seg_file)
    print(base_name)
    x, y, orig_size, resize = base_name[:-4].split('_')
    x, y = int(x), int(y)
    pred = np.load(seg_file)
    number_of_pixels_for_stain = {'k17p': 0, 'cd8': 0, 'cd16': 0, 'cd4': 0, 'cd3': 0, 'cd163': 0, 'k17n': 0}
    json_file_name = './Data2/{}.json'.format(base_name[:-4])
    with open(json_file_name, 'w') as f:
        for stain in stain_dict.keys():
            img_mask = np.zeros((pred.shape[0],pred.shape[1])).astype('uint8')
            stain_idx = stain_dict[stain]
            img_mask[np.where(pred==stain_idx)] = 255
            number_of_pixels_for_stain[stain] = int(np.sum(img_mask))
            img_contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(img_contours)):
                dict_polygon = {}
                dict_polygon['stain_class'] = stain
                dict_polygon['size'] = cv2.contourArea(img_contours[i])
                polygon_coordinates = []
                for coord in img_contours[i]:
                    polygon_coordinates.append([str(coord[0][0]+x), str(coord[0][1]+y)])
                dict_polygon['coordinates'] = polygon_coordinates
                print(dict_polygon)
                json.dump(dict_polygon, f, default=json_util.default)
                f.write('\n')
        f.close()
            