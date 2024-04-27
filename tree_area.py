import cv2
import draw
import numpy as np
import functions
import functions_geo
import pandas as pd
import spans

from const import FILENAME_POINTCLOUD, FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE, FILENAME_TREE_AREA_MINOR, FILENAME_TREE_AREA_MAJOR


def get_area_scale(task_id, span_id):
    img = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}')
    min_lon, max_lon, min_lat, max_lat = draw.get_pointcloud_minmax(task_id, span_id)
    cloud_heigth = functions_geo.get_geo_distance((min_lat, min_lon), (max_lat, min_lon))
    cloud_witdh = functions_geo.get_geo_distance((min_lat, min_lon), (min_lat, max_lon))
    img_height = img.shape[0]
    img_width = img.shape[1]
    ar_scale = round(img_height*img_width/(cloud_heigth*cloud_witdh))

    return ar_scale


def get_tree_area_by_type(task_id, span_id, tree_type):  # tree type is  dry or green

    height_conditions = ('height_less_1.5', 'height_1.5-2.5', 'height_2.5-4', 'height_above_4')
    area_scale = get_area_scale(task_id, span_id)
    result = dict()
    result[tree_type] = dict()

    for height_condition in height_conditions:
        if 'height_less_1.5' in height_condition:
            area = inclosure_mask_and_trees(task_id, span_id, height_condition, tree_type)
            result[tree_type][height_condition] = float(format(area / area_scale, '.3f'))

        elif 'height_1.5-2.5' in height_condition:
            area = inclosure_mask_and_trees(task_id, span_id, height_condition,tree_type)
            result[tree_type][height_condition] = float(format(area / area_scale, '.3f'))

        elif 'height_2.5-4' in height_condition:
            area = inclosure_mask_and_trees(task_id, span_id, height_condition,tree_type)
            result[tree_type][height_condition] = float(format(area / area_scale, '.3f'))

        elif 'height_above_4' in height_condition:
            area = inclosure_mask_and_trees(task_id, span_id, height_condition,tree_type)
            result[tree_type][height_condition] = float(format(area / area_scale, '.3f'))

    return result


def get_all_tree_area(task_id, span_id):
    all_area = dict()
    local_heights_filter(task_id, span_id)  # this heavily increases runtime of function and only saves png files. i disabled it.

    green_tree_area = get_tree_area_by_type(task_id, span_id, 'green')
    dry_tree_area = get_tree_area_by_type(task_id, span_id, 'dry')

    height_conditions = ('height_less_1.5', 'height_1.5-2.5', 'height_2.5-4', 'height_above_4')

    all_area['span_id'] = [span_id]
    for height_condition in height_conditions:
        all_area[height_condition] = [green_tree_area['green'][height_condition] + dry_tree_area['dry'][height_condition]]

    return all_area


def save_tree_area(task_id, span_id):
    tree_area = get_all_tree_area(task_id, span_id)
    dct = spans.get_span_supplementary_data(task_id, span_id)

    df = pd.DataFrame.from_dict(tree_area, orient='columns')
    for key, value in dct.items():  # adding district, dname, voltageclass etc
        df[key] = value

    df.to_csv(f'debug/{task_id}/spans/{span_id}/{FILENAME_TREE_AREA_MINOR}', index=False)
    return


def inclosure_mask_and_trees(task_id, span_id, height_condition, tree_type):
    inclosure_mask_img = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/inclosure_mask.png'), cv2.COLOR_BGR2GRAY)
    trees_mask_img = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/mask_'+ tree_type +'_tree.png'), cv2.COLOR_BGR2GRAY)
    height_mask = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/height_map_grey_' + height_condition +'.png'), cv2.COLOR_BGR2GRAY)
    result_mask = inclosure_mask_img * trees_mask_img * height_mask
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/result_mask_' + tree_type + '_' + height_condition + '.png', result_mask)
    result_mask = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/result_mask_' + tree_type + '_' + height_condition + '.png'), cv2.COLOR_BGR2GRAY)
    white_area = np.sum(result_mask == 255)
    return white_area


def local_heights_filter(task_id, span_id):
    pointcloud_path = f'debug/{task_id}/spans/{span_id}/{FILENAME_POINTCLOUD}'
    grey_img_path = f'debug/{task_id}/spans/{span_id}/height_map_grey_filtered_resize.png'
    img_origin = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}')

    height_map_grey_filtered_img = cv2.imread(f'debug/{task_id}/spans/{span_id}/height_map_grey_filtered.png')

    height_map_grey_filtered_img = draw.img_resize(img_origin, height_map_grey_filtered_img)
    cv2.imwrite(grey_img_path, height_map_grey_filtered_img)

    heights_coords = functions.get_local_heights(grey_img_path, pointcloud_path)  # get local heights on [0..255]
    absolute_max_altitude = heights_coords['absolute_height_max_altitude']
    absolute_min_altitude = heights_coords['absolute_height_min_altitude']
    local_heights_origin = heights_coords['image_local_heights']
    local_heights_origin = (local_heights_origin * (absolute_max_altitude - absolute_min_altitude) / 255)  # convert local heights to meters

    local_heights = np.copy(local_heights_origin)
    local_heights = np.where((local_heights > 0) & (local_heights <= 1.5), 255, 0)
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/height_map_grey_height_less_1.5.png', local_heights)

    local_heights = np.copy(local_heights_origin)
    local_heights = np.where((local_heights > 1.5) & (local_heights <= 2.5), 255, 0)
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/height_map_grey_height_1.5-2.5.png', local_heights)

    local_heights = np.copy(local_heights_origin)
    local_heights = np.where((local_heights > 2.5) & (local_heights <= 4), 255, 0)
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/height_map_grey_height_2.5-4.png', local_heights)

    local_heights = np.copy(local_heights_origin)
    local_heights = np.where(local_heights >= 4, 255, 0)
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/height_map_grey_height_above_4.png', local_heights)

    return
