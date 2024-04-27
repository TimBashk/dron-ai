#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import cv2
import functions
import functions_geo
import spans
import numpy as np

import functions_kml_db
import functions_geo

from const import (PILLAR_COLOR, PILLAR_RADIUS, INCLOSURE_COLOR, POWERLINE_WIDTH, INCLOSURE_LINE_WIDTH,INCLOSURE_WIDTH,
                   FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE, FILENAME_LOCAL_MAXIMA_CSV, FILENAME_POINTCLOUD,
                   COLOR_TREE_GREEN_IN_ZONE, COLOR_TREE_GREEN_OUT_ZONE, COLOR_TREE_DRY_OUT_ZONE, COLOR_TREE_DRY_IN_ZONE,
                   FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE_MAXIMA, FILENAME_ORTHOPHOTO_PILLARS_MAJOR,
                   FILENAME_ORTHOPHOTO_MAJOR_ROTATED, PILLAR_RADIUS_MAJOR, PILLAR_WIDTH_MAJOR, FILENAME_IMAGE_AFTER_SPLIT,
                   FILENAME_ORTHOPHOTO_WITH_MASK, FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS, FILENAME_INCLOSURE_LINES,
                   TREE_RADIUS, TREE_WIDTH, PILLAR_WIDTH, BACKGROUND_WEIGHT, FILENAME_ORTHOPHOTO, FILENAME_SCHEME_SPANS, FILENAME_ORTHOPHOTO_ALIGNED_CROPPED, FILENAME_INCLOSURE_MASK)

from shapely.geometry import LineString
from scipy.spatial import distance
import matplotlib.pyplot as plt
import wires_detect


def get_pointcloud_minmax(task_id, span_id):
    df = pd.read_csv(f'debug/{task_id}/spans/{span_id}/pointcloud.txt')

    min_x = df['x'].min()
    min_y = df['y'].min()

    max_x = df['x'].max()
    max_y = df['y'].max()

    return min_x, max_x, min_y, max_y


def get_pointcloud_minmax_major(task_id):
    df = functions_geo.get_pointcloud_major(task_id)

    min_x = df['x'].min()
    min_y = df['y'].min()

    max_x = df['x'].max()
    max_y = df['y'].max()

    return min_x, max_x, min_y, max_y


def get_span_coords(line_id, target_span_id):
    df_pillar = functions_kml_db.pylons_lat_lon_df(line_id)
    df_pillar['coord'] = [(lat, lon) for lat, lon in zip(df_pillar['lat'], df_pillar['lon'])]

    for span_id, (pillar1, pillar2) in enumerate(zip(df_pillar['coord'], list(df_pillar['coord'])[1:])):
        if span_id == int(target_span_id):
            return pillar1, pillar2


def draw_circle(img, x, y, color, radius, thickness=-1):
    img = cv2.circle(img, (x, y), radius, color, thickness)  # -1 means fill circle  # 15 radius

    return img


def draw_line(img, x1, y1, x2, y2, color, width):
    img = cv2.line(img, (x1, y1), (x2, y2), color, width)  # 10 linewidth
    return img


def draw_pillars_major(task_id):
    df_pillars = spans.get_pillars_df(task_id)
    img_original = cv2.imread(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_ALIGNED_CROPPED}')
    df_pointcloud = functions_geo.get_pointcloud_major(task_id)
    lon_min, lon_max, lat_min, lat_max = functions_geo.get_lon_lat_min_max(df_pointcloud)
    img_circles = img_original.copy()

    #print(img_circles.shape)
    #print(lon_min, lon_max, lat_min, lat_max)
    for lon, lat in zip(df_pillars['lon'], df_pillars['lat']):
        x, y = functions.convert_lonlat_to_index(img_circles, (lon, lat), lon_min, lon_max, lat_min, lat_max)
        #print(x, y)
        img_circles = cv2.circle(img_circles, (x, y), PILLAR_RADIUS_MAJOR, PILLAR_COLOR, PILLAR_WIDTH_MAJOR)  # last argument is thickness

    sum_ = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img_circles, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_PILLARS_MAJOR}', sum_, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def draw_pillars(task_id, span_id):
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK}')
    img = img_original.copy()

    min_lon, max_lon, min_lat, max_lat = get_pointcloud_minmax(task_id, span_id)

    df_spans = spans.load_spans(task_id)
    span_dct = df_spans.to_dict('records')[int(span_id)]

    lon1 = span_dct['lon1']
    lat1 = span_dct['lat1']
    lon2 = span_dct['lon2']
    lat2 = span_dct['lat2']

    x, y = functions.convert_lonlat_to_index(img, (lon1, lat1), min_lon, max_lon, min_lat, max_lat)
    img = draw_circle(img, x, y, PILLAR_COLOR, PILLAR_RADIUS, PILLAR_WIDTH)

    x, y = functions.convert_lonlat_to_index(img, (lon2, lat2), min_lon, max_lon, min_lat, max_lat)
    img = draw_circle(img, x, y, PILLAR_COLOR, PILLAR_RADIUS, PILLAR_WIDTH)

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS}', img)


def draw_powerline(task_id, span_id):
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS}')
    img = img_original.copy()

    min_lon, max_lon, min_lat, max_lat = get_pointcloud_minmax(task_id, span_id)

    df_spans = spans.load_spans(task_id)
    span_dct = df_spans.to_dict('records')[int(span_id)]

    lon1 = span_dct['lon1']
    lat1 = span_dct['lat1']
    lon2 = span_dct['lon2']
    lat2 = span_dct['lat2']

    x1, y1 = functions.convert_lonlat_to_index(img, (lon1, lat1), min_lon, max_lon, min_lat, max_lat)
    x2, y2 = functions.convert_lonlat_to_index(img, (lon2, lat2), min_lon, max_lon, min_lat, max_lat)
    img = draw_line(img, x1, y1, x2, y2, PILLAR_COLOR, POWERLINE_WIDTH)

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}', img)


def get_scale(task_id, span_id):
    img = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO}')
    min_lon, max_lon, min_lat, max_lat = get_pointcloud_minmax(task_id, span_id)
    cloud_heigth = functions_geo.get_geo_distance((min_lat, min_lon), (max_lat, min_lon))
    img_height = img.shape[0]
    scale = round(img_height/cloud_heigth)

    return scale


def get_perpendicular_vector_coords(point, width, position):
    if position == 'bottom':
        vector = LineString([point[1], point[0]])
    elif position == 'top':
        vector = LineString([point[0], point[1]])

    left = vector.parallel_offset(width, 'left')
    right = vector.parallel_offset(width, 'right')
    left_point = left.boundary[1]
    right_point = right.boundary[0]

    perpendicular_vector_coords = {'left_point_x':round(left_point.x),
                                   'left_point_y':round(left_point.y),
                                   'right_point_x':round(right_point.x),
                                   'right_point_y':round(right_point.y)}

    return perpendicular_vector_coords


def get_vector(task_id, span_id):
    img = cv2.imread(f'debug/{task_id}/spans/{span_id}/orthophoto.jpg')
    vector_coords = []
    min_lon, max_lon, min_lat, max_lat = get_pointcloud_minmax(task_id, span_id)

    df_spans = spans.load_spans(task_id)
    span_dct = df_spans.to_dict('records')[int(span_id)]

    lon1 = span_dct['lon1']
    lat1 = span_dct['lat1']
    lon2 = span_dct['lon2']
    lat2 = span_dct['lat2']

    x, y = functions.convert_lonlat_to_index(img, (lon1, lat1), min_lon, max_lon, min_lat, max_lat)
    vector_coords.append((x, y))

    x, y = functions.convert_lonlat_to_index(img, (lon2, lat2), min_lon, max_lon, min_lat, max_lat)
    vector_coords.append((x, y))

    return vector_coords


def drawCircle(img, x, y, color, radius):
    img = cv2.circle(img, (x, y), radius, color, -1)  # -1 means fill circle  # 15 radius
    return img


def drawLine(img, x1, y1, x2, y2, color, width):
    img = cv2.line(img, (x1, y1), (x2, y2), color, width)  # 10 linewidth
    return img


def draw_inclosure_line(task_id, span_id):
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}')
    img = img_original.copy()
    vector = get_vector(task_id, span_id)
    scale = get_scale(task_id, span_id)

    inclosure_width = wires_detect.get_inclosure_area_width_by_voltage_class(task_id)
    inclosure_width = scale * inclosure_width

    bottom_p_v_coords = get_perpendicular_vector_coords(vector, inclosure_width, 'bottom')
    top_p_v_coords = get_perpendicular_vector_coords(vector, inclosure_width, 'top')

    line1_x1 = top_p_v_coords['left_point_x']
    line1_y1 = top_p_v_coords['left_point_y']
    line1_x2 = bottom_p_v_coords['right_point_x']
    line1_y2 = bottom_p_v_coords['right_point_y']

    line2_x1 = top_p_v_coords['right_point_x']
    line2_y1 = top_p_v_coords['right_point_y']
    line2_x2 = bottom_p_v_coords['left_point_x']
    line2_y2 = bottom_p_v_coords['left_point_y']
    polygon_coord = (
        [line1_x1, line2_x1], [line1_y1, line2_y1], [line1_x2, line2_x2], [line1_y2, line2_y2]
    )

    img = drawLine(img, line1_x1, line1_y1, line1_x2, line1_y2, INCLOSURE_COLOR, INCLOSURE_LINE_WIDTH)
    img = drawLine(img, line2_x1, line2_y1, line2_x2, line2_y2, INCLOSURE_COLOR, INCLOSURE_LINE_WIDTH)

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}', img)


def draw_inclosure_lines(task_id, span_id):
    blue_pixel = [255, 0, 0]
    black_threshold = 10
    mask = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_INCLOSURE_MASK}')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}')
    img = img_original.copy()
    lst, contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(mask.shape)
    for i in lst[0]:
        x = i[0][0]
        y = i[0][1]

        mask[y][x] = 122
        mask[y][x] = 122

    for y in range(img_original.shape[0]):
        for x in range(img_original.shape[1]):
            pixel_mask = mask[y][x]
            pixel_original_img = list(img_original[y][x])

            if pixel_mask != 0 and sum(pixel_original_img) >= black_threshold:
                img[y][x] = blue_pixel

                if y - 1 >= 0 and (img_original[y - 1][x]).sum() >= black_threshold:
                    img[y - 1][x] = blue_pixel

                if y - 2 >= 0 and (img_original[y - 2][x]).sum() >= black_threshold:
                    img[y - 2][x] = blue_pixel

                if y - 3 >= 0 and (img_original[y - 3][x]).sum() >= black_threshold:
                    img[y - 3][x] = blue_pixel

                if y + 1 < img_original.shape[0] and (img_original[y + 1][x]).sum() >= black_threshold:
                    img[y + 1][x] = blue_pixel

                if y + 2 < img_original.shape[0] and (img_original[y + 2][x]).sum() >= black_threshold:
                    img[y + 2][x] = blue_pixel

                if y + 3 < img_original.shape[0] and (img_original[y + 3][x]).sum() >= black_threshold:
                    img[y + 3][x] = blue_pixel

                if x - 1 >= 0 and (img_original[y][x - 1]).sum() >= black_threshold:
                    img[y][x - 1] = blue_pixel

                if x - 2 >= 0 and (img_original[y][x - 2]).sum() >= black_threshold:
                    img[y][x - 2] = blue_pixel

                if x - 3 >= 0 and (img_original[y][x - 3]).sum() >= black_threshold:
                    img[y][x - 3] = blue_pixel

                if x + 1 < img_original.shape[1] and (img_original[y][x + 1]).sum() >= black_threshold:
                    img[y][x + 1] = blue_pixel

                if x + 2 < img_original.shape[1] and (img_original[y][x + 2]).sum() >= black_threshold:
                    img[y][x + 2] = blue_pixel

                if x + 3 < img_original.shape[1] and (img_original[y][x + 3]).sum() >= black_threshold:
                    img[y][x + 3] = blue_pixel

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}', img)

    return


def save_inclosure_polygon(task_id, span_id):
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO}')
    img = img_original.copy()
    vector = get_vector(task_id, span_id)
    scale = get_scale(task_id, span_id)

    inclosure_width = wires_detect.get_inclosure_area_width_by_voltage_class(task_id)
    inclosure_width = scale * inclosure_width

    bottom_p_v_coords = get_perpendicular_vector_coords(vector, inclosure_width, 'bottom')
    top_p_v_coords = get_perpendicular_vector_coords(vector, inclosure_width, 'top')

    line1_x1 = top_p_v_coords['left_point_x']
    line1_y1 = top_p_v_coords['left_point_y']
    line1_x2 = bottom_p_v_coords['right_point_x']
    line1_y2 = bottom_p_v_coords['right_point_y']

    line2_x1 = top_p_v_coords['right_point_x']
    line2_y1 = top_p_v_coords['right_point_y']
    line2_x2 = bottom_p_v_coords['left_point_x']
    line2_y2 = bottom_p_v_coords['left_point_y']

    polygon_coord = ([line1_x1, line2_x1], [line1_y1, line2_y1], [line1_x2, line2_x2], [line1_y2, line2_y2])

    img = drawLine(img, line1_x1, line1_y1, line1_x2, line1_y2, INCLOSURE_COLOR, INCLOSURE_LINE_WIDTH)
    img = drawLine(img, line2_x1, line2_y1, line2_x2, line2_y2, INCLOSURE_COLOR, INCLOSURE_LINE_WIDTH)

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}', img)

    pd.DataFrame({'name': ['line1', 'line2'],
                  'x1': polygon_coord[0],
                  'y1': polygon_coord[1],
                  'x2': polygon_coord[2],
                  'y2': polygon_coord[3]}).to_csv(f'debug/{task_id}/spans/{span_id}/{FILENAME_INCLOSURE_LINES}',
                                                  index=False)
    return


def get_inclosure_polygon(task_id, span_id):
    polygon = pd.read_csv(f'debug/{task_id}/spans/{span_id}/inclosure_lines.csv', delimiter=',')
    polygon = np.array([polygon[['x1', 'y1']].to_numpy()[0, :], polygon[['x1', 'y1']].to_numpy()[1, :],
                        polygon[['x2', 'y2']].to_numpy()[0, :], polygon[['x2', 'y2']].to_numpy()[1, :]])
    return sortpts_clockwise(polygon)


def draw_maxima(task_id, span_id):
    img_original = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE}')
    img = img_original.copy()
    min_lon, max_lon, min_lat, max_lat = get_pointcloud_minmax(task_id, span_id)

    df_maxima = pd.read_csv(f'debug/{task_id}/spans/{span_id}/{FILENAME_LOCAL_MAXIMA_CSV}')

    for lon, lat, is_tree_green, is_tree_dry, is_in_secured_zone in zip(df_maxima['lon'],
                                                                        df_maxima['lat'],
                                                                        df_maxima['is_tree_green'],
                                                                        df_maxima['is_tree_dry'],
                                                                        df_maxima['is_in_secured_zone']):

        x, y = functions.convert_lonlat_to_index(img, (lon, lat), min_lon, max_lon, min_lat, max_lat)

        if is_in_secured_zone:
            if is_tree_dry:
                img = draw_circle(img, x, y, COLOR_TREE_DRY_IN_ZONE, TREE_RADIUS, TREE_WIDTH)
            elif is_tree_green:
                img = draw_circle(img, x, y, COLOR_TREE_GREEN_IN_ZONE, TREE_RADIUS, TREE_WIDTH)

    img = cv2.addWeighted(img_original, BACKGROUND_WEIGHT, img, 1 - BACKGROUND_WEIGHT, 0)

    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE_MAXIMA}', img)


def sortpts_clockwise(point):
    # Sort A based on Y(col-2) coordinates
    sortedAc2 = point[np.argsort(point[: , 1]), :]

    # Get top two and bottom two points
    top2 = sortedAc2[0:2, :]
    bottom2 = sortedAc2[2:, :]

    # Sort top2 points to have the first row as the top-left one
    sortedtop2c1 = top2[np.argsort(top2[:, 0]), :]
    top_left = sortedtop2c1[0, :]

    # Use top left point as pivot & calculate sq-euclidean dist against
    # bottom2 points & thus get bottom-right, bottom-left sequentially
    sqdists = distance.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists, 0))[::-1], :]

    # Concatenate all these points for the final output

    return np.concatenate((sortedtop2c1, rest2), axis=0)


def img_resize(img, target_img):
    new_shape = (img.shape[1], img.shape[0])
    target_img = cv2.resize(target_img, new_shape)

    return target_img


def get_mean_coord(lon1, lat1, lon2, lat2):
    lon_mean = (lon1 + lon2) / 2
    lat_mean = (lat1 + lat2) / 2
    return lon_mean, lat_mean


def draw_scheme_spans(task_id):
    df_spans = spans.load_spans(task_id)
    plt.figure(figsize=[30, 30])
    for lon1, lat1, lon2, lat2, span_id in zip(df_spans['lon1'], df_spans['lat1'], df_spans['lon2'], df_spans['lat2'], df_spans['span_id']):
        plt.plot([lon1, lon2], [lat1, lat2])
        lon_mean, lat_mean = get_mean_coord(lon1, lat1, lon2, lat2)
        plt.text(lon_mean, lat_mean, span_id, size=10)

    plt.savefig(f'debug/{task_id}/{FILENAME_SCHEME_SPANS}', bbox_inches='tight')
