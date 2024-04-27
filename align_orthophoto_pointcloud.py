import pandas as pd
from geotiff import GeoTiff
from numba import njit
import numpy as np
import cv2
import functions_geo

from const import FILENAME_ORTHOPHOTO_MAJOR_JPG, FILENAME_ORTHOPHOTO_MAJOR_TIFF, FILENAME_ORTHOPHOTO_ALIGNED, FILENAME_ORTHOPHOTO_ALIGNED_CROPPED, FILENAME_POINTCLOUD_MAJOR


@njit
def get_new_xy(pixel_lon, pixel_lat, min_lon, max_lon, min_lat, max_lat, new_width, new_height):
    width_ratio = (pixel_lon - min_lon) / (max_lon - min_lon)
    height_ratio = (pixel_lat - min_lat) / (max_lat - min_lat)

    new_x = int(round(new_width * width_ratio, 0))
    new_y = int(round(new_height * height_ratio, 0))

    if new_x >= new_width:
        new_x = new_width - 1

    if new_y >= new_height:
        new_y = new_height - 1

    new_y = new_height - new_y

    return new_x, new_y


def get_lats_first_column(geo_upper_left, geo_lower_left, height):
    increment_lat = abs(geo_upper_left[1] - geo_lower_left[1]) / height
    lat_start = geo_upper_left[1]
    lats = [(lat_start - increment_lat * i) for i in range(height)]

    return lats


def get_lats_last_column(geo_upper_right, geo_lower_right, height):
    increment_lat = abs(geo_upper_right[1] - geo_lower_right[1]) / height
    lat_start = geo_upper_right[1]
    lats = [(lat_start - increment_lat * i) for i in range(height)]

    return lats


def get_lons_first_row(geo_upper_left, geo_upper_right, width):
    increment_lon = (geo_upper_right[0] - geo_upper_left[0]) / width
    lon_start = geo_upper_left[0]
    lons = [(lon_start + increment_lon * i) for i in range(width)]

    return lons


def get_lons_last_row(geo_lower_left, geo_lower_right, width):
    increment_lon = (geo_lower_right[0] - geo_lower_left[0]) / width
    lon_start = geo_lower_left[0]
    lons = [(lon_start + increment_lon * i) for i in range(width)]

    return lons


def get_pixel_lonlat(x, y, width, height, first_lons, last_lons, first_lats, last_lats):
    ratio = y / height
    lon = first_lons[x] + (last_lons[x] - first_lons[x]) * ratio

    ratio = x / width
    lat = first_lats[y] + (last_lats[y] - first_lats[y]) * ratio

    return lon, lat


def align(task_id):
    geo_tiff = GeoTiff(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_MAJOR_TIFF}')
    img_original = cv2.imread(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_MAJOR_JPG}')

    height = geo_tiff.tif_shape[0]
    width = geo_tiff.tif_shape[1]

    geo_upper_left = geo_tiff.get_wgs_84_coords(0, 0)
    geo_lower_right = geo_tiff.get_wgs_84_coords(width, height)

    geo_lower_left = geo_tiff.get_wgs_84_coords(0, height)
    geo_upper_right = geo_tiff.get_wgs_84_coords(width, 0)

    lons = [i[0] for i in [geo_upper_left, geo_lower_right, geo_lower_left, geo_upper_right]]
    lats = [i[1] for i in [geo_upper_left, geo_lower_right, geo_lower_left, geo_upper_right]]

    pixel_width = abs((geo_upper_left[0] - geo_upper_right[0])) / (width - 1)  # in geo coodinates
    pixel_height = abs((geo_upper_left[1] - geo_lower_left[1])) / (height - 1)  # in geo coodinates

    new_width = int(round((max(lons) - min(lons)) / pixel_width, 0))
    new_height = int(round((max(lats) - min(lats)) / pixel_height, 0))

    new_img = np.zeros([new_height, new_width, 3], dtype='uint8')

    first_lats = get_lats_first_column(geo_upper_left, geo_lower_left, height)
    last_lats = get_lats_last_column(geo_upper_right, geo_lower_right, height)

    first_lons = get_lons_first_row(geo_upper_left, geo_upper_right, width)
    last_lons = get_lons_last_row(geo_lower_left, geo_lower_right, width)

    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    for y in range(img_original.shape[0]):
        if y % 100 == 0:
            print(y)

        for x in range(img_original.shape[1]):
            pixel = img_original[y][x]
            pixel_lon, pixel_lat = get_pixel_lonlat(x, y, width, height, first_lons, last_lons, first_lats, last_lats)
            new_x, new_y = get_new_xy(pixel_lon, pixel_lat, min_lon, max_lon, min_lat, max_lat, new_width, new_height)
            new_img[new_y][new_x] = pixel

    lon_min_cropped, lon_max_cropped, lat_min_cropped, lat_max_cropped = crop_black(task_id, new_img, min_lon, max_lon, min_lat, max_lat)
    crop_orthophoto_pointcloud(task_id, lon_min_cropped, lon_max_cropped, lat_min_cropped, lat_max_cropped)


def get_min_y(img_grey):
    for y in range(img_grey.shape[0]):
        unique = np.unique(img_grey[y])
        if len(unique) == 1 and unique[0] == 0:
            continue

        upper_boundary = y
        return upper_boundary


def get_max_y(img_grey):
    for y in reversed(range(img_grey.shape[0])):
        unique = np.unique(img_grey[y])
        if len(unique) == 1 and unique[0] == 0:
            continue

        lower_boundary = y
        return lower_boundary


def get_min_x(img_grey):
    for x in range(img_grey.shape[1]):
        unique = np.unique(img_grey[:, x])
        if len(unique) == 1 and unique[0] == 0:
            continue

        left_boundary = x
        return left_boundary


def get_max_x(img_grey):
    for x in reversed(range(img_grey.shape[1])):
        unique = np.unique(img_grey[:, x])
        if len(unique) == 1 and unique[0] == 0:
            continue

        right_boundary = x
        return right_boundary


def convert_y_to_lat(img, boundary_y, lat_min, lat_max):
    height = img.shape[0]
    ratio = boundary_y / height
    boundary_lat = lat_max - (lat_max - lat_min) * ratio
    return boundary_lat


def convert_x_to_lon(img, boundary_x, lon_min, lon_max):
    width = img.shape[1]
    ratio = boundary_x / width
    boundary_lon = lon_min + (lon_max - lon_min) * ratio
    return boundary_lon


def crop_black(task_id, img, lon_min, lon_max, lat_min, lat_max):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y_min_cropped = get_min_y(img_grey)
    y_max_cropped = get_max_y(img_grey)
    x_min_cropped = get_min_x(img_grey)
    x_max_cropped = get_max_x(img_grey)

    lat_min_cropped = convert_y_to_lat(img, y_max_cropped, lat_min, lat_max)
    lat_max_cropped = convert_y_to_lat(img, y_min_cropped, lat_min, lat_max)

    lon_min_cropped = convert_x_to_lon(img, x_min_cropped, lon_min, lon_max)
    lon_max_cropped = convert_x_to_lon(img, x_max_cropped, lon_min, lon_max)

    img = img[y_min_cropped:y_max_cropped, x_min_cropped:x_max_cropped]  # must be after cropped variables e.g lon_max_cropped!!!
    cv2.imwrite(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_ALIGNED_CROPPED}', img)

    # write info
    df = list()
    df.append(['original', lon_min, lon_max, lat_min, lat_max])
    df.append(['cropped', lon_min_cropped, lon_max_cropped, lat_min_cropped, lat_max_cropped])
    df = pd.DataFrame(df, columns=['type', 'lon_min', 'lon_max', 'lat_min', 'lat_max'])
    df.to_csv(f'debug/{task_id}/align_data.csv', index=False)

    return lon_min_cropped, lon_max_cropped, lat_min_cropped, lat_max_cropped


def crop_orthophoto_pointcloud(task_id, lon_min_img, lon_max_img, lat_min_img, lat_max_img):
    img = cv2.imread(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_ALIGNED_CROPPED}')
    df = functions_geo.get_pointcloud_major(task_id)
    lon_min_pc, lon_max_pc, lat_min_pc, lat_max_pc = functions_geo.get_lon_lat_min_max(df)

    lon_min = max(lon_min_img, lon_min_pc)
    lon_max = min(lon_max_img, lon_max_pc)

    lat_min = max(lat_min_img, lat_min_pc)
    lat_max = min(lat_max_img, lat_max_pc)

    crop_image(task_id, img, lon_min, lon_max, lat_min, lat_max, lon_min_img, lon_max_img, lat_min_img, lat_max_img)
    crop_pointcloud(task_id, df, lon_min, lon_max, lat_min, lat_max)


def crop_image(task_id, img, lon_min, lon_max, lat_min, lat_max, lon_min_img, lon_max_img, lat_min_img, lat_max_img):
    height = img.shape[0]
    width = img.shape[1]

    x_min = int(round((lon_min - lon_min_img) / (lon_max_img - lon_min_img) * width, 0))
    x_max = int(round((lon_max - lon_min_img) / (lon_max_img - lon_min_img) * width, 0))

    y_min = int(round((lat_max_img - lat_max) / (lat_max_img - lat_min_img) * height, 0))
    y_max = int(round((lat_max_img - lat_min) / (lat_max_img - lat_min_img) * height, 0))

    img = img[y_min:y_max, x_min:x_max]
    cv2.imwrite(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_ALIGNED_CROPPED}', img)


def crop_pointcloud(task_id, df, lon_min, lon_max, lat_min, lat_max):
    df = df[(df['x'] >= lon_min) & (df['x'] <= lon_max) & (df['y'] >= lat_min) & (df['y'] <= lat_max)]
    df.to_csv(f'debug/{task_id}/{FILENAME_POINTCLOUD_MAJOR}', index=False)
