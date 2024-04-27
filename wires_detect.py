import numpy as np
import cv2
import local_maxima
import draw
from scipy import ndimage
from const import WIRES_MASK
import pandas as pd


def get_inclosure_area_width_by_voltage_class(task_id):
    inclosure_area_width = 2

    voltage_class = get_span_voltage_class(task_id)

    if voltage_class < 1:
        inclosure_area_width = 2
    elif voltage_class >= 1 and voltage_class <= 20:
        inclosure_area_width = 10
    elif voltage_class == 35:
        inclosure_area_width = 15
    elif voltage_class in (150, 220):
        inclosure_area_width = 25
    elif voltage_class in (300, 400, 500):
        inclosure_area_width = 30
    elif voltage_class == 750:
        inclosure_area_width = 40
    elif voltage_class == 1150:
        inclosure_area_width = 55

    return inclosure_area_width


def get_span_voltage_class(task_id):
    voltage_class = 2
    df = pd.read_csv(f'debug/{task_id}/spans.csv')
    voltage_class = list(set(df['voltageclass']))[0]
    voltage_class = int(voltage_class.replace('VC', ''))
    return voltage_class


def mult_wires_and_inclosure_masks(task_id, span_id):
    inclosure_mask_img = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/inclosure_mask.png'), cv2.COLOR_BGR2GRAY)
    wires_mask = cv2.cvtColor(cv2.imread(f'debug/{task_id}/spans/{span_id}/{WIRES_MASK}'), cv2.COLOR_BGR2GRAY)
    result_mask = inclosure_mask_img * wires_mask
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/clear_wires_mask.png',result_mask)
    return


def get_koeff_b_by_span_id(task_id):
    koeff_b = 0
    df = pd.read_csv(f'debug/{task_id}/spans.csv')
    spans = list(df['span_id'])
    b_koeff = np.array([])

    for span_id in spans:
        df_ = df[df['span_id'] == int(span_id)]
        print(list(df_['lon1'])[0])
        print(list(df_['lat1'])[0], '\n')
        lat = (list(df_['lat1'])[0], list(df_['lat2'])[0])
        lon = (list(df_['lon1'])[0], list(df_['lon2'])[0])
        koeff = np.polyfit(lon, lat, 1)
        b_koeff = np.append(b_koeff, koeff[0])
    return b_koeff


def get_labels_coord(img):
    image_labelled, nr_objects = ndimage.label(img)
    coords = []
    for label in range(1, nr_objects + 1):
        result = np.where(image_labelled == label)
        coords.append((label, result[0], result[1], len(result[0])))
    return coords


def get_distance_between_wires(task_id, span_id):
    img = cv2.imread(f'debug/{task_id}/spans/{span_id}/mask_wires.png')
    #img = cv2.imread('debug/wires2.png')
    kernel = np.ones((10, 10), 'uint8')
    dil_img = cv2.dilate(img, kernel, 16)

    labels_coords = get_labels_coord(dil_img)
    label_coords = labels_coords.sort(key=lambda x: -x[3])

    b_koeff = np.array([])

    for index, label_coords in enumerate(labels_coords):
        if index >= 3:
            break
        x = label_coords[2]
        y = label_coords[1]
        koeff = np.polyfit(x, y, 1)
        b_koeff = np.append(b_koeff, koeff[1])
    if b_koeff.shape[0] > 0:
        distance_between_wires = np.max(b_koeff) - np.min(b_koeff)
    else:
        distance_between_wires = 0
    scale = 1
    #scale = draw.get_scale(task_id, span_id)
    distance_between_wires = distance_between_wires * scale

    return span_id, distance_between_wires


def get_span_koeff(task_id, span_id):

    koeff = np.polyfit(span_lon, span_lat, 1)
    return koeff


