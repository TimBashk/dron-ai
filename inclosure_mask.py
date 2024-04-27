import cv2
import numpy as np
import pandas as pd

from const import FILENAME_INCLOSURE_LINES, FILENAME_MASK_DRY, FILENAME_INCLOSURE_MASK, FILENAME_MASK_GREEN


def fit_line(line_dct):
    x1 = line_dct['x1']
    y1 = line_dct['y1']

    x2 = line_dct['x2']
    y2 = line_dct['y2']

    xs = np.array([x1, x2])
    ys = np.array([y1, y2])

    z = np.polyfit(xs, ys, 1)
    fit = np.poly1d(z)
    return fit


# def get_line_ys(img, fit):
#     width = img.shape[1]
#     ys = [fit(i) for i in range(width)]
#     return ys


def get_polygon_coord(img, fit):
    y_min, y_max = fit(0), fit(img.shape[1])
    return [0, y_min], [img.shape[1], y_max]


def is_between_lines(ys1, ys2, x, y):
    return ys1[x] <= y <= ys2[x] or ys2[x] <= y <= ys1[x]


def save_inclosure_mask(task_id, span_id):
    df = pd.read_csv(f'debug/{task_id}/spans/{span_id}/{FILENAME_INCLOSURE_LINES}')
    img = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_DRY}')  # we don't need this image. only it's shape.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.zeros(img.shape)

    line_dct = df.to_dict('records')[0]
    fit1 = fit_line(line_dct)
    coord1_1, coord1_2 = get_polygon_coord(img, fit1)  # get two corners of polygon from line1

    line_dct = df.to_dict('records')[1]
    fit2 = fit_line(line_dct)
    coord2_1, coord2_2 = get_polygon_coord(img, fit2)

    img_mask = img.copy()

    # for y_index in range(img.shape[0]):
    #     for x_index in range(img.shape[1]):
    #         if is_between_lines(ys1, ys2, x_index, y_index):
    #             img_mask[y_index][x_index] = 255
    #         else:
    #             img_mask[y_index][x_index] = 0

    a3 = np.array([[coord1_1, coord1_2, coord2_1, coord2_2]], dtype=np.int32)
    #im = np.zeros([240, 320], dtype=np.uint8)
    cv2.fillPoly(img_mask, a3, 255)

    img_mask = img_mask.astype('uint8')
    cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_INCLOSURE_MASK}', img_mask)
    return
