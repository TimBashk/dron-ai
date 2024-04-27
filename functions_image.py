import cv2
import numpy as np
from const import FILENAME_ORTHOPHOTO_ALIGNED_CROPPED


def crop_image(arr, border):

    # input: 1. grey or coloured image 2. border grayed color as integer (e.g. 0 or 255 to remove black border or white border)

    if len(arr.shape) == 3:  # if input is coloured image
        arr_grey = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    else:
        arr_grey = arr

    for y_start in range(arr_grey.shape[0]):
        if set(np.unique(arr_grey[y_start])) == {border}:
            continue
        else:
            break

    for y_end in reversed(range(0, arr_grey.shape[0])):
        if set(np.unique(arr_grey[y_end])) == {border}:
            continue
        else:
            break

    for x_start in range(arr_grey.shape[1]):
        if set(np.unique(arr_grey[:, x_start])) == {border}:
            continue
        else:
            break

    for x_end in reversed(range(0, arr_grey.shape[1])):
        if set(np.unique(arr_grey[:, x_end])) == {border}:
            continue
        else:
            break

    return arr[y_start:y_end, x_start:x_end]


def get_orthophoto_major(task_id):
    img = cv2.imread(f'debug/{task_id}/{FILENAME_ORTHOPHOTO_ALIGNED_CROPPED}')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img