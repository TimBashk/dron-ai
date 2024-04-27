import os
import cv2
import math
import torch
import numpy as np
import segmentation_models_pytorch as smp

from const import (NN_DEVICE,
                   MODEL_PATH_GREEN_TREE, ENCODER_GREEN_TREE, ENCODER_WEIGHTS_GREEN_TREE, TILE_SIZE_GREEN_TREE,
                   MODEL_PATH_DRY_TREE, ENCODER_DRY_TREE, ENCODER_WEIGHTS_DRY_TREE, TILE_SIZE_DRY_TREE,
                   FILENAME_MASK_GREEN, FILENAME_MASK_GREEN_ED, FILENAME_MASK_DRY_ED,
                   FILENAME_ORTHOPHOTO, FILENAME_MASK_DRY, WIRES_MASK)



def add_border(image, tile_size):
    COLOR = [0, 0, 0]

    if image.shape[0] != tile_size:
        image = cv2.copyMakeBorder(image,
                                   0,
                                   tile_size - image.shape[0],
                                   0,
                                   0,
                                   cv2.BORDER_CONSTANT,
                                   value=COLOR)

    if image.shape[1] != tile_size:
        image = cv2.copyMakeBorder(image,
                                   0,
                                   0,
                                   0,
                                   tile_size - image.shape[1],
                                   cv2.BORDER_CONSTANT,
                                   value=COLOR)

    return image


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def remove_border(image, img_width, img_height):
    return image[:img_height, :img_width]


def predict_mask(model, image):
    if not np.any(image):
        return np.zeros(image.shape)

    x_tensor = torch.from_numpy(image).to(NN_DEVICE).unsqueeze(0)
    mask_predicted = model.predict(x_tensor)
    mask_predicted = mask_predicted.squeeze().cpu().numpy().round() * 255

    return mask_predicted


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def assemble_tiles(tiles, n):
    mask_predicted_chunks = list(get_chunks(tiles, n))
    image = cv2.vconcat(list([cv2.hconcat(i) for i in mask_predicted_chunks]))
    return image


def get_eroded_dilated_segmentation(task_id, span_id, mode):
    kernel = np.ones((3, 3), 'uint8')

    if mode == 'tree_green':
        mask = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_GREEN}')
    elif mode == 'tree_dry':
        mask = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_DRY}')

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = cv2.erode(mask, kernel, iterations=8)
    mask = cv2.dilate(mask, kernel, iterations=8)

    if mode == 'tree_green':
        cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_GREEN_ED}', mask)

    elif mode == 'tree_dry':
        cv2.imwrite(f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_DRY_ED}', mask)

    return


def get_mask(task_id, span_id, mode):
    if mode == 'tree_green':
        model_path = MODEL_PATH_GREEN_TREE
        encoder = ENCODER_GREEN_TREE
        encoder_weights = ENCODER_WEIGHTS_GREEN_TREE
        tile_size = TILE_SIZE_GREEN_TREE
        filepath_to_save = f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_GREEN}'

    elif mode == 'tree_dry':
        model_path = MODEL_PATH_DRY_TREE
        encoder = ENCODER_DRY_TREE
        encoder_weights = ENCODER_WEIGHTS_DRY_TREE
        tile_size = TILE_SIZE_DRY_TREE
        filepath_to_save = f'debug/{task_id}/spans/{span_id}/{FILENAME_MASK_DRY}'

    elif mode =='wires':
        model_path = MODEL_PATH_WIRES
        encoder = ENCODER_DRY_TREE
        encoder_weights = ENCODER_WEIGHTS_DRY_TREE
        tile_size = TILE_SIZE_DRY_TREE
        filepath_to_save = f'debug/{task_id}/spans/{span_id}/{WIRES_MASK}'

    if NN_DEVICE == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)

    preprocess_input = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    image = cv2.imread(f'debug/{task_id}/spans/{span_id}/{FILENAME_ORTHOPHOTO}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_width = image.shape[1]
    img_height = image.shape[0]

    masks = list()
    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            tile = add_border(tile, tile_size)
            tile = preprocess_input(tile)
            tile = to_tensor(tile)
            mask = predict_mask(model, tile)
            masks.append(mask)

    n_chunks = math.ceil(img_width / tile_size)  # how many tiles in image by width
    mask = assemble_tiles(masks, n_chunks)
    mask = remove_border(mask, img_width, img_height)
    cv2.imwrite(filepath_to_save, mask)

    #get_eroded_dilated_segmentation(task_id, span_id, mode)  # you can comment this if needed and insert FILENAME_MASK_GREEN in FILENAME_MASK_GREEN_ED
    return

#masks, mask = get_mask(image_path, MODEL_PATH_GREEN_TREE, ENCODER_GREEN_TREE, ENCODER_WEIGHTS_GREEN_TREE, TILE_SIZE_GREEN_TREE)