#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:50:07 2017

@author: pierluigiferrari
"""

import numpy as np
import cv2
import pickle

def load_camera_calib(filepath):
    '''
    Load a camera matrix and distortion coefficients from a pickle file.
    '''

    with open(filepath, "rb") as f:
        cam_calib = pickle.load(f)

    cam_matrix = cam_calib['cam_matrix']
    dist_coeffs = cam_calib['dist_coeffs']

    print("Camera matrix and distortion coefficients loaded.")

    return cam_matrix, dist_coeffs

def load_warp_params(filepath):
    '''
    Load warp parameters for a bird's eye perspective transformation
    from a pickle file.

    Returns:
        1. M: The warp matrix to transform an image from a car's dashboard camera
           to bird's eye (warped) view.
        2. Minv: The inverse of the above matrix to transform a bird's-eye-view image
           back to the original perspective.
        3. image_width_height: The expected orinal image width and height.
        4. warped_width_height: The expected warped image width and height.
        5. mppv: The meters-per-pixel ratio of the warped image in vertical direction.
        6. mpph: The meters-per-pixel ratio of the warped image in horizontal direction.
    '''

    with open(filepath, "rb") as f:
        warp_params = pickle.load(f)

    M = warp_params['M']
    Minv = warp_params['Minv']
    image_width_height = warp_params['image_width_height']
    warped_width_height = warp_params['warped_width_height']
    mppv = warp_params['mppv']
    mpph = warp_params['mpph']
    print("Warp parameters loaded.")

    return M, Minv, image_width_height, warped_width_height, mppv, mpph

def create_split_view(target_size, images, positions, sizes, captions=[]):
    '''
    Place images onto a rectangular canvas to create a split view.

    Arguments:
        target_size (tuple): The target size of the output canvas in the format
            (width, height). The output canvas will always have three color channels.
        images (list): A list containing the images to be placed onto the output canvas.
            The images can vary in size and can have one or three color channels.
        positions (list): A list containing the desired top left corner positions of
            the images in the output canvas in the format (x, y), where x refers
            to the horizontal coordinate and y refers to the vertical coordinate
            and both are non-negative integers.
        sizes (list): A list containing tuples with the desired sizes of the images
            in the format (width, height).
        captions (list, optional): A list containing either a caption string or
            `None` for each image. The list must have the same length as `images`.
            Defaults to an empty list, i.e. no captions will be added.

    Returns:
        The split view image of size `target_size`.
    '''

    assert len(images) == len(positions) == len(sizes), "`images`, `positions`, and `sizes` must have the same length, but it is `len(images) == {}`, `len(poisitons) = {}`, `len(sizes) == {}`".format(len(images), len(positions), len(sizes))

    x_max, y_max = target_size
    canvas = np.zeros((y_max, x_max, 3), dtype=np.uint8)

    for i, img in enumerate(images):

        # Resize the image
        if img.shape[0] != sizes[i][1] | img.shape[1] != sizes[i][0]:
            img = cv2.resize(img, dsize=sizes[i])

        # Place the resized image onto the canvas
        x, y = positions[i]
        w, h = sizes[i]
        # If img is grayscale, Numpy broadcasting will put the same intensity value for each the R, G, and B channels.
        # The indexing below protects against index-out-of-range issues.
        canvas[y:min(y + h, y_max), x:min(x + w, x_max), :] = img[:min(h, y_max - y), :min(w, x_max - x)]

        # Print captions onto the canvas if there are any
        if captions and (captions[i] is not None):
            cv2.putText(canvas, "{}".format(captions[i]), (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    return canvas
