#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:50:07 2017

@author: pierluigiferrari
"""

import numpy as np
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
