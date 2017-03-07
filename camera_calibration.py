#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 01:36:40 2017

@author: pierluigiferrari
"""

import numpy as np
import cv2
import glob
import pickle

def get_calibration_points(images, nx, ny):
    '''
    Generate two lists of calibration points from a set of calibration images
    of chess boards to needed for `cv2.calibrateCamera()`.
    
    It is recommended that `images` contain at least 20 images. All images
    are expected to be of identical size and to contain the same, complete
    chess board pattern.
    
    Args:
        images (array-like): A list of file names of the images to be
            used for calibration.
        nx (int): The number of horizontal inner corners (i.e. corners where two
            white and two black tiles meet) of the chess board.
        ny (int): The number of vertical inner corners (i.e. corners where two
            white and two black tiles meet) of the chess board.
            
    Returns:
        object_points (list): The list of 3-D object points for calibration.
        image_points (list): The list of 2-D image points for calibration.
    '''
    
    image_size = []
    
    # Arrays to store object points and image points
    # of all calibration images for `cv2.calibrateCamera()`.
    object_points = [] # 3-D points in real world space
    image_points = [] # 2-D points in image plane.

    # All calibration images are expected to contain the same calibration pattern,
    # so the object points are the same for all images.
    # Format: (0,0,0), (1,0,0), (2,0,0), ...., (8,5,0)
    # The third coordinate is always zero as the points lie in a plane.
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Step through the list and search for chess board corners
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        size = (img.shape[1], img.shape[0])
        if i == 0:
            image_size = size
        if size != image_size:
            raise ValueError("Expected all images to have identical size, but found varying sizes.")
        image_size = size
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            object_points.append(objp)
            image_points.append(corners)

    return object_points, image_points, image_size

def calibrate_camera(filepaths, nx, ny):
    # Compute camera matrix and distortion coefficients
    
    # Get the calibration points
    object_points, image_points, image_size = get_calibration_points(images, nx, ny)
    
    # Compute camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    # Save the camera calibration result to disk (we won't worry about rvecs / tvecs)
    cam_calib = {"cam_matrix": mtx,
                   "dist_coeffs": dist}
    with open("cam_calib.p", "wb") as f:
        pickle.dump(cam_calib, f)
    
    return mtx, dist

# Run the calibration process

# Specify the filepaths to the calibration images
# The images are expected to contain only chessboard patterns and bright background
images = glob.glob('camera_calib/calibration*.jpg')
# Run the calibration
calibrate_camera(images, 9, 6) # Our images contain 9x6 chessboard patterns