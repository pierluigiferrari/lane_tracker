#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:50:07 2017

@author: pierluigiferrari
"""

import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

def load_camera_calib(filepath):
    
    with open(filepath, "rb") as f:
        cam_calib = pickle.load(f)

    cam_matrix = cam_calib['cam_matrix']
    dist_coeffs = cam_calib['dist_coeffs']
    
    print("Camera matrix and distortion coefficients loaded.")
    
    return cam_matrix, dist_coeffs

def load_warp_params(filepath):

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

def bilateral_adaptive_threshold(img, ksize=30, C=0, mode='floor', true_value=255, false_value=0):
    '''
    Perform adaptive color thresholding on a single-channel input image.
    
    The function uses a cross-shaped filter kernel of 1-pixel thickness.
    
    The intensity value of a given pixel needs to be relatively higher (or lower, depending on the mode)
    than the average intensity value of either both the left and right sides or both the upper and lower
    sides of the kernel cross. This means in order to exceed the threshold, a pixel does not just need
    to be brighter than its average neighborhood, but instead it needs to be brighter than both sides
    of its neighborhood independently in either horizontal or vertical direction.
    
    This is useful for filtering lane line pixels, because the area on both sides of a lane line is darker
    than the lane line itself.
    '''
    
    mask = np.full(img.shape, false_value, dtype=np.uint8)

    kernel_l = np.array([[1] * (ksize) + [-ksize]], dtype=np.int16)
    kernel_r = np.array([[-ksize] + [1] * (ksize)], dtype=np.int16)
    kernel_u = np.array([[1]] * (ksize) + [[-ksize]], dtype=np.int16)
    kernel_d = np.array([[-ksize]] + [[1]] * (ksize), dtype=np.int16)
    
    if mode == 'floor':
        delta = C * ksize
    elif mode == 'ceil':
        delta = -C * ksize
    else: raise ValueError("Unexpected mode value. Expected value is 'floor' or 'ceil'.")

    left_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_l, anchor=(ksize,0), delta=delta, borderType=cv2.BORDER_CONSTANT)
    right_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_r, anchor=(0,0), delta=delta, borderType=cv2.BORDER_CONSTANT)
    up_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_u, anchor=(0,ksize), delta=delta, borderType=cv2.BORDER_CONSTANT)
    down_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_d, anchor=(0,0), delta=delta, borderType=cv2.BORDER_CONSTANT)

    if mode == 'floor':
        mask[((0 > left_thresh) & (0 > right_thresh)) | ((0 > up_thresh) & (0 > down_thresh))] = true_value
    elif mode == 'ceil':
        mask[((0 < left_thresh) & (0 < right_thresh)) | ((0 < up_thresh) & (0 < down_thresh))] = true_value

    return mask

class LaneTracker:
    
    def __init__(self, img_size, warped_size, cam_matrix, dist_coeffs, warp_matrices, mpp_conversion, n_fail, n_reset, n_average, print_frame_count=False):
        # The image size, format: (width, height)
        self.img_size = img_size
        # The output size of the perspective transformation, format: (width, height)
        self.warped_size = warped_size
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.M = warp_matrices[0]
        self.Minv = warp_matrices[1]
        self.mppv = mpp_conversion[0]
        self.mpph = mpp_conversion[1]
        # The number of frames since the last detection after which we revert to sliding window search
        self.n_reset = n_reset
        # The number of frames since the last detection after which we print a failure message on the image
        self.n_fail = n_fail
        # The number of past frames for which to store fit data and to average over
        self.n_average = n_average
        # Counter: How many frames ago was the last valid lane detection  
        self.last_detection = n_reset + 1
        # Did we find any lane pixels in this frame?
        self.detected_pixels = False
        # Did we find valid lane lines in this frame?
        self.valid_lane_lines = False
        # The last `n_average` polynomial coefficients...
        self.left_fit_coeffs = []
        self.right_fit_coeffs = []
        # ...and the last coefficients by themselves...
        self.last_left_coeffs = None
        self.last_right_coeffs = None
        # ...and their averages for the smoothed lane lines
        self.left_avg_coeffs = None
        self.right_avg_coeffs = None
        # The graph points of the most recent lane lines
        self.left_avg_y = np.array([])
        self.left_avg_x = np.array([])
        self.right_avg_y = np.array([])
        self.right_avg_x = np.array([])
        # The current lane pixels
        self.left_y = None
        self.left_x = None
        self.right_y = None
        self.right_x = None
        # The current window centroids
        self.left_window_centroids = None
        self.right_window_centroids = None
        # The current curve radius in meters
        self.left_curve_radius = None
        self.right_curve_radius = None
        self.average_curve_radius = None
        self.average_curve_radii = []
        # The current distance from the center of the lane in meters
        self.eccentricity = None
        # Counter to print the frame number on the processed image. Useful for debugging.
        self.print_frame_count = print_frame_count
        self.counter = 0
        self.success = 0
    
    def get_success_ratio(self):
        # Returns on what fraction of the processes images LaneTracker detected lane lines
        
        return self.success / self.counter, self.success, self.counter
    
    def filter_lane_points(self,
                           img,
                           filter_mode='bilateral',
                           ksize_r=25,
                           C_r=8,
                           ksize_b=35,
                           C_b=5,
                           mask_noise=False,
                           ksize_noise=65,
                           C_noise=10,
                           noise_thresh=135):
        '''
        Filter an image to isolate lane lines and return a binary version.
        
        All image color space conversion, thresholding, filtering and morphing
        happens inside this method. It takes an RGB color image as input and
        returns a binary filtered version.
        '''
        
        # Define structuring elements for cv2 functions
        strel_lab_b = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(55,55))
        strel_rgb_r = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(29,29))
        strel_open = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
        # Extract RGB R-channel and LAB B-channel
        rgb_r_channel = img[:,:,0]
        lab_b_channel = (cv2.cvtColor(img, cv2.COLOR_RGB2LAB))[:,:,2]
        # Apply tophat morphology
        rgb_r_tophat = cv2.morphologyEx(rgb_r_channel, cv2.MORPH_TOPHAT, strel_rgb_r, iterations=1)
        lab_b_tophat = cv2.morphologyEx(lab_b_channel, cv2.MORPH_TOPHAT, strel_lab_b, iterations=1)
        if filter_mode == 'bilateral':
            # Apply bilateral adaptive color thresholding
            rgb_r_thresh = bilateral_adaptive_threshold(rgb_r_tophat, ksize=ksize_r, C=C_r)
            lab_b_thresh = bilateral_adaptive_threshold(lab_b_tophat, ksize=ksize_b, C=C_b)
        elif filter_mode == 'neighborhood':
            rgb_r_thresh = cv2.adaptiveThreshold(rgb_r_channel, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=ksize_r, C=-C_r)
            lab_b_thresh = cv2.adaptiveThreshold(lab_b_channel, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=ksize_b, C=-C_b)
        else:
            raise ValueError("Unexpected filter mode. Expected modes are 'bilateral' or 'neighborhood'.")
        if mask_noise: # Merge both color channels and the noise mask
            # Create a mask to filter out noise such as trees and other greenery based on the LAB B-channel
            noise_mask_part1 = cv2.inRange(lab_b_channel, noise_thresh, 255) # This catches the noise, but unfortunately also the yellow line, therefore...
            noise_mask_part2 = bilateral_adaptive_threshold(lab_b_channel, ksize=ksize_noise, C=C_noise) # ...this brings the yellow line back...
            noise_bool = np.logical_or(np.logical_not(noise_mask_part1), noise_mask_part2) # ...once we combine the two.
            noise_mask = np.zeros_like(rgb_r_channel, dtype=np.uint8)
            noise_mask[noise_bool] = 255

            merged_bool = np.logical_and(np.logical_or(rgb_r_thresh, lab_b_thresh), noise_mask)
            merged = np.zeros_like(rgb_r_channel, dtype=np.uint8)
            merged[merged_bool] = 255
        else: # Only merge the two color channels
            merged_bool = np.logical_or(rgb_r_thresh, lab_b_thresh)
            merged = np.zeros_like(rgb_r_channel, dtype=np.uint8)
            merged[merged_bool] = 255
        
        # Apply open morphology
        opened = cv2.morphologyEx(merged, cv2.MORPH_OPEN, strel_open, iterations=1)

        return opened
    
    def sliding_window_search(self,
                              img,
                              window_width,
                              window_height,
                              search_range,
                              mu,
                              no_success_limit,
                              start_slice=0.25,
                              ignore_sides=360,
                              ignore_bottom=30,
                              partial=1,
                              print_diagnostics=False):
        '''
        Perform an iterative lane pixel search and store the detected lane pixels.
        
        This method takes as input the output of `filter_lane_points()` (the image
        should be warped) and tries to find lane pixels in it. It scans the image from
        the bottom to the top using rectangular search windows that are convolved with
        the image. Each successive search window covers only a certain range around the
        horizontal center of its preceding search window.
        
        This method is only used if no lane line information from preceding frames
        is available, otherwise the band search method below is used.
        '''
        
        if print_diagnostics: print("Using sliding window search.")
        
        # The lists to save all the left and right window centroids
        left_window_centroids = []
        right_window_centroids = []
        # The lists to save the relative change of the left and right window centroids
        left_window_centroids_differences = []
        right_window_centroids_differences = []
        
        img_width = img.shape[1]
        img_height = img.shape[0] - ignore_bottom
        img_center = int(img_width/2) # The horizontal center of the image
        y_start = int((1-start_slice)*img_height) # The horizontal image slice over which to take the sum below

        window = np.ones(window_width) # The 1-D convolution filter
        nlevels = int((partial*img_height)/window_height) # The number of search steps

        # Create empty lists to save the left and right lane pixel indices
        leftx, lefty, rightx, righty = [], [], [], []

        # Sum all pixels vertically along the bottom image slice, convolve the resulting array with
        # `window`, and pick thee horizontal index with the highest value as the first centroid.
        # Do this once for the left search range and once for the right.
        left_sum = np.sum(img[y_start:(img_height),ignore_sides:img_center], axis=0)
        if np.any(left_sum):
            # First, find the first centroid
            conv = np.convolve(window, left_sum)
            n_max = len(conv[conv == np.amax(conv)]) # Count how many maximum values there are
            max_inds = np.argpartition(-conv, n_max-1) # Get the (unsorted) indices of all max values
            max_center = int((np.amin(max_inds[:n_max]) + np.amax(max_inds[:n_max])) / 2) # ...and take their middle index as the centroid
            left_centroid = max_center - int(window_width/2) + ignore_sides
            # Next, find the non-zero pixels within the window around the centroid
            roi = img[img_height-window_height:img_height, left_centroid-int(window_width/2):left_centroid+int(window_width/2)]
            nonzero = roi.nonzero() # Compute the indices of all non-zero values in the ROI
            # Now correct those indices for the ROI's position within the global image...
            nonzeroy = np.array(nonzero[0]) + img_height - window_height
            nonzerox = np.array(nonzero[1]) + left_centroid - int(window_width/2)
            # ...and add the resulting indices to our list of lane line pixel indices
            lefty.append(nonzeroy)
            leftx.append(nonzerox)
        else:
            left_centroid = int(img_width*0.4)
        # Now perform the above procedure for the right search range
        right_sum = np.sum(img[y_start:img_height,img_center:(img_width-ignore_sides)], axis=0)
        if np.any(right_sum):
            # First, find the first centroid
            conv = np.convolve(window, right_sum)
            n_max = len(conv[conv == np.amax(conv)]) # Count how many maximum values there are
            max_inds = np.argpartition(-conv, n_max-1) # Get the (unsorted) indices of all max values
            max_center = int((np.amin(max_inds[:n_max]) + np.amax(max_inds[:n_max])) / 2) # ...and take their middle index as the centroid
            right_centroid = max_center - int(window_width/2) + img_center
            # Next, find the non-zero pixels within the window around the centroid
            roi = img[img_height-window_height:img_height, right_centroid-int(window_width/2):right_centroid+int(window_width/2)]
            nonzero = roi.nonzero() # Compute the indices of all non-zero values in the ROI
            # Now correct those indices for the ROI's position within the global image...
            nonzeroy = np.array(nonzero[0]) + img_height - window_height
            nonzerox = np.array(nonzero[1]) + right_centroid - int(window_width/2)
            # ...and add the resulting indices to our list of lane line pixel indices
            righty.append(nonzeroy)
            rightx.append(nonzerox)
        else:
            right_centroid = int(img_width*0.6)

        # Append the found initial centroids to the list
        left_window_centroids.append(left_centroid)
        right_window_centroids.append(right_centroid)

        # Counters to increment each consecutive time we didn't find any pixels on a level
        left_no_success = 0
        right_no_success = 0

        # Momentum for the search range
        mu = mu # Momentum factor for search range adjustment
        left_search_range_min = -search_range
        left_search_range_max = search_range
        right_search_range_min = -search_range
        right_search_range_max = search_range

        # Now do the same for all subsequent levels
        for level in range(1, nlevels):

            # Same procedure as above, but this time across the the entire width of the image.
            # We'll pick the maximal values for the left and right centroids below.
            sm = np.sum(img[img_height-(1+level)*window_height:img_height-level*window_height,:], axis=0)
            conv = np.convolve(window, sm)

            # We only keep searching for lane points if we weren't unsuccessful (i.e. found nothing) too often before
            if left_no_success < no_success_limit:
                # Set the search range over which we optimize
                left_min_index = max(left_centroid + left_search_range_min + int(window_width/2), 0)
                left_max_index = min(left_centroid + left_search_range_max + int(window_width/2), img_width)
                conv_left = conv[left_min_index:left_max_index]
                # If any points were found within the search range
                if np.any(conv_left):
                    n_max = len(conv_left[conv_left == np.amax(conv_left)]) # Count how many maximum values there are
                    max_inds = (np.argpartition(-conv_left, n_max-1))[:n_max] # Get the (unsorted) indices of all max values
                    max_center = int(np.ceil((np.amin(max_inds) + np.amax(max_inds)) / 2)) # ...and take their middle index as the centroid
                    left_centroid = max_center + left_min_index - int(window_width/2)
                    left_window_centroids.append(left_centroid)
                    left_window_centroids_differences.append(left_window_centroids[-1] - left_window_centroids[-2])
                    # Reset the no_success counter
                    left_no_success = 0
                    # Next, find the non-zero pixels within the window around the new centroid
                    # Slice the window around the centroid to be the region of interest (ROI)
                    roi = img[img_height-(1+level)*window_height:img_height-level*window_height, left_centroid-int(window_width/2):left_centroid+int(window_width/2)]
                    nonzero = roi.nonzero() # Compute the indices of all non-zero values in the ROI
                    # Now correct those indices for the ROI's position within the global image...
                    nonzeroy = np.array(nonzero[0]) + img_height - (1+level)*window_height
                    nonzerox = np.array(nonzero[1]) + left_centroid - int(window_width/2)
                    # ...and add the resulting indices to our list of lane line pixel indices
                    lefty.append(nonzeroy)
                    leftx.append(nonzerox)
                    # Adjust the search range for the next centroid
                    left_search_range_min += int(mu * left_window_centroids_differences[-1])
                    left_search_range_max += int(mu * left_window_centroids_differences[-1])
                # If no points were found within the search range
                else:
                    # ...go in the same direction as the right side (maybe it has more luck)
                    if (len(right_window_centroids_differences) > 0) & (right_no_success == 0):
                        left_centroid += int(right_window_centroids_differences[-1])
                        left_window_centroids.append(left_centroid)
                    else:
                        left_window_centroids.append(left_centroid)
                    left_no_success += 1
                    if left_no_success >= no_success_limit:
                        del left_window_centroids[-no_success_limit:]

            # Do the same thing for the right lane line that we did for the left lane line above
            if right_no_success < no_success_limit:
                right_min_index = max(right_centroid + right_search_range_min + int(window_width/2), 0)
                right_max_index = min(right_centroid + right_search_range_max + int(window_width/2), img_width)
                conv_right = conv[right_min_index:right_max_index]
                if np.any(conv_right):
                    n_max = len(conv_right[conv_right == np.amax(conv_right)]) # Count how many maximum values there are
                    max_inds = (np.argpartition(-conv_right, n_max-1))[:n_max] # Get the (unsorted) indices of all max values
                    max_center = int(np.ceil((np.amin(max_inds) + np.amax(max_inds)) / 2)) # ...and take their middle index as the centroid
                    right_centroid = max_center + right_min_index - int(window_width/2)
                    right_window_centroids.append(right_centroid)
                    right_window_centroids_differences.append(right_window_centroids[-1] - right_window_centroids[-2])
                    right_no_success = 0
                    # Next, find the non-zero pixels within the window around the new centroid
                    # Slice the window around the centroid to be the region of interest (ROI)
                    roi = img[img_height-(1+level)*window_height:img_height-level*window_height, right_centroid-int(window_width/2):right_centroid+int(window_width/2)]
                    nonzero = roi.nonzero() # Compute the indices of all non-zero values in the ROI
                    # Now correct those indices for the ROI's position within the global image...
                    nonzeroy = np.array(nonzero[0]) + img_height - (1+level)*window_height
                    nonzerox = np.array(nonzero[1]) + right_centroid - int(window_width/2)
                    # ...and add the resulting indices to our list of lane line pixel indices
                    righty.append(nonzeroy)
                    rightx.append(nonzerox)
                    # Now adjust the search range for the next centroid
                    right_search_range_min += int(mu * right_window_centroids_differences[-1])
                    right_search_range_max += int(mu * right_window_centroids_differences[-1])
                # If no points were found within the search range...
                else:
                    # ...go in the same direction as the right side (maybe it has more luck)
                    if (len(left_window_centroids_differences) > 0) & (left_no_success == 0):
                        right_centroid += int(left_window_centroids_differences[-1])
                        right_window_centroids.append(right_centroid)
                    else:
                        right_window_centroids.append(right_centroid)
                    right_no_success += 1
                    if right_no_success >= no_success_limit:
                        del right_window_centroids[-no_success_limit:]
        
        if (len(leftx) > 0) & (len(rightx) > 0):
            if (np.concatenate(leftx).size > 0) & (np.concatenate(rightx).size > 0):
                self.left_y = np.concatenate(lefty)
                self.left_x = np.concatenate(leftx)
                self.right_y = np.concatenate(righty)
                self.right_x = np.concatenate(rightx)
                self.detected_pixels = True
                self.left_window_centroids = left_window_centroids
                self.right_window_centroids = right_window_centroids
                if print_diagnostics: print("Lane pixels found.")
            else:
                self.detected_pixels = False
                if print_diagnostics: print("No lane pixels found.")
        else:
            self.detected_pixels = False
            if print_diagnostics: print("No lane pixels found.")
    
    def band_search(self, img, bandwidth, ignore_bottom=30, partial=1, print_diagnostics=False):
        '''
        Search the input image for lane line pixels and store the detected pixels.
        
        This method takes as input the output of `filter_lane_points()` (the image
        should be warped) and tries to find lane pixels in it. It searches for pixels
        in a specified band around the fitted lane line polynomial of a previously
        detected lane line.
        
        This method is used whenever previous lane line information is available.
        '''
        
        if print_diagnostics: print("Using band search.")
        
        img1 = np.copy(img)
        
        img1[img1.shape[0]-ignore_bottom:,:] = 0
        img1[:img1.shape[0]*(1-partial),:] = 0
        
        # Get the indices of all non-zero pixels
        nonzero = img1.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Filter those non-zero pixels that lie within `bandwidth` pixels around the previous polynomial
        left_lane_inds = ((nonzerox > (self.last_left_coeffs[0] * (nonzeroy**2)
                                       + self.last_left_coeffs[1] * nonzeroy
                                       + self.last_left_coeffs[2]
                                       - bandwidth))
                          & (nonzerox < (self.last_left_coeffs[0] * (nonzeroy**2)
                                         + self.last_left_coeffs[1] * nonzeroy
                                         + self.last_left_coeffs[2]
                                         + bandwidth)))
        right_lane_inds = ((nonzerox > (self.last_right_coeffs[0] * (nonzeroy**2)
                                        + self.last_right_coeffs[1] * nonzeroy
                                        + self.last_right_coeffs[2]
                                        - bandwidth))
                           & (nonzerox < (self.last_right_coeffs[0] * (nonzeroy**2)
                                          + self.last_right_coeffs[1] * nonzeroy
                                          + self.last_right_coeffs[2]
                                          + bandwidth)))

        if (nonzerox[left_lane_inds].size != 0) & (nonzerox[right_lane_inds].size != 0):
            self.left_y = nonzeroy[left_lane_inds]
            self.left_x = nonzerox[left_lane_inds]
            self.right_y = nonzeroy[right_lane_inds]
            self.right_x = nonzerox[right_lane_inds]
            self.detected_pixels = True
            if print_diagnostics: print("Lane pixels found.")
        else:
            self.detected_pixels = False
            if print_diagnostics: print("No lane pixels found.")

    def fit_poly(self):
        # Fit a second-order polynomial to each the left and right point sets.
        # `left_fit_coeffs` and `right_fit_coeffs` contain the coefficients for the polynomials.
        
        left_fit_coeffs = np.polyfit(self.left_y, self.left_x, 2)
        right_fit_coeffs = np.polyfit(self.right_y, self.right_x, 2)

        return left_fit_coeffs, right_fit_coeffs
    
    def get_poly_points(self, left_fit_coeffs, right_fit_coeffs, partial=1):
        # Return the graph points of a second-order polynomial given its coefficients.
        
        img_height = self.warped_size[1]
        img_width = self.warped_size[0]
        
        # Compute the points to plot the polynomials...
        ploty = np.linspace(img_height*(1-partial), img_height-1, img_height*partial)
        left_fitx = left_fit_coeffs[0]*ploty**2 + left_fit_coeffs[1]*ploty + left_fit_coeffs[2]
        right_fitx = right_fit_coeffs[0]*ploty**2 + right_fit_coeffs[1]*ploty + right_fit_coeffs[2]
        
        #...but keep only those that lie within the image
        left_fit_x = left_fitx[(left_fitx <= img_width-1) & (left_fitx >= 0)]
        right_fit_x = right_fitx[(right_fitx <= img_width-1) & (right_fitx >= 0)]
        left_fit_y = np.linspace(img_height-len(left_fit_x), img_height-1, len(left_fit_x))
        right_fit_y = np.linspace(img_height-len(right_fit_x), img_height-1, len(right_fit_x))
        
        return left_fit_y.astype(np.int), left_fit_x.astype(np.int), right_fit_y.astype(np.int), right_fit_x.astype(np.int)
           
    def get_curve_radius(self):
        # Compute the curve radius in meters.
        
        # In order to do that, use the metric conversion factors `mppv` and `mpph` to fit the curve polynomials.
        left_fit_meters = np.polyfit(self.left_y*self.mppv, self.left_x*self.mpph, 2)
        right_fit_meters = np.polyfit(self.right_y*self.mppv, self.right_x*self.mpph, 2)
        
        y_eval = self.warped_size[1]
        
        self.left_curve_radius = int(((1 + (2 * left_fit_meters[0] * y_eval * self.mppv + left_fit_meters[1])**2)**1.5) 
                                     / np.absolute(2*left_fit_meters[0]))
        self.right_curve_radius = int(((1 + (2 * right_fit_meters[0] * y_eval * self.mppv + right_fit_meters[1])**2)**1.5)
                                      / np.absolute(2*right_fit_meters[0]))
        average_curve_radius = int(0.5 * (self.left_curve_radius + self.right_curve_radius))
        self.average_curve_radii.append(average_curve_radius)
        # Remove the oldest curve radius if the list gets too long
        if len(self.average_curve_radii) > self.n_average:
            self.average_curve_radii.pop(0)
        real_curve_radii = [radius for radius in self.average_curve_radii if radius > 0]
        self.average_curve_radius = int(np.average(real_curve_radii))
        
    def get_eccentricity(self):
        # Compute the horizontal distance of the car from the center of the lane in meters.
        
        left = self.left_avg_x[-1]
        right = self.right_avg_x[-1]
        mid = int(self.warped_size[0] / 2)
        dx1 = mid - left
        dx2 = right - mid
        self.eccentricity = ((dx1 - dx2) / 2) * self.mpph
                           
    def check_validity(self, left_fit_coeffs, right_fit_coeffs, print_diagnostics=False):
        '''
        Decide whether the graphs of two given sets of second-order polynomial coefficients
        constitute valid lane lines.
        '''
        
        left_fit_y, left_fit_x, right_fit_y, right_fit_x = self.get_poly_points(left_fit_coeffs, right_fit_coeffs)
        
        # Step 1: Check whether the two lines lie within a plausible distance from one another for three distinct y-values
        
        y1 = self.warped_size[0] - 1 # For the y-value, take the bottom of the picture.
        y2 = self.warped_size[0] - int(min(len(left_fit_y), len(right_fit_y)) * 0.25) # For the second y-value, take a value between y1 and the top-most available value.
        y3 = self.warped_size[0] - int(min(len(left_fit_y), len(right_fit_y)) * 0.5)
        
        # Compute the respective x-values for both polynomials
        x1l = left_fit_coeffs[0] * (y1**2) + left_fit_coeffs[1] * y1 + left_fit_coeffs[2]
        x1r = right_fit_coeffs[0] * (y1**2) + right_fit_coeffs[1] * y1 + right_fit_coeffs[2]
        x2l = left_fit_coeffs[0] * (y2**2) + left_fit_coeffs[1] * y2 + left_fit_coeffs[2]
        x2r = right_fit_coeffs[0] * (y2**2) + right_fit_coeffs[1] * y2 + right_fit_coeffs[2]
        x3l = left_fit_coeffs[0] * (y3**2) + left_fit_coeffs[1] * y3 + left_fit_coeffs[2]
        x3r = right_fit_coeffs[0] * (y3**2) + right_fit_coeffs[1] * y3 + right_fit_coeffs[2]
        
        # Compute the L1-norms of their differences
        x1_diff = abs(x1l - x1r)
        x2_diff = abs(x2l - x2r)
        x3_diff = abs(x3l - x3r)
        
        min_dist_y1 = 150
        max_dist_y1 = 245
        min_dist_y2 = 140
        max_dist_y2 = 265
        min_dist_y3 = 125
        max_dist_y3 = 290
        if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
            self.valid_lane_lines = False
            if print_diagnostics:
                print("No valid lane lines found, violated distance criterion: x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f} (min_dist_y1 == {}, max_dist_y1 == {}, min_dist_y2 == {}, max_dist_y2 == {}, min_dist_y3 == {}, max_dist_y3 == {})".format(x1_diff, x2_diff, x3_diff,
                        min_dist_y1, max_dist_y1, min_dist_y2, max_dist_y2, min_dist_y3, max_dist_y3))
            return
        
        # Step 2: Check whether the line derivatives are similar for two distinct y-values
        # (Since our polynomial has degree 2 and its derivative has degree 1, the derivative values at two
        #  distinct points completely determine the polynomial up to a constant shift parameter)
        
        # dx/dy = 2Ay + B
        left_y1_dx = 2 * left_fit_coeffs[0] * y1 + left_fit_coeffs[1]
        left_y3_dx = 2 * left_fit_coeffs[0] * y3 + left_fit_coeffs[1]
        right_y1_dx = 2 * right_fit_coeffs[0] * y1 + right_fit_coeffs[1]
        right_y3_dx = 2 * right_fit_coeffs[0] * y3 + right_fit_coeffs[1]

        # Compute the L1-norm of the difference of the derivatives at the two y-values.
        norm1 = abs(left_y1_dx - right_y1_dx)
        norm2 = abs(left_y3_dx - right_y3_dx)
        
        # If either of the two norms is larger than the threshold level,
        # return `False`.
        thresh = 0.46
        if (norm1 >= thresh) | (norm2 >= thresh):
            self.valid_lane_lines = False
            if print_diagnostics:
                print("No valid lane lines found, violated derivative criterion: norm1 == {:.3f}, norm2 == {:.3f} (thresh == {}). Distance: x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f} (min_dist_y1 == {}, max_dist_y1 == {}, min_dist_y2 == {}, max_dist_y2 == {}, min_dist_y3 == {}, max_dist_y3 == {}".format(norm1, norm2, thresh,
                        x1_diff, x2_diff, x3_diff, min_dist_y1, max_dist_y1, min_dist_y2, max_dist_y2, min_dist_y3, max_dist_y3))
        else:
            self.valid_lane_lines = True
            if print_diagnostics:
                print("Valid lane lines found. Derivatives: norm1 == {:.3f}, norm2 == {:.3f} (thresh == {}). Distance: x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f} (min_dist_y1 == {}, max_dist_y1 == {}, min_dist_y2 == {}, max_dist_y2 == {}, min_dist_y3 == {}, max_dist_y3 == {}".format(norm1, norm2, thresh,
                        x1_diff, x2_diff, x3_diff, min_dist_y1, max_dist_y1, min_dist_y2, max_dist_y2, min_dist_y3, max_dist_y3))
    
    def draw_lane(self, img):
        '''
        Highlight the lane the car is on and print curve radius and eccentricity onto the image.
        
        The method uses the polynomial graph points of the two lane lines to form the polygon that
        will be highlighted in the image.
        '''
        
        # Create an blank image to draw the lane on
        warped_lane = np.zeros((self.warped_size[1], self.warped_size[0])).astype(np.uint8)
        warped_lane = np.stack((warped_lane, warped_lane, warped_lane), axis=2)
            
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_avg_x, self.left_avg_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_avg_x, self.right_avg_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the warped lane onto the blank image
        cv2.fillPoly(warped_lane, np.int_([pts]), (0, 255, 0))

        # Unwarp the lane image using the inverse perspective matrix `Minv`
        unwarped_lane = cv2.warpPerspective(warped_lane, self.Minv, (img.shape[1], img.shape[0])) 
        
        # Draw a text field with the curve radius onto the original image
        cv2.putText(img, "Curve Radius: {} m".format(self.average_curve_radius), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, "Eccentricity: {:.2f} m".format(self.eccentricity), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        if self.print_frame_count:
            cv2.putText(img, "Frame: {}".format(self.counter-1), (20, 105), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

        # Combine the result with the original image
        return cv2.addWeighted(img, 1, unwarped_lane, 0.3, 0)
    
    def print_failure(self, img):
        # Print a failure message onto the image.
        # Do this if no valid lane lines are available.
        
        cv2.putText(img, "Lane Line Detection Failed", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        if self.print_frame_count:
            cv2.putText(img, "Frame: {}".format(self.counter-1), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        return img
    
    def window_mask(self, img, window_width, window_height, center, level, ignore_bottom):
        # Return a mask for the respective window measures.
        # This method is only used by `visualize_sliding_window_search()`.
        
        output = np.zeros_like(img)

        img_width = img.shape[1]
        img_height = img.shape[0] - ignore_bottom

        output[int(img_height-(level+1)*window_height):int(img_height-level*window_height), max(int(center-window_width/2), 0):min(int(center+window_width/2), img_width)] = 1

        return output
    
    def visualize_sliding_window_search(self, binary_img, left_fit_coeffs, right_fit_coeffs, window_width, window_height, ignore_bottom):
        '''
        Draw the search windows and the detected points of a sliding window search
        and the fitted polynomial graph onto the image.
        
        This method only serves for debugging: Visualizing the sliding window searcch
        process provides insight into what the algorithm does exactly.
        '''
        
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_img)
        r_points = np.zeros_like(binary_img)

        # Go through each level and determine the search window points
        for level in range(0, max(len(self.left_window_centroids), len(self.right_window_centroids))):
            if level < len(self.left_window_centroids):
                l_mask = self.window_mask(binary_img, window_width, window_height, self.left_window_centroids[level], level, ignore_bottom)
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if level < len(self.right_window_centroids):
                r_mask = self.window_mask(binary_img, window_width, window_height, self.right_window_centroids[level], level, ignore_bottom)
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the search windows
        template = np.array(r_points+l_points, np.uint8) # Add both left and right window pixels together
        zero_channel = np.zeros_like(template) # Create a zero color channel 
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # Make window pixels green
        color = np.array(cv2.merge((binary_img, binary_img, binary_img)), np.uint8) # Making the original road pixels 3 color channels
        output = cv2.addWeighted(color, 1, template, 0.5, 0.0) # Overlay the orignal road image with window results

        # Now highlight the lane pixels
        output[self.left_y, self.left_x] = [255, 0, 0]
        output[self.right_y, self.right_x] = [0, 0, 255]
        
        # Get the graph points of the new polynomial, the one we fitted after doing band search
        left_fit_y, left_fit_x, right_fit_y, right_fit_x = self.get_poly_points(left_fit_coeffs, right_fit_coeffs)
        
        # Now highlight the polynomial graph points
        output[left_fit_y, left_fit_x] = [255, 235, 0] # RGB color code for yellow
        output[right_fit_y, right_fit_x] = [255, 235, 0]
        
        return output
    
    def visualize_band_search(self, binary_img, left_fit_coeffs, right_fit_coeffs, bandwidth, partial):
        '''
        Draw the search band and the detected points of a band search
        and the fitted polynomial graph onto the image.
        
        This method only serves for debugging: Visualizing the band search
        process provides insight into what the algorithm does exactly.
        '''
        
        # Turn the binary image into a color image to draw on and create a blank image to show the search band
        output = np.array(cv2.merge((binary_img, binary_img, binary_img)), np.uint8)
        window_img = np.zeros_like(output)
        
        # Color in left and right lane pixels
        output[self.left_y, self.left_x] = [255, 0, 0]
        output[self.right_y, self.right_x] = [0, 0, 255]
        
        # Get the graph points of the old polynomial, the one we used to do band search
        left_band_y, left_band_x, right_band_y, right_band_x = self.get_poly_points(self.last_left_coeffs, self.last_right_coeffs, partial)
        
        # Generate a polygon to represent the search band
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_band_x-bandwidth, left_band_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_band_x+bandwidth, left_band_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_band_x-bandwidth, right_band_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_band_x+bandwidth, right_band_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the search band onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(output, 1, window_img, 0.3, 0)
        
        # Get the graph points of the new polynomial, the one we fitted after doing band search
        left_fit_y, left_fit_x, right_fit_y, right_fit_x = self.get_poly_points(left_fit_coeffs, right_fit_coeffs)
        
        # Now highlight the new polynomial graph points
        result[left_fit_y, left_fit_x] = [255, 235, 0] # RGB color code for yellow
        result[right_fit_y, right_fit_x] = [255, 235, 0]
        
        return result
    
    def find_lane_points(self,
                         img,
                         ksize_r=15,
                         C_r=8,
                         ksize_b=35,
                         C_b=5,
                         filter_mode='bilateral',
                         mask_noise=True,
                         noise_thresh=140,
                         ksize_noise=65,
                         C_noise=10,
                         window_width=30,
                         window_height=40,
                         search_range=20,
                         mu=0.1,
                         no_success_limit=50,
                         start_slice=0.25,
                         ignore_sides=360,
                         ignore_bottom=30,
                         bandwidth=30,
                         partial=0.5,
                         print_diagnostics=False):
        '''
        Perform a lane line pixel search on the input image.
        
        This method calls all other relevant methods to put the entire process
        from taking in a raw input image to determining the lane line pixels
        into one method.
        
        The input image is rectified of distortion, warped into bird's eye perspective,
        filtered, morphed and thresholded to isolate lane pixels, and then the appropriate
        lane pixel search method is applied.
        '''
        
        # 1. Transform the input image
        
        # Undistort the input image
        img = cv2.undistort(img, self.cam_matrix, self.dist_coeffs, None, self.cam_matrix)
        # Warp it
        warped_img = cv2.warpPerspective(img, self.M, self.warped_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Try to isolate lane points in a binary image
        binary_img = self.filter_lane_points(warped_img,
                                             filter_mode=filter_mode,
                                             ksize_r=ksize_r,
                                             C_r=C_r,
                                             ksize_b=ksize_b,
                                             C_b=C_b,
                                             mask_noise=mask_noise,
                                             ksize_noise=ksize_noise,
                                             C_noise=C_noise,
                                             noise_thresh=noise_thresh)
        
        # 2. Find points that belong to the lane lines
        
        # If we have not found any lane lines for several frames...
        if self.last_detection > self.n_reset:
            # Use sliding window blind search to find pixels that belong to lane lines
            self.sliding_window_search(binary_img,
                                       window_width=window_width,
                                       window_height=window_height,
                                       search_range=search_range,
                                       mu=mu,
                                       no_success_limit=no_success_limit,
                                       start_slice=start_slice,
                                       ignore_sides=ignore_sides,
                                       ignore_bottom=ignore_bottom,
                                       partial=partial,
                                       print_diagnostics=print_diagnostics)
            
            search_mode = 'sws'
                
        # If we have found lane lines in recent frames...
        else:
            # Use band search to find pixels within a narrow band around the previous lines
            self.band_search(binary_img, bandwidth=bandwidth, ignore_bottom=ignore_bottom, partial=partial, print_diagnostics=print_diagnostics)
            
            search_mode = 'bs'

        return binary_img, search_mode
    
    def process(self,
                img,
                ksize_r=15,
                C_r=8,
                ksize_b=35,
                C_b=5,
                filter_mode='bilateral',
                mask_noise=True,
                noise_thresh=140,
                ksize_noise=65,
                C_noise=10,
                window_width=30,
                window_height=40,
                search_range=20,
                mu=0.1,
                no_success_limit=50,
                start_slice=0.25,
                ignore_sides=360,
                ignore_bottom=30,
                bandwidth=30,
                partial=0.5,
                visualize_search=False,
                n_tries=2,
                print_diagnostics=True):
        '''
        Find the two lane lines of the lane which the car is on in an image and
        hightlight the lane in the image.
        
        This method is the only method that needs to be called, it contains the
        detection process and makes all necessary method calls.
        
        If `visualize_search` is `True`, it returns not only the processed image,
        but also an image with the visualization of the search method that was used.
        '''
        
        self.counter += 1
        self.detected_pixels = False
        self.valid_lane_lines = False
        
        ### 1. Try (possibly a few times with different parameters) to find valid lane lines
        
        # First try: This is our standard config and is mostly likely to work in general, but due to the
        # bilateral filter it will only succeed if lanes lines are visible on both sides of the lane 
        
        binary_img, search_mode = self.find_lane_points(img,
                                                        ksize_r=ksize_r,
                                                        C_r=C_r,
                                                        ksize_b=ksize_b,
                                                        C_b=C_b,
                                                        filter_mode=filter_mode,
                                                        mask_noise=mask_noise,
                                                        noise_thresh=noise_thresh,
                                                        ksize_noise=ksize_noise,
                                                        C_noise=C_noise,
                                                        window_width=window_width,
                                                        window_height=window_height,
                                                        search_range=search_range,
                                                        mu=mu,
                                                        no_success_limit=no_success_limit,
                                                        start_slice=start_slice,
                                                        ignore_sides=ignore_sides,
                                                        ignore_bottom=ignore_bottom,
                                                        bandwidth=bandwidth,
                                                        partial=partial,
                                                        print_diagnostics=print_diagnostics)
        
        if self.detected_pixels:
            # Fit a 2nd-order polynomial and get the coefficients and curve radius
            left_fit_coeffs, right_fit_coeffs = self.fit_poly()
            # Perform a validity check: Do the found curves make sense?
            self.check_validity(left_fit_coeffs, right_fit_coeffs, print_diagnostics)
            if print_diagnostics & self.valid_lane_lines: print("Success at first attempt!")
        
        if ((not self.detected_pixels) | (not self.valid_lane_lines)) & ((n_tries >= 2) | (n_tries == -1)):
            
            if print_diagnostics: print("No success at first attempt, now trying second.")
            
            # Second try: This config does not use the bilateral filter, nor does it use the tophat morphology.
            # It is generally more noisy than other configs, but it has a chance to succeed in situations where
            # one or both lane lines are not visible if there is still some brightness contrast between the lane
            # and its surroundings.

            # Parameters
            ksize_r=15
            C_r=8
            ksize_b=35
            C_b=5
            filter_mode='neighborhood'
            mask_noise=False
            noise_thresh=140
            ksize_noise=65
            C_noise=10
            window_width=30
            window_height=40
            search_range=20
            mu=0.1
            no_success_limit=50
            start_slice=0.25
            ignore_sides=360
            ignore_bottom=30
            bandwidth=30
            partial=0.5
            
            binary_img, search_mode = self.find_lane_points(img,
                                                            ksize_r=ksize_r,
                                                            C_r=C_r,
                                                            ksize_b=ksize_b,
                                                            C_b=C_b,
                                                            filter_mode=filter_mode,
                                                            mask_noise=mask_noise,
                                                            noise_thresh=noise_thresh,
                                                            ksize_noise=ksize_noise,
                                                            C_noise=C_noise,
                                                            window_width=window_width,
                                                            window_height=window_height,
                                                            search_range=search_range,
                                                            mu=mu,
                                                            no_success_limit=no_success_limit,
                                                            start_slice=start_slice,
                                                            ignore_sides=ignore_sides,
                                                            ignore_bottom=ignore_bottom,
                                                            bandwidth=bandwidth,
                                                            partial=partial,
                                                            print_diagnostics=print_diagnostics)
            
            if self.detected_pixels:
                # Fit a 2nd-order polynomial and get the coefficients and curve radius
                left_fit_coeffs, right_fit_coeffs = self.fit_poly()
                # Perform a validity check: Do the found curves make sense?
                self.check_validity(left_fit_coeffs, right_fit_coeffs, print_diagnostics)
                if print_diagnostics & self.valid_lane_lines: print("Success at second attempt!")
        
        if self.detected_pixels:
            # If the search is to be visualized, generate the appropriate visualization image
            if visualize_search:
                if search_mode == 'sws':
                    search_visualization = self.visualize_sliding_window_search(binary_img, left_fit_coeffs, right_fit_coeffs, window_width, window_height, ignore_bottom)
                else:
                    search_visualization = self.visualize_band_search(binary_img, left_fit_coeffs, right_fit_coeffs, bandwidth, partial)
        
        ### 2. If we didn't succeed in any of the above attempts, do some maintenance variable updates
        ### and see if we can just use the lane lines from a past frame, otherwise fail
        
        if not self.valid_lane_lines:
            if print_diagnostics: print("No success after all attempts.")
            # Append a failure entry to the coefficients lists to keep track 
            self.left_fit_coeffs.append(np.array([]))
            self.right_fit_coeffs.append(np.array([]))
            # Do the same for the curve radii
            self.average_curve_radii.append(-1)
            # Remove the oldest entries from the above lists
            # (but only do so once they get longer than `n_averages`)
            if len(self.left_fit_coeffs) > self.n_average:
                self.left_fit_coeffs.pop(0)
                self.right_fit_coeffs.pop(0)
            # Again, do the same for the curve radii
            if len(self.average_curve_radii) > self.n_average:
                self.average_curve_radii.pop(0)
            # Increment the failure counter by one
            self.last_detection += 1
            # Draw the last found lane lines (if there are any) on the image and return it
            if (self.left_avg_y.size != 0) & (self.last_detection <= self.n_fail):
                if visualize_search:
                    return self.draw_lane(img), search_visualization
                else:
                    return self.draw_lane(img)
            else:
                if visualize_search:
                    return self.print_failure(img), search_visualization
                else:
                    return self.print_failure(img)
        
        ### 3. If we did succeed and find valid lane lines, do some maintenance variable updates,
        ### print the detected lane lines onto the image and return it
        
        else:
            # Store fit coefficients
            self.left_fit_coeffs.append(left_fit_coeffs)
            self.right_fit_coeffs.append(right_fit_coeffs)
            self.last_left_coeffs = left_fit_coeffs
            self.last_right_coeffs = right_fit_coeffs
            # Remove the oldest entries from the above lists
            # (but only do so once they get longer than `n_averages`)
            if len(self.left_fit_coeffs) > self.n_average:
                self.left_fit_coeffs.pop(0)
                self.right_fit_coeffs.pop(0)
            # Reset the failure counter
            self.last_detection = 0
            # Record a success
            self.success += 1
            # Compute the average over the last `n_average` fits
            left_real_coeffs = [coeffs for coeffs in self.left_fit_coeffs if coeffs.size != 0] # Exclude failure entries
            right_real_coeffs = [coeffs for coeffs in self.right_fit_coeffs if coeffs.size != 0]
            self.left_avg_coeffs = np.average(left_real_coeffs, axis=0)
            self.right_avg_coeffs = np.average(right_real_coeffs, axis=0)
            # Get the graph points of the polynomial given its coefficients.
            self.left_avg_y, self.left_avg_x, self.right_avg_y, self.right_avg_x = self.get_poly_points(self.left_avg_coeffs, self.right_avg_coeffs, partial)
            # Compute curve radius and eccentricity
            self.get_curve_radius()
            self.get_eccentricity()
            # Draw the lane lines on the image and return it
            if visualize_search:
                return self.draw_lane(img), search_visualization
            else:
                return self.draw_lane(img)

### Run the program on a video

# 1. Load the camera matrix, distortion coefficients, warp matrices, image size info, and metric conversion factors.

cam_matrix, dist_coeffs = load_camera_calib('cam_calib.p')
M, Minv, image_width_height, warped_width_height, mppv, mpph = load_warp_params('warp_params.p')

# 2. Create a LaneTracker instance with the desired parameters

lt = LaneTracker(img_size = image_width_height,
                 warped_size = warped_width_height,
                 cam_matrix = cam_matrix,
                 dist_coeffs = dist_coeffs,
                 warp_matrices = (M, Minv),
                 mpp_conversion = (mppv, mpph),
                 n_fail = 8, # After eight failed detection frames we print the failure message onto subsequent images
                 n_reset = 4, # After four unsuccessful band searches we revert to sliding window search
                 n_average = 2, # It's enough to average over 2 frames
                 print_frame_count=False) # In case we'd like to print the frame number onto each image, but we don't

# 3. Load the video file and run the process

output_video = 'harder_challenge_video_lane_lines.mp4'
clip = VideoFileClip('harder_challenge_video.mp4')
clip1 = clip.fl_image(lt.process) # Note: This function expects color images
clip1.write_videofile(output_video, audio=False)

# Print on what fraction of the frames we believe to have found valid lane lines
success_ratio, success, total = lt.get_success_ratio()
print("Success ratio:", success_ratio)
print("Success absolute:", success)
