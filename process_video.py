#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:50:07 2017

@author: pierluigiferrari
"""

# Import third party libraries
import numpy as np
from moviepy.editor import VideoFileClip

# Import modules
from lane_tracker import bilateral_adaptive_threshold
from lane_tracker import LaneTracker
from utils import load_camera_calib
from utils import load_warp_params

### Run the lane tracker program on a video

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

output_clip_filename = 'harder_challenge_video_lane_lines.mp4' # Set the name and filepath of the output video here
input_clip_filename = VideoFileClip('harder_challenge_video.mp4') # Set the name and filepath of the input video here
processed_clip = input_clip_filename.fl_image(lt.process) # Note: This function expects color images
processed_clip.write_videofile(output_clip_filename, audio=False)

# Print on what fraction of the frames we believe to have found valid lane lines
success_ratio, success, total = lt.get_success_ratio()
print("Success ratio: ", success_ratio)
print("Success absolute: ", success)
