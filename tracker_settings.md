### 1. Demo Video 1

#### process():

- ksize_r = 15
- C_r = 8
- ksize_b = 35
- C_b = 5
- filter_mode = 'bilateral'
- mask_noise = True
- noise_thresh = 140
- ksize_noise = 65
- C_noise = 10
- window_width = 30
- window_height = 40
- search_range = 20
- mu = 0.1
- no_success_limit = 50
- start_slice = 0.25
- ignore_sides = 360
- ignore_bottom = 30
- bandwidth = 30
- partial = 1.0
- n_tries = 2

#### check_validity():

- min_dist_y1 = 150
- max_dist_y1 = 245
- min_dist_y2 = 150
- max_dist_y2 = 255
- min_dist_y3 = 150
- max_dist_y3 = 255
- thresh = 0.25

### 2. Demo Video 2

#### process():

The contrast between the white lane markers and the pavement is very low in this scenario. In order to compensate for that, the settings for the RGB r-channel are adjusted slightly: The minimum contrast `C_r` was slightly reduced from 8 to 5 and the filter size was slightly increased from 15 to 20 pixels, allowing a little more tolerance for wider lane lines. The LAB b-channel, on the other hand, is not affected by the low contrast, it recognizes the yellow lane lines well. The `mask_noise` feature is turned off here. It was designed specifically to reduce the effect of greenery in immediate proximity to the road, of which there is none here. Also, `n_tries` was set to 1 in this case, the reason being that the whole point of the second processing attempt is to switch the filter mode from 'bilateral' to 'neighborhood', but the 'neighborhood' mode causes problems when there are large pavement patches of different color, as is the case on this highway stretch.

- **ksize_r = 20**
- **C_r = 5**
- ksize_b = 35
- C_b = 5
- filter_mode = 'bilateral'
- **mask_noise = False**
- noise_thresh = 140
- ksize_noise = 65
- C_noise = 10
- window_width = 30
- window_height = 40
- search_range = 20
- mu = 0.1
- no_success_limit = 50
- start_slice = 0.25
- ignore_sides = 360
- ignore_bottom = 30
- bandwidth = 30
- partial = 1.0
- **n_tries = 1**

#### check_validty():

The perspective transform that was measured for the project video scenario is not accurate for this case. The reason might be that the dashboard camera is mounted in a slightly different angle in this video compared to the project video. This shouldn't usually happen, i.e. the camera should always be mounted in a fixed way to record the exact same perspective, but since this isn't the case here, we have to deal with this inaccuracy. The proper way to deal with this would be to compute a new transform for this video, but instead we'll slightly change the criteria for what defines valid lane lines to accommodate the perspective distortion.

- min_dist_y1 = 150
- **max_dist_y1 = 230**
- **min_dist_y2 = 110**
- **max_dist_y2 = 230**
- **min_dist_y3 = 80**
- **max_dist_y3 = 200**
- thresh = 0.25

### 3. Demo Video 3

#### process():

In order to deal with sharper and more frequent turns, the horizon of the tracker needs to be a bit shorter than in the highway case, therefore `partial` is reduced to half of the warped image's height.

- ksize_r = 15
- C_r = 8
- ksize_b = 35
- C_b = 5
- filter_mode = 'bilateral'
- mask_noise = True
- noise_thresh = 140
- ksize_noise = 65
- C_noise = 10
- window_height = 40
- search_range = 20
- mu = 0.1
- no_success_limit = 50
- start_slice = 0.25
- ignore_sides = 360
- ignore_bottom = 30
- bandwidth = 30
- **partial = 0.5**
- n_tries = 2

#### check_validity():

The bird's eye perspective transformation does not compensate for the varying slope of the road, so the warped image will be distorted whenever the slope of the road ahead changes. In order to compensate for this at least to some degree, the tolerance for accepting a lane line detection as valid must be a bit higher than in the constant-slope case. It is a very suboptimal way to deal with the problem, but it's feasible.

- min_dist_y1 = 150
- max_dist_y1 = 245
- **min_dist_y2 = 140**
- **max_dist_y2 = 265**
- **min_dist_y3 = 125**
- **max_dist_y3 = 290**
- **thresh = 0.46**
