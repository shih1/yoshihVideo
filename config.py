# Configuration file for water analysis framework

# Video settings
INPUT_VIDEO = 'water_video.mp4'
MAX_FRAMES = 300  # Maximum frames to process (set to None for entire video)
OUTPUT_DIR = 'output'

# Preprocessing settings
STABILIZATION = {
    'enabled': True,
    'smoothing_radius': 30,
    'max_corners': 200,
    'quality_level': 0.01,
    'min_distance': 30
}

COLOR_ENHANCEMENT = {
    'enabled': True,
    'clahe_clip_limit': 3.0,
    'clahe_tile_size': (8, 8),
    'saturation_boost': 1.3,
    'blue_reduction': 0.95,
    'green_boost': 1.02,
    'red_boost': 1.05
}

# Optical flow settings
FLOW_FARNEBACK = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2
}

FLOW_LUCAS_KANADE = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (3, 10, 0.03)  # cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
}

FLOW_DIS = {
    'preset': 2,  # DISOPTICAL_FLOW_PRESET_MEDIUM
    'use_spatial_propagation': True
}

# Visualization settings
FLOW_VISUALIZATION = {
    'arrow_step': 20,
    'arrow_scale': 2,
    'min_magnitude': 1.0,
    'blend_alpha': 0.6,
    'arrow_color': (0, 255, 0)
}

# Light reflection settings
LIGHT_REFLECTION = {
    'brightness_threshold': 200,
    'min_area': 50,
    'blur_kernel': (5, 5)
}

# Texture analysis settings
TEXTURE_ANALYSIS = {
    'block_size': 32,
    'gabor_frequencies': [0.1, 0.2, 0.3],
    'gabor_orientations': [0, 45, 90, 135]
}