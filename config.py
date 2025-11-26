# Configuration file for CV Analysis Toolkit

# Video settings
INPUT_VIDEO = 'name.mp4'
MAX_FRAMES = 300  # Maximum frames to process (set to None for entire video)
OUTPUT_DIR = 'output_name_3'

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

# Pose estimation settings
POSE_ESTIMATION = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1,  # 0=Lite, 1=Full, 2=Heavy
    'enable_segmentation': True,
    'smooth_landmarks': True
}

HAND_TRACKING = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

FACE_MESH = {
    'max_num_faces': 1,
    'refine_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Feature Tracking (ORB/AKAZE) Settings
FEATURE_TRACKING = {
    'use_orb': True,           # Enable ORB detector
    'use_akaze': True,         # Enable AKAZE detector
    'max_features': 500,       # Maximum features to detect
    'akaze_threshold': 0.001,  # AKAZE detection threshold (lower = more features)
    'match_threshold': 50,     # Maximum distance for good matches
    'max_display_matches': 100 # Maximum matches to draw (for clarity)
}

# YOLO Object Detection Settings
YOLO_DETECTION = {
    'weights_path': 'yolo/yolov4-tiny.weights',  # or yolov4.weights for better accuracy
    'config_path': 'yolo/yolov4-tiny.cfg',       # or yolov4.cfg
    'names_path': 'yolo/coco.names',
    'confidence_threshold': 0.2,    # Minimum confidence to show detection
    'nms_threshold': 0.4,           # Non-maximum suppression threshold
    'use_gpu': False                # Set to True if you have CUDA-enabled GPU
}
