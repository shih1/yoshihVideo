import cv2
import numpy as np
from config import STABILIZATION, COLOR_ENHANCEMENT

def load_video(video_path, max_frames=None):
    """Load video and return frames with metadata"""
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_count = 0
    
    print(f"Loading video: {width}x{height} @ {fps}fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
        
        if frame_count % 100 == 0:
            print(f"Loaded {frame_count} frames...")
    
    cap.release()
    print(f"Total frames loaded: {len(frames)}")
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': len(frames)
    }
    
    return frames, metadata

def stabilize_video(frames):
    """Stabilize video using feature tracking and smoothing"""
    if not STABILIZATION['enabled']:
        print("Stabilization disabled, skipping...")
        return frames
    
    print("Stabilizing video...")
    
    transforms = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    # Calculate transforms between consecutive frames
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=STABILIZATION['max_corners'],
            qualityLevel=STABILIZATION['quality_level'],
            minDistance=STABILIZATION['min_distance'],
            blockSize=3
        )
        
        if prev_pts is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            if len(prev_pts) >= 4:
                transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                if transform is not None:
                    transforms.append(transform)
                else:
                    transforms.append(np.eye(2, 3, dtype=np.float32))
            else:
                transforms.append(np.eye(2, 3, dtype=np.float32))
        else:
            transforms.append(np.eye(2, 3, dtype=np.float32))
        
        prev_gray = gray
        
        if (i + 1) % 50 == 0:
            print(f"Analyzed {i + 1}/{len(frames)} frames")
    
    # Calculate cumulative trajectory and smooth it
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.copy(trajectory)
    
    radius = STABILIZATION['smoothing_radius']
    for i in range(len(smoothed_trajectory)):
        start = max(0, i - radius)
        end = min(len(smoothed_trajectory), i + radius + 1)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)
    
    smooth_transforms = smoothed_trajectory - trajectory
    smooth_transforms = transforms + smooth_transforms
    
    # Apply stabilization
    stabilized_frames = [frames[0]]
    height, width = frames[0].shape[:2]
    
    for i in range(len(smooth_transforms)):
        stabilized = cv2.warpAffine(
            frames[i + 1],
            smooth_transforms[i],
            (width, height),
            borderMode=cv2.BORDER_REPLICATE
        )
        stabilized_frames.append(stabilized)
        
        if (i + 1) % 50 == 0:
            print(f"Stabilized {i + 1}/{len(smooth_transforms)} frames")
    
    print("Stabilization complete!")
    return stabilized_frames

def enhance_colors(frame):
    """Apply color enhancement for cloudy/bland footage"""
    if not COLOR_ENHANCEMENT['enabled']:
        return frame
    
    # Convert to LAB and apply CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=COLOR_ENHANCEMENT['clahe_clip_limit'],
        tileGridSize=COLOR_ENHANCEMENT['clahe_tile_size']
    )
    l = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * COLOR_ENHANCEMENT['saturation_boost'], 0, 255)
    hsv = hsv.astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Warmth adjustment
    enhanced = enhanced.astype(np.float32)
    enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * COLOR_ENHANCEMENT['blue_reduction'], 0, 255)
    enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * COLOR_ENHANCEMENT['green_boost'], 0, 255)
    enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * COLOR_ENHANCEMENT['red_boost'], 0, 255)
    
    return enhanced.astype(np.uint8)

def enhance_frames(frames):
    """Apply color enhancement to all frames"""
    if not COLOR_ENHANCEMENT['enabled']:
        print("Color enhancement disabled, skipping...")
        return frames
    
    print("Enhancing colors...")
    enhanced_frames = []
    
    for i, frame in enumerate(frames):
        enhanced_frames.append(enhance_colors(frame))
        
        if (i + 1) % 50 == 0:
            print(f"Enhanced {i + 1}/{len(frames)} frames")
    
    print("Color enhancement complete!")
    return enhanced_frames

def preprocess_video(video_path, max_frames=None):
    """Complete preprocessing pipeline"""
    frames, metadata = load_video(video_path, max_frames)
    frames = stabilize_video(frames)
    frames = enhance_frames(frames)
    
    return frames, metadata