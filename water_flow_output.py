import cv2
import numpy as np

# Load video
video_path = 'water_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

# Video stabilization setup
transforms = []
prev_frame_gray = None

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Store all frames for two-pass processing
print("Pass 1: Analyzing camera motion for stabilization...")
frames = [prev_frame]
frame_count = 1
max_frames = 300

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                        minDistance=30, blockSize=3)
    
    if prev_pts is not None:
        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Find transformation matrix
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
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Analyzed {frame_count} frames")

print(f"\nPass 2: Stabilizing and processing {len(frames)} frames...")

# Calculate cumulative transformations and smooth them
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = np.copy(trajectory)

# Apply moving average smoothing
smoothing_radius = 30
for i in range(len(smoothed_trajectory)):
    start = max(0, i - smoothing_radius)
    end = min(len(smoothed_trajectory), i + smoothing_radius + 1)
    smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)

# Calculate smooth transforms
smooth_transforms = smoothed_trajectory - trajectory
smooth_transforms = transforms + smooth_transforms

def enhance_colors(frame):
    """Apply color enhancement for cloudy/bland footage"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Boost saturation by 30%
    hsv = hsv.astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Slight warmth adjustment (reduce blue, boost red/green slightly)
    enhanced = enhanced.astype(np.float32)
    enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 0.95, 0, 255)  # Reduce blue
    enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.02, 0, 255)  # Slight green boost
    enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.05, 0, 255)  # Slight red boost
    
    return enhanced.astype(np.uint8)

# Optical flow parameters
flow_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('water_flow_output.mp4', fourcc, fps, (width, height))

# Process stabilized frames
prev_stabilized_gray = None

for i in range(len(frames)):
    frame = frames[i]
    
    # Apply stabilization transform
    if i < len(smooth_transforms):
        transform = smooth_transforms[i]
        stabilized = cv2.warpAffine(frame, transform, (width, height),
                                    borderMode=cv2.BORDER_REPLICATE)
    else:
        stabilized = frame
    
    # Apply color enhancement
    enhanced = enhance_colors(stabilized)
    
    # Calculate optical flow on stabilized footage
    if i > 0 and prev_stabilized_gray is not None:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_stabilized_gray, gray, None, **flow_params)
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV visualization
        hsv = np.zeros_like(enhanced)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = cv2.addWeighted(enhanced, 0.6, flow_vis, 0.4, 0)
        
        # Draw flow vectors
        step = 20
        for y in range(0, height, step):
            for x in range(0, width, step):
                fx, fy = flow[y, x]
                if mag[y, x] > 1:
                    cv2.arrowedLine(result, 
                                  (x, y), 
                                  (int(x + fx*2), int(y + fy*2)),
                                  (0, 255, 0), 1, tipLength=0.3)
        
        # Add frame info
        cv2.putText(result, f"Frame: {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        avg_flow = np.mean(mag)
        cv2.putText(result, f"Avg Flow: {avg_flow:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(result)
        prev_stabilized_gray = gray
    else:
        # First frame - just write enhanced version
        out.write(enhanced)
        prev_stabilized_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    if (i + 1) % 30 == 0:
        print(f"Processed {i + 1}/{len(frames)} frames")

# Cleanup
cap.release()
out.release()
print(f"\nDone! Output saved to 'water_flow_output.mp4'")
print("\nEnhancements applied:")
print("- Video stabilization (camera shake removal)")
print("- CLAHE contrast enhancement")
print("- Saturation boost (+30%)")
print("- Warmth adjustment (reduced blue cast)")
print("- Flow visualization on stabilized footage")