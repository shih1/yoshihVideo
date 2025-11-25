import cv2
import numpy as np
import os
from config import OUTPUT_DIR

def analyze_velocity_magnitude(frames, metadata):
    """Analyze and visualize water speed (velocity magnitude)"""
    print("\n=== Velocity Magnitude Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'velocity_magnitude_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Create DIS optical flow instance for accurate flow
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    velocity_stats = []
    max_velocities = []
    
    # Store all velocity magnitudes to find global max for consistent scaling
    all_magnitudes = []
    
    # First pass: collect all magnitudes to find global max
    print("Pass 1: Analyzing velocity ranges...")
    temp_gray = prev_gray
    for i in range(1, min(len(frames), 100)):  # Sample first 100 frames
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(temp_gray, gray, None)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        all_magnitudes.append(np.max(mag))
        temp_gray = gray
        if i % 20 == 0:
            print(f"Sampled {i} frames...")
    
    global_max = np.percentile(all_magnitudes, 99)  # Use 99th percentile to avoid outliers
    print(f"Global max velocity (99th percentile): {global_max:.2f}")
    
    # Second pass: generate visualization with consistent scaling
    print("\nPass 2: Generating visualizations...")
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = dis.calc(prev_gray, gray, None)
        
        # Calculate velocity magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize magnitude to global max for consistent coloring
        mag_normalized = np.clip(mag / global_max * 255, 0, 255).astype(np.uint8)
        
        # Create multiple visualizations
        
        # 1. Hot colormap (black -> red -> yellow -> white)
        velocity_hot = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_HOT)
        
        # 2. Jet colormap (blue -> cyan -> green -> yellow -> red)
        velocity_jet = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)
        
        # 3. Inferno colormap (dark purple -> red -> orange -> yellow)
        velocity_inferno = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_INFERNO)
        
        # Create blended visualizations
        blend_hot = cv2.addWeighted(frame, 0.4, velocity_hot, 0.6, 0)
        blend_jet = cv2.addWeighted(frame, 0.4, velocity_jet, 0.6, 0)
        
        # Create side-by-side comparison
        top_row = np.hstack([blend_hot, blend_jet])
        bottom_row = np.hstack([velocity_hot, velocity_inferno])
        
        # Resize to fit output
        h, w = frame.shape[:2]
        top_row = cv2.resize(top_row, (w, h // 2))
        bottom_row = cv2.resize(bottom_row, (w, h // 2))
        result = np.vstack([top_row, bottom_row])
        
        # Calculate statistics
        avg_velocity = np.mean(mag)
        max_velocity = np.max(mag)
        velocity_stats.append(avg_velocity)
        max_velocities.append(max_velocity)
        
        # Find fastest region
        max_loc = np.unravel_index(np.argmax(mag), mag.shape)
        
        # Add labels and stats
        label_y = 30
        cv2.putText(result, f"Frame {i}", (10, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, "Hot + Original", (w//4, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, "Jet + Original", (w + w//4, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        label_y2 = h // 2 + 30
        cv2.putText(result, "Hot (Pure)", (w//4, label_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, "Inferno (Pure)", (w + w//4, label_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add velocity statistics
        stats_y = h - 120
        cv2.putText(result, f"Avg Velocity: {avg_velocity:.2f} px/frame", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Max Velocity: {max_velocity:.2f} px/frame", (10, stats_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Max Location: ({max_loc[1]}, {max_loc[0]})", (10, stats_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add color legend
        legend_y = h - 30
        cv2.putText(result, "Color: Dark=Slow -> Bright=Fast", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        out.write(result)
        prev_gray = gray
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    print(f"\nVelocity Statistics:")
    print(f"  Average velocity across all frames: {np.mean(velocity_stats):.2f} px/frame")
    print(f"  Maximum velocity detected: {np.max(max_velocities):.2f} px/frame")
    print(f"  Minimum velocity: {np.min(velocity_stats):.2f} px/frame")
    print(f"  Velocity std deviation: {np.std(velocity_stats):.2f}")
    
    # Classify water speed
    avg_vel = np.mean(velocity_stats)
    if avg_vel < 2:
        print("  → Water classification: VERY SLOW (nearly still)")
    elif avg_vel < 5:
        print("  → Water classification: SLOW (gentle flow)")
    elif avg_vel < 10:
        print("  → Water classification: MODERATE (steady flow)")
    elif avg_vel < 20:
        print("  → Water classification: FAST (rapids/waterfall)")
    else:
        print("  → Water classification: VERY FAST (strong waterfall/rapids)")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nVisualization Guide:")
    print("  - Waterfall regions should appear BRIGHT WHITE/YELLOW")
    print("  - Still pools should appear DARK BLUE/BLACK")
    print("  - Top row: Blended with original (shows context)")
    print("  - Bottom row: Pure velocity heatmaps (shows intensity)")
    
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_velocity_magnitude(frames, metadata)