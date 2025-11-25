import cv2
import numpy as np
import os
from config import LIGHT_REFLECTION, OUTPUT_DIR

def analyze_light_reflection(frames, metadata):
    """Analyze light reflections and sparkles on water surface"""
    print("\n=== Light Reflection Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'light_reflection_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    reflection_counts = []
    reflection_areas = []
    sparkle_intensities = []
    
    for i, frame in enumerate(frames):
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, LIGHT_REFLECTION['blur_kernel'], 0)
        
        # Threshold to find bright spots (reflections)
        _, bright_mask = cv2.threshold(blurred, LIGHT_REFLECTION['brightness_threshold'],
                                       255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization
        result = frame.copy()
        reflection_overlay = np.zeros_like(frame)
        
        # Analyze each reflection
        valid_reflections = 0
        total_area = 0
        total_intensity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= LIGHT_REFLECTION['min_area']:
                valid_reflections += 1
                total_area += area
                
                # Get bounding box and calculate intensity
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                intensity = np.mean(roi)
                total_intensity += intensity
                
                # Draw contour and info
                cv2.drawContours(reflection_overlay, [contour], -1, (0, 255, 255), -1)
                cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                
                # Add center point
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(result, (cx, cy), 3, (255, 0, 255), -1)
        
        # Blend reflection overlay
        result = cv2.addWeighted(result, 0.7, reflection_overlay, 0.3, 0)
        
        # Create heatmap of brightness
        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        heatmap_blend = cv2.addWeighted(result, 0.7, heatmap, 0.3, 0)
        
        # Calculate statistics
        avg_intensity = total_intensity / max(valid_reflections, 1)
        reflection_counts.append(valid_reflections)
        reflection_areas.append(total_area)
        sparkle_intensities.append(avg_intensity)
        
        # Add statistics overlay
        cv2.putText(heatmap_blend, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(heatmap_blend, f"Reflections: {valid_reflections}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(heatmap_blend, f"Total Area: {int(total_area)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(heatmap_blend, f"Avg Brightness: {int(avg_intensity)}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add legend
        cv2.putText(heatmap_blend, "Cyan = Reflections", (10, metadata['height'] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(heatmap_blend, "Heatmap = Brightness", (10, metadata['height'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(heatmap_blend)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print overall statistics
    print(f"\nLight Reflection Statistics:")
    print(f"  Average reflections per frame: {np.mean(reflection_counts):.2f}")
    print(f"  Max reflections in a frame: {int(np.max(reflection_counts))}")
    print(f"  Average reflection area: {np.mean(reflection_areas):.2f} pixels")
    print(f"  Average sparkle intensity: {np.mean(sparkle_intensities):.2f}")
    print(f"  Total frames with reflections: {sum(1 for c in reflection_counts if c > 0)}")
    
    # Detect sparkle patterns
    reflection_variance = np.std(reflection_counts)
    print(f"  Reflection variance: {reflection_variance:.2f}")
    if reflection_variance > 5:
        print("  → High variance: Dynamic sparkling detected")
    else:
        print("  → Low variance: Steady reflections")
    
    print(f"\nOutput saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_light_reflection(frames, metadata)