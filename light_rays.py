import cv2
import numpy as np
import os
from config import OUTPUT_DIR

def analyze_light_rays(frames, metadata):
    """Visualize light ray radiation from bright spots (sun reflections)"""
    print("\n=== Light Ray Radiation Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'light_rays_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Parameters for light detection and ray projection
    brightness_threshold = 200
    min_area = 30
    ray_length = 150
    num_rays = 16  # Number of rays per light source
    ray_fade_steps = 30  # Steps for ray fade-out
    
    total_light_sources = []
    
    for i, frame in enumerate(frames):
        result = frame.copy()
        
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find bright spots
        _, bright_mask = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create ray overlay
        ray_overlay = np.zeros_like(frame, dtype=np.uint8)
        glow_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        light_sources_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                light_sources_count += 1
                
                # Get center of light source
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Get brightness intensity at center
                    intensity = gray[cy, cx]
                    intensity_factor = intensity / 255.0
                    
                    # Calculate ray color based on intensity (yellow to white)
                    ray_color = (
                        int(100 + 155 * intensity_factor),  # Blue
                        int(200 + 55 * intensity_factor),   # Green
                        int(255)                             # Red
                    )
                    
                    # Draw multiple rays emanating from light source
                    for ray_idx in range(num_rays):
                        angle = (2 * np.pi * ray_idx) / num_rays
                        
                        # Draw ray with fade-out effect
                        for step in range(ray_fade_steps):
                            t = step / ray_fade_steps
                            
                            # Calculate position along ray
                            ray_len = ray_length * (1 - t * 0.3)  # Rays taper slightly
                            end_x = int(cx + ray_len * np.cos(angle))
                            end_y = int(cy + ray_len * np.sin(angle))
                            
                            # Fade alpha
                            alpha = (1 - t) * intensity_factor
                            
                            # Add some randomness for organic look
                            wiggle_x = int(np.random.normal(0, 2 * t))
                            wiggle_y = int(np.random.normal(0, 2 * t))
                            end_x += wiggle_x
                            end_y += wiggle_y
                            
                            # Draw ray segment
                            if step > 0:
                                prev_t = (step - 1) / ray_fade_steps
                                prev_len = ray_length * (1 - prev_t * 0.3)
                                prev_x = int(cx + prev_len * np.cos(angle) + wiggle_x)
                                prev_y = int(cy + prev_len * np.sin(angle) + wiggle_y)
                                
                                # Ensure points are within frame
                                if (0 <= end_x < metadata['width'] and 
                                    0 <= end_y < metadata['height'] and
                                    0 <= prev_x < metadata['width'] and 
                                    0 <= prev_y < metadata['height']):
                                    
                                    thickness = max(1, int(3 * (1 - t)))
                                    cv2.line(ray_overlay, (prev_x, prev_y), (end_x, end_y),
                                           ray_color, thickness, cv2.LINE_AA)
                    
                    # Draw central glow
                    glow_radius = int(15 * intensity_factor)
                    for r in range(glow_radius, 0, -2):
                        alpha = (r / glow_radius) * 0.6
                        color = tuple(int(c * alpha) for c in ray_color)
                        cv2.circle(glow_overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
                    
                    # Draw bright core
                    cv2.circle(result, (cx, cy), 4, (255, 255, 255), -1)
                    cv2.circle(result, (cx, cy), 6, (200, 255, 255), 1)
        
        # Blend overlays
        # Apply Gaussian blur to rays for soft glow effect
        ray_overlay = cv2.GaussianBlur(ray_overlay, (15, 15), 0)
        glow_overlay = cv2.GaussianBlur(glow_overlay, (21, 21), 0)
        
        # Combine layers
        result = cv2.addWeighted(result, 1.0, glow_overlay, 0.7, 0)
        result = cv2.addWeighted(result, 1.0, ray_overlay, 0.5, 0)
        
        total_light_sources.append(light_sources_count)
        
        # Add frame info and statistics
        cv2.putText(result, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Light Sources: {light_sources_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Rays per Source: {num_rays}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add artistic note
        cv2.putText(result, "Simulated UV/Sun Ray Projection", (10, metadata['height'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        out.write(result)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    print(f"\nLight Ray Statistics:")
    print(f"  Average light sources per frame: {np.mean(total_light_sources):.2f}")
    print(f"  Max light sources in a frame: {int(np.max(total_light_sources))}")
    print(f"  Total frames with light sources: {sum(1 for c in total_light_sources if c > 0)}")
    print(f"  Frames with no light: {sum(1 for c in total_light_sources if c == 0)}")
    
    # Calculate light dynamics
    light_variance = np.std(total_light_sources)
    print(f"  Light source variance: {light_variance:.2f}")
    if light_variance > 3:
        print("  → Lighting: DYNAMIC (changing sun positions/reflections)")
    else:
        print("  → Lighting: STEADY (consistent light sources)")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nVisualization Guide:")
    print("  - Each bright spot projects radial rays")
    print("  - Ray intensity = brightness of source")
    print("  - Rays simulate light scattering/radiation")
    print("  - Center glow = immediate light field")
    print("  - Best for: Sparkly water, sun glints, light play")
    
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_light_rays(frames, metadata)