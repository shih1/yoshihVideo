import cv2
import numpy as np
import os
from config import TEXTURE_ANALYSIS, OUTPUT_DIR

def create_gabor_kernels():
    """Create Gabor filter bank for texture analysis"""
    kernels = []
    for freq in TEXTURE_ANALYSIS['gabor_frequencies']:
        for theta in TEXTURE_ANALYSIS['gabor_orientations']:
            theta_rad = theta * np.pi / 180
            kernel = cv2.getGaborKernel((21, 21), 5, theta_rad, 10/freq, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
    return kernels

def analyze_texture(frames, metadata):
    """Analyze water surface texture and turbulence patterns"""
    print("\n=== Surface Texture Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'texture_analysis_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Create Gabor filter bank
    print("Creating Gabor filter bank...")
    gabor_kernels = create_gabor_kernels()
    
    texture_complexities = []
    turbulence_scores = []
    energy_values = []
    
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Local Binary Pattern (LBP) for texture
        # Simplified LBP-like approach
        texture_map = np.zeros_like(gray, dtype=np.float32)
        
        # Calculate texture using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Method 2: Gabor filter responses for multi-scale texture
        gabor_responses = []
        for kernel in gabor_kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_responses.append(filtered)
        
        # Combine Gabor responses
        gabor_energy = np.zeros_like(gray, dtype=np.float32)
        for response in gabor_responses:
            gabor_energy += np.abs(response)
        gabor_energy = gabor_energy / len(gabor_responses)
        
        # Method 3: Local variance for turbulence detection
        block_size = TEXTURE_ANALYSIS['block_size']
        variance_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(0, gray.shape[0] - block_size, block_size):
            for x in range(0, gray.shape[1] - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                variance = np.var(block)
                variance_map[y:y+block_size, x:x+block_size] = variance
        
        # Normalize maps
        gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gabor_norm = cv2.normalize(gabor_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        variance_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create visualizations
        gradient_color = cv2.applyColorMap(gradient_norm, cv2.COLORMAP_HOT)
        gabor_color = cv2.applyColorMap(gabor_norm, cv2.COLORMAP_VIRIDIS)
        variance_color = cv2.applyColorMap(variance_norm, cv2.COLORMAP_PLASMA)
        
        # Create composite visualization
        top_row = np.hstack([frame, gradient_color])
        bottom_row = np.hstack([gabor_color, variance_color])
        
        # Resize to fit
        h, w = frame.shape[:2]
        top_row = cv2.resize(top_row, (w, h // 2))
        bottom_row = cv2.resize(bottom_row, (w, h // 2))
        result = np.vstack([top_row, bottom_row])
        
        # Calculate metrics
        texture_complexity = np.mean(gradient_magnitude)
        turbulence_score = np.mean(variance_map)
        gabor_response_energy = np.mean(gabor_energy)
        
        texture_complexities.append(texture_complexity)
        turbulence_scores.append(turbulence_score)
        energy_values.append(gabor_response_energy)
        
        # Add labels
        label_y = 30
        cv2.putText(result, f"Frame {i}", (10, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Original", (w//4, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Gradient (Edges)", (w + w//4, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        label_y2 = h//2 + 30
        cv2.putText(result, f"Gabor (Texture)", (w//4, label_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Variance (Turbulence)", (w + w//4, label_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add metrics
        metrics_y = h - 100
        cv2.putText(result, f"Complexity: {texture_complexity:.2f}", (10, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Turbulence: {turbulence_score:.2f}", (10, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Energy: {gabor_response_energy:.2f}", (10, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(result)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    print(f"\nTexture Analysis Statistics:")
    print(f"  Average texture complexity: {np.mean(texture_complexities):.2f}")
    print(f"  Complexity std deviation: {np.std(texture_complexities):.2f}")
    print(f"  Average turbulence score: {np.mean(turbulence_scores):.2f}")
    print(f"  Turbulence std deviation: {np.std(turbulence_scores):.2f}")
    print(f"  Average Gabor energy: {np.mean(energy_values):.2f}")
    
    # Classify water state
    avg_turbulence = np.mean(turbulence_scores)
    if avg_turbulence < 100:
        print("  → Water state: CALM (smooth flow)")
    elif avg_turbulence < 300:
        print("  → Water state: MODERATE (gentle ripples)")
    elif avg_turbulence < 600:
        print("  → Water state: TURBULENT (active ripples/waves)")
    else:
        print("  → Water state: HIGHLY TURBULENT (chaotic flow)")
    
    print(f"\nOutput saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_texture(frames, metadata)