import cv2
import numpy as np
import os
from config import FLOW_DIS, FLOW_VISUALIZATION, OUTPUT_DIR

def analyze_dis(frames, metadata):
    """Analyze water flow using DIS (Dense Inverse Search) optical flow"""
    print("\n=== DIS Optical Flow Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'dis_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Create DIS optical flow instance
    dis = cv2.DISOpticalFlow_create(FLOW_DIS['preset'])
    dis.setUseSpatialPropagation(FLOW_DIS['use_spatial_propagation'])
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    flow_magnitudes = []
    
    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate DIS optical flow
        flow = dis.calc(prev_gray, gray, None)
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV visualization
        hsv = np.zeros_like(frame)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = speed
        
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, FLOW_VISUALIZATION['blend_alpha'],
                                flow_vis, 1 - FLOW_VISUALIZATION['blend_alpha'], 0)
        
        # Draw flow vectors (sample grid)
        step = FLOW_VISUALIZATION['arrow_step']
        for y in range(0, metadata['height'], step):
            for x in range(0, metadata['width'], step):
                fx, fy = flow[y, x]
                if mag[y, x] > FLOW_VISUALIZATION['min_magnitude']:
                    cv2.arrowedLine(result,
                                  (x, y),
                                  (int(x + fx * FLOW_VISUALIZATION['arrow_scale']),
                                   int(y + fy * FLOW_VISUALIZATION['arrow_scale'])),
                                  FLOW_VISUALIZATION['arrow_color'], 1, tipLength=0.3)
        
        # Calculate statistics
        avg_flow = np.mean(mag)
        max_flow = np.max(mag)
        flow_magnitudes.append(avg_flow)
        
        # Add text overlay
        cv2.putText(result, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Avg Flow: {avg_flow:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Max Flow: {max_flow:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(result)
        prev_gray = gray
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    if flow_magnitudes:
        print(f"\nFlow Statistics:")
        print(f"  Average flow magnitude: {np.mean(flow_magnitudes):.2f}")
        print(f"  Max flow magnitude: {np.max(flow_magnitudes):.2f}")
        print(f"  Min flow magnitude: {np.min(flow_magnitudes):.2f}")
        print(f"  Std deviation: {np.std(flow_magnitudes):.2f}")
    
    print(f"\nOutput saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_dis(frames, metadata)