import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import FLOW_LUCAS_KANADE, FLOW_VISUALIZATION, OUTPUT_DIR

def analyze_lucas_kanade(frames, metadata):
    """Analyze water flow using Lucas-Kanade sparse optical flow"""
    print("\n=== Lucas-Kanade Optical Flow Analysis ===")
    
    output_path = os.path.join(OUTPUT_DIR, 'lucas_kanade_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'], 
                         (metadata['width'], metadata['height']))
    
    # LK parameters
    lk_params = dict(
        winSize=FLOW_LUCAS_KANADE['winSize'],
        maxLevel=FLOW_LUCAS_KANADE['maxLevel'],
        criteria=FLOW_LUCAS_KANADE['criteria']
    )
    
    # Feature parameters
    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7
    )
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    # Create color trails for visualization
    color = np.random.randint(0, 255, (300, 3))
    
    flow_magnitudes = []
    
    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_pts is not None and len(prev_pts) > 0:
            # Calculate optical flow
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )
            
            # Select good points
            if next_pts is not None:
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]
                
                # Draw tracks
                mask = np.zeros_like(frame)
                total_mag = 0
                
                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    
                    # Calculate magnitude
                    mag = np.sqrt((a - c)**2 + (b - d)**2)
                    total_mag += mag
                    
                    # Draw line and circle
                    if mag > FLOW_VISUALIZATION['min_magnitude']:
                        cv2.line(mask, (a, b), (c, d), color[j % len(color)].tolist(), 2)
                        cv2.circle(frame, (a, b), 5, color[j % len(color)].tolist(), -1)
                
                # Blend with frame
                result = cv2.add(frame, mask)
                
                # Calculate average flow
                avg_flow = total_mag / max(len(good_new), 1)
                flow_magnitudes.append(avg_flow)
                
                # Add statistics
                cv2.putText(result, f"Frame: {i}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(result, f"Tracked Points: {len(good_new)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(result, f"Avg Flow: {avg_flow:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(result)
                
                # Update for next iteration
                prev_gray = gray
                prev_pts = good_new.reshape(-1, 1, 2)
                
                # Refresh points periodically
                if len(prev_pts) < 100:
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            else:
                out.write(frame)
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        else:
            out.write(frame)
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
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
    analyze_lucas_kanade(frames, metadata)