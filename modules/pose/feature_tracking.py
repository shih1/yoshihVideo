import cv2
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import OUTPUT_DIR, FEATURE_TRACKING

def analyze_feature_tracking(frames, metadata):
    """
    Track features using ORB and AKAZE detectors
    Good for anime due to sharp corners and high-contrast features
    """
    output_path = os.path.join(OUTPUT_DIR, 'feature_tracking_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Initialize detectors
    use_orb = FEATURE_TRACKING['use_orb']
    use_akaze = FEATURE_TRACKING['use_akaze']
    
    if use_orb:
        orb = cv2.ORB_create(
            nfeatures=FEATURE_TRACKING['max_features'],
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
    
    if use_akaze:
        akaze = cv2.AKAZE_create(
            threshold=FEATURE_TRACKING['akaze_threshold']
        )
    
    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Process first frame
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    # Detect features in first frame
    prev_kp_orb = None
    prev_desc_orb = None
    prev_kp_akaze = None
    prev_desc_akaze = None
    
    if use_orb:
        prev_kp_orb, prev_desc_orb = orb.detectAndCompute(prev_gray, None)
    if use_akaze:
        prev_kp_akaze, prev_desc_akaze = akaze.detectAndCompute(prev_gray, None)
    
    # Statistics tracking
    total_orb_matches = 0
    total_akaze_matches = 0
    orb_match_counts = []
    akaze_match_counts = []
    
    print(f"Initial ORB features: {len(prev_kp_orb) if prev_kp_orb else 0}")
    print(f"Initial AKAZE features: {len(prev_kp_akaze) if prev_kp_akaze else 0}")
    
    # Write first frame with features
    first_frame = frames[0].copy()
    if use_orb and prev_kp_orb:
        cv2.drawKeypoints(first_frame, prev_kp_orb, first_frame, 
                         color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if use_akaze and prev_kp_akaze:
        cv2.drawKeypoints(first_frame, prev_kp_akaze, first_frame,
                         color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Add legend
    cv2.putText(first_frame, "ORB: Green | AKAZE: Blue", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(first_frame)
    
    # Process remaining frames
    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        curr_kp_orb = None
        curr_desc_orb = None
        curr_kp_akaze = None
        curr_desc_akaze = None
        
        if use_orb:
            curr_kp_orb, curr_desc_orb = orb.detectAndCompute(gray, None)
        if use_akaze:
            curr_kp_akaze, curr_desc_akaze = akaze.detectAndCompute(gray, None)
        
        # Match ORB features
        orb_matches = []
        if use_orb and prev_desc_orb is not None and curr_desc_orb is not None:
            if len(prev_desc_orb) > 0 and len(curr_desc_orb) > 0:
                matches = bf.match(prev_desc_orb, curr_desc_orb)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep only good matches
                good_matches = [m for m in matches if m.distance < FEATURE_TRACKING['match_threshold']]
                orb_matches = good_matches[:FEATURE_TRACKING['max_display_matches']]
                
                total_orb_matches += len(good_matches)
                orb_match_counts.append(len(good_matches))
                
                # Draw matches
                for match in orb_matches:
                    pt1 = tuple(map(int, prev_kp_orb[match.queryIdx].pt))
                    pt2 = tuple(map(int, curr_kp_orb[match.trainIdx].pt))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(frame, pt2, 3, (0, 255, 0), -1)
        
        # Match AKAZE features
        akaze_matches = []
        if use_akaze and prev_desc_akaze is not None and curr_desc_akaze is not None:
            if len(prev_desc_akaze) > 0 and len(curr_desc_akaze) > 0:
                matches = bf.match(prev_desc_akaze, curr_desc_akaze)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep only good matches
                good_matches = [m for m in matches if m.distance < FEATURE_TRACKING['match_threshold']]
                akaze_matches = good_matches[:FEATURE_TRACKING['max_display_matches']]
                
                total_akaze_matches += len(good_matches)
                akaze_match_counts.append(len(good_matches))
                
                # Draw matches
                for match in akaze_matches:
                    pt1 = tuple(map(int, prev_kp_akaze[match.queryIdx].pt))
                    pt2 = tuple(map(int, curr_kp_akaze[match.trainIdx].pt))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
                    cv2.circle(frame, pt2, 3, (255, 0, 0), -1)
        
        # Add info overlay
        cv2.putText(frame, f"ORB Matches: {len(orb_matches)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"AKAZE Matches: {len(akaze_matches)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Frame: {i+1}/{len(frames)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
        # Update previous frame data
        prev_gray = gray
        prev_kp_orb = curr_kp_orb
        prev_desc_orb = curr_desc_orb
        prev_kp_akaze = curr_kp_akaze
        prev_desc_akaze = curr_desc_akaze
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("FEATURE TRACKING STATISTICS")
    print(f"{'='*60}")
    
    if use_orb and orb_match_counts:
        avg_orb = np.mean(orb_match_counts)
        max_orb = np.max(orb_match_counts)
        min_orb = np.min(orb_match_counts)
        print(f"\nORB Features:")
        print(f"  Total matches: {total_orb_matches}")
        print(f"  Avg matches/frame: {avg_orb:.1f}")
        print(f"  Max matches: {max_orb}")
        print(f"  Min matches: {min_orb}")
    
    if use_akaze and akaze_match_counts:
        avg_akaze = np.mean(akaze_match_counts)
        max_akaze = np.max(akaze_match_counts)
        min_akaze = np.min(akaze_match_counts)
        print(f"\nAKAZE Features:")
        print(f"  Total matches: {total_akaze_matches}")
        print(f"  Avg matches/frame: {avg_akaze:.1f}")
        print(f"  Max matches: {max_akaze}")
        print(f"  Min matches: {min_akaze}")
    
    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}")
    
    return output_path


# Allow standalone execution
if __name__ == "__main__":
    from preprocessing import preprocess_video, load_raw_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    # For anime, raw video might be better (no stabilization artifacts)
    print("Loading raw video (recommended for anime)...")
    frames, metadata = load_raw_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_feature_tracking(frames, metadata)