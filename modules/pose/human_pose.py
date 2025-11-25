import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import POSE_ESTIMATION, OUTPUT_DIR

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Install with: pip install mediapipe")

def analyze_human_pose(frames, metadata):
    """Analyze human pose and body movement"""
    print("\n=== Human Pose Estimation Analysis ===")
    
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: MediaPipe is required for pose estimation")
        print("Install with: pip install mediapipe")
        return None
    
    output_path = os.path.join(OUTPUT_DIR, 'human_pose_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        min_detection_confidence=POSE_ESTIMATION['min_detection_confidence'],
        min_tracking_confidence=POSE_ESTIMATION['min_tracking_confidence'],
        model_complexity=POSE_ESTIMATION['model_complexity'],
        enable_segmentation=POSE_ESTIMATION['enable_segmentation'],
        smooth_landmarks=POSE_ESTIMATION['smooth_landmarks']
    )
    
    people_detected = []
    landmark_velocities = []
    
    prev_landmarks = None
    
    for i, frame in enumerate(frames):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        # Draw on the frame
        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            people_detected.append(1)
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Draw segmentation mask if enabled
            if POSE_ESTIMATION['enable_segmentation'] and results.segmentation_mask is not None:
                # Create colored mask
                mask = results.segmentation_mask
                mask_3channel = np.stack([mask] * 3, axis=-1)
                
                # Color the person region
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = 255  # Green mask
                
                # Blend mask with frame
                mask_overlay = np.where(mask_3channel > 0.5, colored_mask, np.zeros_like(frame))
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.3, 0)
            
            # Calculate velocities if we have previous landmarks
            if prev_landmarks is not None:
                total_velocity = 0
                landmark_count = 0
                
                for j, (curr_lm, prev_lm) in enumerate(zip(
                    results.pose_landmarks.landmark,
                    prev_landmarks.landmark
                )):
                    # Calculate displacement
                    dx = (curr_lm.x - prev_lm.x) * metadata['width']
                    dy = (curr_lm.y - prev_lm.y) * metadata['height']
                    velocity = np.sqrt(dx**2 + dy**2)
                    
                    total_velocity += velocity
                    landmark_count += 1
                
                avg_velocity = total_velocity / max(landmark_count, 1)
                landmark_velocities.append(avg_velocity)
            
            prev_landmarks = results.pose_landmarks
            
            # Add person detected indicator
            cv2.putText(annotated_frame, "PERSON DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add landmark count
            num_landmarks = len(results.pose_landmarks.landmark)
            cv2.putText(annotated_frame, f"Landmarks: {num_landmarks}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add velocity if available
            if landmark_velocities:
                cv2.putText(annotated_frame, f"Movement: {landmark_velocities[-1]:.2f} px/frame",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            people_detected.append(0)
            cv2.putText(annotated_frame, "NO PERSON DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add frame number
        cv2.putText(annotated_frame, f"Frame: {i}", (10, metadata['height'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    pose.close()
    out.release()
    
    # Print statistics
    total_people = sum(people_detected)
    detection_rate = (total_people / len(frames)) * 100
    
    print(f"\nPose Detection Statistics:")
    print(f"  Frames with people detected: {total_people}/{len(frames)} ({detection_rate:.1f}%)")
    print(f"  Frames with no detection: {len(frames) - total_people}")
    
    if landmark_velocities:
        print(f"\nMovement Statistics:")
        print(f"  Average movement speed: {np.mean(landmark_velocities):.2f} px/frame")
        print(f"  Max movement speed: {np.max(landmark_velocities):.2f} px/frame")
        print(f"  Min movement speed: {np.min(landmark_velocities):.2f} px/frame")
        
        # Classify activity level
        avg_vel = np.mean(landmark_velocities)
        if avg_vel < 5:
            print("  → Activity: STATIONARY (standing/sitting still)")
        elif avg_vel < 15:
            print("  → Activity: SLOW MOVEMENT (walking slowly)")
        elif avg_vel < 30:
            print("  → Activity: MODERATE MOVEMENT (normal walking)")
        elif avg_vel < 50:
            print("  → Activity: FAST MOVEMENT (jogging/running)")
        else:
            print("  → Activity: VERY FAST MOVEMENT (sprinting/jumping)")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nVisualization includes:")
    print("  - Skeleton overlay (33 body landmarks)")
    print("  - Segmentation mask (person vs background)")
    print("  - Movement velocity tracking")
    
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_human_pose(frames, metadata)