import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import HAND_TRACKING, OUTPUT_DIR

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

def analyze_hand_tracking(frames, metadata):
    """Analyze hand landmarks and gestures"""
    print("\n=== Hand Tracking Analysis ===")
    
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: MediaPipe is required for hand tracking")
        print("Install with: pip install mediapipe")
        return None
    
    output_path = os.path.join(OUTPUT_DIR, 'hand_tracking_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        max_num_hands=HAND_TRACKING['max_num_hands'],
        min_detection_confidence=HAND_TRACKING['min_detection_confidence'],
        min_tracking_confidence=HAND_TRACKING['min_tracking_confidence']
    )
    
    hands_detected_per_frame = []
    hand_sizes = []
    
    for i, frame in enumerate(frames):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        # Draw on the frame
        annotated_frame = frame.copy()
        
        num_hands = 0
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            hands_detected_per_frame.append(num_hands)
            
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            )):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand label (Left/Right)
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                # Calculate hand bounding box and size
                landmarks = hand_landmarks.landmark
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert to pixel coordinates
                x_min_px = int(x_min * metadata['width'])
                x_max_px = int(x_max * metadata['width'])
                y_min_px = int(y_min * metadata['height'])
                y_max_px = int(y_max * metadata['height'])
                
                # Calculate hand size
                hand_width = x_max_px - x_min_px
                hand_height = y_max_px - y_min_px
                hand_size = hand_width * hand_height
                hand_sizes.append(hand_size)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x_min_px, y_min_px), (x_max_px, y_max_px),
                            (0, 255, 0), 2)
                
                # Add hand label
                label_y = y_min_px - 10 if y_min_px > 30 else y_max_px + 30
                cv2.putText(annotated_frame, f"{hand_label} ({hand_score:.2f})",
                           (x_min_px, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detect simple gestures (open vs closed)
                # Calculate distance between thumb tip and pinky tip
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                
                distance = np.sqrt(
                    (thumb_tip.x - pinky_tip.x)**2 + 
                    (thumb_tip.y - pinky_tip.y)**2
                )
                
                gesture = "OPEN" if distance > 0.3 else "CLOSED"
                cv2.putText(annotated_frame, gesture,
                           (x_min_px, y_max_px + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            hands_detected_per_frame.append(0)
        
        # Add statistics overlay
        cv2.putText(annotated_frame, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Hands Detected: {num_hands}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)
        
        out.write(annotated_frame)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    hands.close()
    out.release()
    
    # Print statistics
    frames_with_hands = sum(1 for count in hands_detected_per_frame if count > 0)
    detection_rate = (frames_with_hands / len(frames)) * 100
    
    print(f"\nHand Detection Statistics:")
    print(f"  Frames with hands: {frames_with_hands}/{len(frames)} ({detection_rate:.1f}%)")
    print(f"  Frames with no hands: {len(frames) - frames_with_hands}")
    
    if hands_detected_per_frame:
        total_hands = sum(hands_detected_per_frame)
        avg_hands = np.mean([h for h in hands_detected_per_frame if h > 0]) if frames_with_hands > 0 else 0
        print(f"  Total hand detections: {total_hands}")
        print(f"  Average hands when detected: {avg_hands:.2f}")
    
    if hand_sizes:
        print(f"\nHand Size Statistics:")
        print(f"  Average hand size: {np.mean(hand_sizes):.0f} sq pixels")
        print(f"  Max hand size: {np.max(hand_sizes):.0f} sq pixels")
        print(f"  Min hand size: {np.min(hand_sizes):.0f} sq pixels")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nVisualization includes:")
    print("  - Hand skeleton (21 landmarks per hand)")
    print("  - Left/Right hand classification")
    print("  - Bounding boxes")
    print("  - Simple gesture detection (open/closed)")
    
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_hand_tracking(frames, metadata)