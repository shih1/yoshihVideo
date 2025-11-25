import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import FACE_MESH, OUTPUT_DIR

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

def analyze_face_mesh(frames, metadata):
    """Analyze facial landmarks and expressions"""
    print("\n=== Face Mesh Analysis ===")
    
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: MediaPipe is required for face mesh")
        print("Install with: pip install mediapipe")
        return None
    
    output_path = os.path.join(OUTPUT_DIR, 'face_mesh_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=FACE_MESH['max_num_faces'],
        refine_landmarks=FACE_MESH['refine_landmarks'],
        min_detection_confidence=FACE_MESH['min_detection_confidence'],
        min_tracking_confidence=FACE_MESH['min_tracking_confidence']
    )
    
    faces_detected = []
    face_sizes = []
    head_movements = []
    
    prev_nose_tip = None
    
    for i, frame in enumerate(frames):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = face_mesh.process(rgb_frame)
        
        # Draw on the frame
        annotated_frame = frame.copy()
        
        num_faces = 0
        
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            faces_detected.append(num_faces)
            
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Draw face mesh tesselation
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw face contours
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw irises if refined landmarks enabled
                if FACE_MESH['refine_landmarks']:
                    mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                
                # Calculate face bounding box
                landmarks = face_landmarks.landmark
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert to pixel coordinates
                x_min_px = int(x_min * metadata['width'])
                x_max_px = int(x_max * metadata['width'])
                y_min_px = int(y_min * metadata['height'])
                y_max_px = int(y_max * metadata['height'])
                
                # Calculate face size
                face_width = x_max_px - x_min_px
                face_height = y_max_px - y_min_px
                face_size = face_width * face_height
                face_sizes.append(face_size)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x_min_px, y_min_px), (x_max_px, y_max_px),
                            (255, 0, 0), 2)
                
                # Track head movement using nose tip (landmark 1)
                nose_tip = landmarks[1]
                nose_x = int(nose_tip.x * metadata['width'])
                nose_y = int(nose_tip.y * metadata['height'])
                
                # Draw nose tip
                cv2.circle(annotated_frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
                
                if prev_nose_tip is not None:
                    # Calculate head movement
                    dx = nose_x - prev_nose_tip[0]
                    dy = nose_y - prev_nose_tip[1]
                    movement = np.sqrt(dx**2 + dy**2)
                    head_movements.append(movement)
                    
                    # Draw movement vector
                    cv2.arrowedLine(annotated_frame, prev_nose_tip, (nose_x, nose_y),
                                  (0, 255, 255), 2, tipLength=0.3)
                
                prev_nose_tip = (nose_x, nose_y)
                
                # Add face label
                cv2.putText(annotated_frame, f"Face {face_idx + 1}",
                           (x_min_px, y_min_px - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Add face size info
                cv2.putText(annotated_frame, f"Size: {face_width}x{face_height}",
                           (x_min_px, y_max_px + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            faces_detected.append(0)
            prev_nose_tip = None
        
        # Add statistics overlay
        cv2.putText(annotated_frame, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Faces Detected: {num_faces}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0) if num_faces > 0 else (0, 0, 255), 2)
        
        if head_movements:
            cv2.putText(annotated_frame, f"Head Movement: {head_movements[-1]:.2f} px",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(annotated_frame)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    face_mesh.close()
    out.release()
    
    # Print statistics
    frames_with_faces = sum(1 for count in faces_detected if count > 0)
    detection_rate = (frames_with_faces / len(frames)) * 100
    
    print(f"\nFace Detection Statistics:")
    print(f"  Frames with faces: {frames_with_faces}/{len(frames)} ({detection_rate:.1f}%)")
    print(f"  Frames with no faces: {len(frames) - frames_with_faces}")
    
    if faces_detected:
        total_faces = sum(faces_detected)
        print(f"  Total face detections: {total_faces}")
    
    if face_sizes:
        print(f"\nFace Size Statistics:")
        print(f"  Average face size: {np.mean(face_sizes):.0f} sq pixels")
        print(f"  Max face size: {np.max(face_sizes):.0f} sq pixels")
        print(f"  Min face size: {np.min(face_sizes):.0f} sq pixels")
    
    if head_movements:
        print(f"\nHead Movement Statistics:")
        print(f"  Average movement: {np.mean(head_movements):.2f} px/frame")
        print(f"  Max movement: {np.max(head_movements):.2f} px/frame")
        
        avg_movement = np.mean(head_movements)
        if avg_movement < 2:
            print("  → Head movement: STILL (stationary)")
        elif avg_movement < 5:
            print("  → Head movement: SUBTLE (small movements)")
        elif avg_movement < 15:
            print("  → Head movement: MODERATE (normal talking/gesturing)")
        else:
            print("  → Head movement: ACTIVE (turning, nodding frequently)")
    
    num_landmarks = 468 if not FACE_MESH['refine_landmarks'] else 478
    print(f"\nOutput saved to: {output_path}")
    print("\nVisualization includes:")
    print(f"  - Face mesh ({num_landmarks} landmarks)")
    print("  - Face contours and tesselation")
    if FACE_MESH['refine_landmarks']:
        print("  - Iris tracking (refined landmarks)")
    print("  - Head movement tracking")
    
    return output_path

if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_face_mesh(frames, metadata)