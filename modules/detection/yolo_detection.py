import cv2
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import OUTPUT_DIR, YOLO_DETECTION

def analyze_yolo_detection(frames, metadata):
    """
    Object detection using YOLO (You Only Look Once)
    Detects common objects: person, face, and 80 COCO classes
    """
    output_path = os.path.join(OUTPUT_DIR, 'yolo_detection_output.mp4')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if YOLO files exist
    weights_path = YOLO_DETECTION['weights_path']
    config_path = YOLO_DETECTION['config_path']
    names_path = YOLO_DETECTION['names_path']
    
    if not os.path.exists(weights_path):
        print(f"\nERROR: YOLO weights not found at: {weights_path}")
        print("\nTo download YOLO files:")
        print("1. Create a 'yolo/' directory in your project")
        print("2. Download YOLOv4-tiny (lighter) or YOLOv4 (more accurate):")
        print("   YOLOv4-tiny:")
        print("   - wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
        print("   - wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg")
        print("   YOLOv4:")
        print("   - wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        print("   - wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        print("   Class names:")
        print("   - wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        return None
    
    # Load YOLO
    print("Loading YOLO model...")
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Set backend (try CUDA if available, else CPU)
    if YOLO_DETECTION['use_gpu']:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU acceleration")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU")
    
    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    # Statistics
    detection_counts = {cls: 0 for cls in classes}
    total_detections = 0
    frames_with_detections = 0
    
    print(f"Processing {len(frames)} frames...")
    
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                     swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run detection
        detections = net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > YOLO_DETECTION['confidence_threshold']:
                    # Object detected
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                   YOLO_DETECTION['confidence_threshold'],
                                   YOLO_DETECTION['nms_threshold'])
        
        frame_detections = 0
        
        if len(indices) > 0:
            frames_with_detections += 1
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                label = classes[class_ids[idx]]
                confidence = confidences[idx]
                color = colors[class_ids[idx]]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update statistics
                detection_counts[label] += 1
                total_detections += 1
                frame_detections += 1
        
        # Add frame info
        info_text = f"Frame: {i+1}/{len(frames)} | Detections: {frame_detections}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    out.release()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("YOLO OBJECT DETECTION STATISTICS")
    print(f"{'='*60}")
    print(f"\nTotal detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{len(frames)}")
    print(f"Detection rate: {frames_with_detections/len(frames)*100:.1f}%")
    
    print(f"\nDetections by class:")
    detected_classes = {k: v for k, v in detection_counts.items() if v > 0}
    if detected_classes:
        for cls, count in sorted(detected_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
    else:
        print("  No objects detected")
    
    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}")
    
    return output_path


# Allow standalone execution
if __name__ == "__main__":
    from preprocessing import preprocess_video, load_raw_video
    from config import INPUT_VIDEO, MAX_FRAMES
    
    print("Loading raw video...")
    frames, metadata = load_raw_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_yolo_detection(frames, metadata)