# INSTRUCTIONS:
# Download the PPE YOLOv8 model weights from Roboflow or similar source:
# wget https://github.com/roboflow-ai/models/releases/download/yolov8n-ppe-helmet/yolov8n-ppe-helmet.pt -O yolov8n-ppe-helmet.pt
# Place yolov8n-ppe-helmet.pt in your project directory.
#
# This model detects: 0=person, 1=helmet, 2=vest, 3=boots

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import os
from pathlib import Path
import requests
from roboflow import Roboflow 

class SafetyMonitor:
    def __init__(self):
        # Initialize Roboflow model
        rf = Roboflow(api_key="35phBzEeTnRLGk2aHLqx")
        project = rf.workspace().project("ppe-detection-public-kqerh")
        self.model = project.version("2").model
        
        # Define PPE classes as per the model
        self.ppe_classes = {
            'person': 0,
            'helmet': 1,
            'vest': 2,
            'boots': 3
        }
        
        # Initialize violation tracking
        self.violation_history = {}
        self.violation_cooldown = 50  # seconds
        self.last_violation_time = {}
        
        # Create violations directory if it doesn't exist
        self.violations_dir = Path("violations")
        self.violations_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        
    def add_warning_text(self, frame, text, position, index):
        # Calculate y position based on index to avoid overlapping
        y_position = position[1] + (index * 40)
        
        # Add red background rectangle
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame, 
                     (position[0], y_position - 30),
                     (position[0] + text_size[0], y_position + 10),
                     (0, 0, 255),
                     -1)  # Filled rectangle
        
        # Add white text
        cv2.putText(frame, text,
                    (position[0], y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    def process_frame(self, frame):
        # Run Roboflow model prediction
        # Convert frame to RGB and save as temp file for Roboflow SDK
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, rgb_frame)
        results = self.model.predict(temp_path, confidence=40, overlap=30).json()
        os.remove(temp_path)

        # Parse results to match previous format
        preds = results['predictions']
        if len(preds) == 0:
            xyxy = np.zeros((0, 4))
            confidence = np.zeros((0,))
            class_id = np.zeros((0,), dtype=int)
        else:
            xyxy = np.array([
                [pred['x'] - pred['width'] / 2, pred['y'] - pred['height'] / 2,
                 pred['x'] + pred['width'] / 2, pred['y'] + pred['height'] / 2]
                for pred in preds
            ])
            confidence = np.array([pred['confidence'] for pred in preds])
            class_id = np.array([
                pred['class_id'] if 'class_id' in pred else self.ppe_classes.get(pred['class'], -1)
                for pred in preds
            ])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Print detected class IDs for debugging
        print("Detected class IDs:", detections.class_id)
        
        # Track objects
        detections = self.tracker.update_with_detections(detections)
        
        # Check for violations
        violations = self.check_violations(detections)
        
        # Draw bounding boxes
        annotated_frame = frame.copy()
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for PPE items
            if class_id == self.ppe_classes['person']:
                color = (0, 0, 255)  # Red for person
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{self.model.names[class_id]} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add warning signs for violations
        warning_position = (10, 50)  # Starting position for warnings
        warning_index = 0
        
        # Group detections by tracker_id
        tracked_objects = {}
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            if tracker_id not in tracked_objects:
                tracked_objects[tracker_id] = []
            tracked_objects[tracker_id].append(detections.class_id[i])
        
        # Check each person for missing PPE
        for person_id, detected_classes in tracked_objects.items():
            has_helmet = any(class_id == self.ppe_classes['helmet'] for class_id in detected_classes)
            has_boots = any(class_id == self.ppe_classes['boots'] for class_id in detected_classes)
            has_vest = any(class_id == self.ppe_classes['vest'] for class_id in detected_classes)
            
            if not has_helmet:
                self.add_warning_text(annotated_frame, f"⚠ WARNING: Person {person_id} - NO HELMET", warning_position, warning_index)
                warning_index += 1
            
            if not has_boots:
                self.add_warning_text(annotated_frame, f"⚠ WARNING: Person {person_id} - NO SAFETY BOOTS", warning_position, warning_index)
                warning_index += 1
            
            if not has_vest:
                self.add_warning_text(annotated_frame, f"⚠ WARNING: Person {person_id} - NO SAFETY VEST", warning_position, warning_index)
                warning_index += 1
        
        return annotated_frame, violations
    
    def check_violations(self, detections):
        current_time = datetime.now()
        violations = []
        
        # Group detections by tracker_id
        tracked_objects = {}
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            if tracker_id not in tracked_objects:
                tracked_objects[tracker_id] = []
            tracked_objects[tracker_id].append(detections.class_id[i])
        
        # Check violations for each tracked person
        for person_id, detected_classes in tracked_objects.items():
            # Check if person has all required PPE
            has_helmet = any(class_id == self.ppe_classes['helmet'] for class_id in detected_classes)
            has_boots = any(class_id == self.ppe_classes['boots'] for class_id in detected_classes)
            has_vest = any(class_id == self.ppe_classes['vest'] for class_id in detected_classes)
            
            # Check for violations
            missing_ppe = []
            if not has_helmet:
                missing_ppe.append("No Helmet")
            if not has_boots:
                missing_ppe.append("No Safety Boots")
            if not has_vest:
                missing_ppe.append("No Safety Vest")
            
            if missing_ppe:
                violation_type = " & ".join(missing_ppe)
                
                # Check if enough time has passed since last violation
                if (person_id not in self.last_violation_time or 
                    (current_time - self.last_violation_time[person_id]).seconds > self.violation_cooldown):
                    
                    violations.append({
                        'person_id': person_id,
                        'type': violation_type,
                        'timestamp': current_time
                    })
                    
                    self.last_violation_time[person_id] = current_time
        
        return violations
    
    def save_violation(self, frame, violation):
        timestamp = violation['timestamp'].strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{violation['person_id']}_{violation['type']}_{timestamp}.jpg"
        filepath = self.violations_dir / filename
        cv2.imwrite(str(filepath), frame)

def main():
    # Initialize safety monitor
    monitor = SafetyMonitor()
    
    # Initialize video capture with the specified video file
    cap = cv2.VideoCapture('clips/clip2.mp4')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, violations = monitor.process_frame(frame)
        
        # Save violations
        for violation in violations:
            monitor.save_violation(frame, violation)
            print(f"Violation detected: {violation['type']} for person {violation['person_id']}")
        
        # Create side by side preview
        # Resize frames to have the same height
        height = 480  # Set a fixed height
        width = int(frame.shape[1] * (height / frame.shape[0]))
        dim = (width, height)
        
        original_resized = cv2.resize(frame, dim)
        processed_resized = cv2.resize(processed_frame, dim)
        
        # Combine frames horizontally
        preview = np.hstack((original_resized, processed_resized))
        
        # Add labels above each frame
        cv2.putText(preview, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(preview, "Safety Monitor", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite('preview.jpg', preview)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 