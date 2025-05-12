import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import os
from pathlib import Path

class SafetyMonitor:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Define PPE classes we want to detect
        self.ppe_classes = {
            'helmet': 0,
            'boots': 1,
            'vest': 2,  # Added safety vest
            'person': 3
        }
        
        # Initialize violation tracking
        self.violation_history = {}
        self.violation_cooldown = 30  # seconds
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
        # Run YOLO detection
        results = self.model(frame, classes=[0, 1, 2, 3])[0]
        
        # Convert results to supervision format
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        
        # Track objects
        detections = self.tracker.update_with_detections(detections)
        
        # Check for violations
        violations = self.check_violations(detections)
        
        # Create labels for annotations
        labels = [
            f"{self.model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        
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
            has_helmet = False
            has_boots = False
            has_vest = False
            
            for class_id in detected_classes:
                if class_id == self.ppe_classes['helmet']:
                    has_helmet = True
                elif class_id == self.ppe_classes['boots']:
                    has_boots = True
                elif class_id == self.ppe_classes['vest']:
                    has_vest = True
            
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
    
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
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
        
        # Display frame
        cv2.imshow('Safety Monitoring', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 