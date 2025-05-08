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
        
    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame, classes=[0, 1, 2, 3])[0]
        
        # Convert results to supervision format
        detections = sv.Detections.from_yolov8(results)
        
        # Track objects
        detections = self.tracker.update_with_detections(detections)
        
        # Check for violations
        violations = self.check_violations(detections)
        
        # Draw annotations
        frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=[f"{self.model.names[class_id]}" for class_id in detections.class_id]
        )
        
        return frame, violations
    
    def check_violations(self, detections):
        current_time = datetime.now()
        violations = []
        
        for detection in detections:
            person_id = detection.tracker_id
            if person_id is None:
                continue
                
            # Check if person has all required PPE
            has_helmet = False
            has_boots = False
            has_vest = False
            
            for class_id in detection.class_id:
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