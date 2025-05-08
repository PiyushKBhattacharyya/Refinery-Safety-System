# Oil Rig Safety Monitoring System

This system monitors video feeds to detect PPE (Personal Protective Equipment) violations in oil rig environments. It specifically tracks:
- Helmet usage
- Safety boots usage

## Features
- Real-time PPE violation detection
- Object tracking to avoid duplicate detections
- Automatic violation image capture and storage
- Configurable violation cooldown period

## Requirements
- Python 3.8 or higher
- Webcam or video feed
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the safety monitoring system:
```bash
python safety_monitor.py
```

2. The system will:
   - Open your webcam feed (or specified video source)
   - Display the video feed with detection boxes
   - Save violation images to the `violations` directory
   - Print violation alerts to the console

3. Press 'q' to quit the application

## Configuration

You can modify the following parameters in `safety_monitor.py`:
- `violation_cooldown`: Time in seconds between repeated violations (default: 30)
- Video source: Change `cv2.VideoCapture(0)` to use a different video source

## Output

Violation images are saved in the `violations` directory with the following naming format:
```
violation_[person_id]_[violation_type]_[timestamp].jpg
```

## Notes
- The system uses YOLOv8 for object detection
- ByteTrack is used for object tracking
- Violations are only recorded once per person within the cooldown period 