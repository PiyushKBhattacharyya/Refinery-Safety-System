from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from datetime import timedelta
import uuid
import sqlite3

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for processed videos
PROCESSED_DIR = "processed_videos"
os.makedirs(PROCESSED_DIR, exist_ok=True)
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_DIR), name="processed_videos")

# Load YOLOv8 model (adjust path as needed)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Get class names from model
CLASS_NAMES = model.names  # dict: {0: 'class1', 1: 'class2', ...}
COOLDOWN_SECONDS = 5
FRAME_SKIP = 2
HEIGHT_THRESHOLD = 0.6  # 60% of image height; adjust as needed

DB_PATH = 'violations.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
        video_id TEXT,
        type TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Generate a unique video_id
    video_id = str(uuid.uuid4())
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    # Prepare processed video path
    processed_filename = f"processed_{os.path.basename(temp_path)}"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)

    # Process video
    violations = []
    last_violation_time = {}  # {class_name: last_timestamp}
    stats = {}  # {class_name: count}
    frame_detections = []  # List of {frame: int, detections: [{class, bbox}]}

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
    frame_count = 0
    last_frame_boxes = []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_sec = frame_count / fps
        frame_boxes = []
        if frame_count % FRAME_SKIP == 0:
            results = model(frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = CLASS_NAMES[class_id]
                    # Draw bounding box
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                    cv2.putText(frame, class_name, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    # Add detection for frontend overlay
                    frame_boxes.append({
                        "class": class_name,
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    })
                    # Cooldown logic for regular violations
                    last_time = last_violation_time.get(class_name, -COOLDOWN_SECONDS)
                    if timestamp_sec - last_time >= COOLDOWN_SECONDS:
                        violation = {
                            "type": class_name,
                            "timestamp": str(timedelta(seconds=int(timestamp_sec)))
                        }
                        violations.append(violation)
                        last_violation_time[class_name] = timestamp_sec
                        stats[class_name] = stats.get(class_name, 0) + 1
                        c.execute("INSERT INTO violations (video_id, type, timestamp) VALUES (?, ?, ?)",
                                  (video_id, class_name, str(timedelta(seconds=int(timestamp_sec)))))
                    # At height detection for person
                    if class_name.lower() == 'person':
                        y2 = int(xyxy[3])
                        img_h = frame.shape[0]
                        if y2 < int(HEIGHT_THRESHOLD * img_h):
                            # Cooldown for at height
                            last_time = last_violation_time.get('at_height', -COOLDOWN_SECONDS)
                            if timestamp_sec - last_time >= COOLDOWN_SECONDS:
                                violation = {
                                    "type": "Person at height",
                                    "timestamp": str(timedelta(seconds=int(timestamp_sec)))
                                }
                                violations.append(violation)
                                last_violation_time['at_height'] = timestamp_sec
                                stats['Person at height'] = stats.get('Person at height', 0) + 1
                                c.execute("INSERT INTO violations (video_id, type, timestamp) VALUES (?, ?, ?)",
                                          (video_id, "Person at height", str(timedelta(seconds=int(timestamp_sec)))))
            last_frame_boxes = frame_boxes
        else:
            # For skipped frames, use last detections for overlay (no new violation check)
            frame_boxes = last_frame_boxes
        frame_detections.append({
            "frame": frame_count,
            "detections": frame_boxes
        })
        out.write(frame)
        frame_count += 1

    conn.commit()
    conn.close()
    cap.release()
    out.release()
    os.remove(temp_path)

    processed_url = f"/processed_videos/{processed_filename}"
    return JSONResponse({
        "video_id": video_id,
        "violations": violations,
        "stats": stats,
        "processed_video_url": processed_url,
        "frame_detections": frame_detections,
        "fps": fps
    })

@app.post("/detect_frame")
async def detect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(frame)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "class": class_name,
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            })
    return JSONResponse({"detections": detections})

@app.get("/violations/{video_id}")
def get_violations(video_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT type, timestamp FROM violations WHERE video_id = ?", (video_id,))
    rows = c.fetchall()
    conn.close()
    return JSONResponse({
        "video_id": video_id,
        "violations": [{"type": row[0], "timestamp": row[1]} for row in rows]
    }) 