#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 3: MediaPipe BlazePose + YOLO + ByteTrack Integration

This script integrates student tracking (YOLO11 + ByteTrack) with pose landmark extraction.
For each tracked student, it crops the ROI, runs MediaPipe BlazePose, extracts 33 landmarks,
maps them to global frame coordinates, draws the skeleton, and logs landmark coordinates.
"""

import os
import sys
import json
import logging
import urllib.request
import cv2
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_3.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Module3_Pose")

# Define Pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso
    (11, 13), (13, 15), # Left arm
    (12, 14), (14, 16), # Right arm
    (23, 25), (25, 27), # Left leg
    (24, 26), (26, 28)  # Right leg
]

@dataclass
class PoseConfig:
    video_path: str = r"C:\Users\rishi\Downloads\WhatsApp Video 2026-07-25 at 10.59.18 PM.mp4"
    yolo_model_name: str = "yolo11n.pt"
    pose_model_url: str = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    pose_model_name: str = "pose_landmarker_full.task"
    
    # Confidence thresholds
    yolo_conf: float = 0.35
    pose_conf: float = 0.50
    
    # Output configurations
    output_video_path: str = os.path.join(PROJECT_DIR, "output", "tracked_pose_output.mp4")
    output_json_path: str = os.path.join(PROJECT_DIR, "output", "pose_landmarks.json")

def download_pose_model(config: PoseConfig):
    model_path = os.path.join(PROJECT_DIR, "models", config.pose_model_name)
    if not os.path.exists(model_path):
        logger.info(f"Downloading MediaPipe BlazePose model from: {config.pose_model_url}")
        try:
            urllib.request.urlretrieve(config.pose_model_url, model_path)
            logger.info("BlazePose model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download BlazePose model: {e}")
            raise
    else:
        logger.info("BlazePose model already exists.")
    return model_path

def initialize_mediapipe_landmarker(model_path: str, config: PoseConfig):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    logger.info("Initializing MediaPipe Tasks PoseLandmarker...")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.pose_conf
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return landmarker

def run_pose_pipeline(config: PoseConfig):
    # Download models and initialize pipelines
    pose_model_path = download_pose_model(config)
    landmarker = initialize_mediapipe_landmarker(pose_model_path, config)
    
    # Initialize YOLO11 Tracker
    logger.info("Initializing YOLO11 detector...")
    yolo_model = YOLO(config.yolo_model_name)
    
    # Open Video
    logger.info(f"Opening video: {config.video_path}")
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error("Could not open input video stream.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video Info - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # Define Video Writer (using MP4V for maximum compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output_video_path, fourcc, fps, (width, height))
    
    # Landmark storage dict structured as:
    # { student_id: { frame_index: [ {x: val, y: val, z: val, visibility: val}, ... ] } }
    landmark_history = {}
    
    frame_idx = 0
    import mediapipe as mp
    
    logger.info("--- Starting Multi-Student Tracking & Pose Landmarking ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO + ByteTrack
        # We process the frame using model.track with classes=[0] (person)
        # Note: stream=True returns a generator for memory efficiency
        results = yolo_model.track(
            source=frame,
            conf=config.yolo_conf,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        # If tracking detections are found
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for bbox, student_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = bbox
                
                # Clip coordinates to frame limits
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Crop ROI of student
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                    
                # Convert ROI to RGB for MediaPipe Tasks
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                
                # Run pose detection on the cropped student
                pose_result = landmarker.detect(mp_image)
                
                # Process landmarks
                if pose_result.pose_landmarks:
                    # Get the first pose detected in ROI
                    landmarks = pose_result.pose_landmarks[0]
                    roi_w, roi_h = x2 - x1, y2 - y1
                    
                    global_landmarks = []
                    stored_landmarks = []
                    
                    # Map normalized ROI coordinates back to full frame
                    for lm in landmarks:
                        gx = x1 + int(lm.x * roi_w)
                        gy = y1 + int(lm.y * roi_h)
                        global_landmarks.append((gx, gy))
                        
                        # Store in output list
                        stored_landmarks.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        })
                        
                    # Save coordinates associated with Student ID and frame index
                    if student_id not in landmark_history:
                        landmark_history[int(student_id)] = {}
                    landmark_history[int(student_id)][int(frame_idx)] = stored_landmarks
                    
                    # Draw Skeleton on the original frame
                    for connection in POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(global_landmarks) and end_idx < len(global_landmarks):
                            pt1 = global_landmarks[start_idx]
                            pt2 = global_landmarks[end_idx]
                            cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
                            
                    # Draw landmark joints
                    for pt in global_landmarks:
                        cv2.circle(annotated_frame, pt, 3, (0, 0, 255), -1)
                        
                # Draw Student Bounding Box & ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"Student ID: {student_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
                
        # Write annotated frame to video file
        out.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"Processed frame {frame_idx}/{total_frames}...")
            
    cap.release()
    out.release()
    
    # Save landmark coordinates to JSON file
    logger.info(f"Writing landmark coordinates to JSON: {config.output_json_path}")
    with open(config.output_json_path, "w") as f:
        json.dump(landmark_history, f, indent=4)
        
    logger.info(f"Module 3 completed. Annotated video saved at: {config.output_video_path}")
    return config.output_video_path

def main():
    config = PoseConfig()
    
    if not os.path.exists(config.video_path):
        logger.error(f"Input video file not found at: {config.video_path}")
        return
        
    run_pose_pipeline(config)

if __name__ == "__main__":
    main()
