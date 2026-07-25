#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 4: Feature Extraction and Engineering

This script processes video frames using YOLO11 tracking and MediaPipe BlazePose,
computes advanced behavioral feature vectors (head yaw/pitch, body/shoulder/torso rotation,
wrist velocity, movement speed, object-hand distance, and pose stability), and
visualizes/saves these features.
"""

import os
import sys
import json
import logging
import urllib.request
import cv2
import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_4.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Module4_Features")

@dataclass
class FeatureConfig:
    video_path: str = r"C:\Users\rishi\Downloads\WhatsApp Video 2026-07-25 at 10.59.18 PM.mp4"
    yolo_model_name: str = "yolo11n.pt"
    pose_model_url: str = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    pose_model_name: str = "pose_landmarker_full.task"
    
    # Confidence thresholds
    yolo_conf: float = 0.35
    pose_conf: float = 0.50
    
    # Window size for pose stability tracking (frames)
    stability_window: int = 10
    
    # Output configurations
    output_video_path: str = os.path.join(PROJECT_DIR, "output", "feature_extraction_output.mp4")
    output_features_path: str = os.path.join(PROJECT_DIR, "output", "student_features.json")

def download_pose_model(config: FeatureConfig):
    model_path = os.path.join(PROJECT_DIR, "models", config.pose_model_name)
    if not os.path.exists(model_path):
        logger.info(f"Downloading BlazePose model from: {config.pose_model_url}")
        try:
            urllib.request.urlretrieve(config.pose_model_url, model_path)
            logger.info("BlazePose model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download BlazePose model: {e}")
            raise
    return model_path

def initialize_pose_landmarker(model_path: str, config: FeatureConfig):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.pose_conf
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return landmarker

# --- Feature Extraction Functions ---

def compute_head_orientation(landmarks):
    """
    Computes head yaw and pitch based on relative distances and depth coordinate z.
    """
    nose = landmarks[0]
    l_ear = landmarks[7]
    r_ear = landmarks[8]
    l_eye = landmarks[2]
    r_eye = landmarks[5]
    
    # Yaw: nose horizontal projection relative to ear-to-ear span
    mid_ear_x = (l_ear.x + r_ear.x) / 2.0
    mid_ear_z = (l_ear.z + r_ear.z) / 2.0
    v_nose_x = nose.x - mid_ear_x
    v_nose_z = nose.z - mid_ear_z
    yaw = np.degrees(np.arctan2(v_nose_x, v_nose_z + 1e-6))
    
    # Pitch: nose vertical height relative to eye center
    mid_eye_y = (l_eye.y + r_eye.y) / 2.0
    mid_eye_z = (l_eye.z + r_eye.z) / 2.0
    v_pitch_y = nose.y - mid_eye_y
    v_pitch_z = nose.z - mid_eye_z
    pitch = np.degrees(np.arctan2(v_pitch_y, v_pitch_z + 1e-6))
    
    return float(yaw), float(pitch)

def compute_body_angles(landmarks):
    """
    Computes body rotation, shoulder tilt, and torso inclination.
    """
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    
    # Body Rotation: Angle of shoulders vector in the X-Z plane
    dx_sh = r_sh.x - l_sh.x
    dz_sh = r_sh.z - l_sh.z
    body_rotation = np.degrees(np.arctan2(dz_sh, dx_sh + 1e-6))
    
    # Shoulder Tilt: Angle relative to the horizontal in X-Y plane
    dy_sh = r_sh.y - l_sh.y
    shoulder_angle = np.degrees(np.arctan2(dy_sh, dx_sh + 1e-6))
    
    # Torso Angle: Mid-shoulder to mid-hip vector relative to gravity (Y vertical)
    mid_sh_x = (l_sh.x + r_sh.x) / 2.0
    mid_sh_y = (l_sh.y + r_sh.y) / 2.0
    mid_hip_x = (l_hip.x + r_hip.x) / 2.0
    mid_hip_y = (l_hip.y + r_hip.y) / 2.0
    
    dy_torso = mid_hip_y - mid_sh_y
    dx_torso = mid_hip_x - mid_sh_x
    torso_angle = np.degrees(np.arctan2(dx_torso, dy_torso + 1e-6))
    
    return float(body_rotation), float(shoulder_angle), float(torso_angle)

def compute_kinematics(student_history, current_landmarks, frame_idx, fps):
    """
    Computes hand trajectory, wrist velocity, and overall movement speed.
    """
    dt = 1.0 / fps if fps > 0 else 0.033
    l_wrist = current_landmarks[15]
    r_wrist = current_landmarks[16]
    l_sh = current_landmarks[11]
    r_sh = current_landmarks[12]
    
    torso_center = np.array([(l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0])
    curr_l_wrist = np.array([l_wrist.x, l_wrist.y])
    curr_r_wrist = np.array([r_wrist.x, r_wrist.y])
    
    wrist_velocity = 0.0
    movement_speed = 0.0
    
    # Extract historical coordinates if available
    prev = student_history.get("last_kinematics", None)
    if prev is not None:
        # Wrist displacement
        dl = np.linalg.norm(curr_l_wrist - prev["l_wrist"]) / dt
        dr = np.linalg.norm(curr_r_wrist - prev["r_wrist"]) / dt
        wrist_velocity = float((dl + dr) / 2.0)
        
        # Torso displacement (movement speed)
        movement_speed = float(np.linalg.norm(torso_center - prev["torso_center"]) / dt)
        
    # Update history dict
    student_history["last_kinematics"] = {
        "l_wrist": curr_l_wrist,
        "r_wrist": curr_r_wrist,
        "torso_center": torso_center
    }
    
    # Store hand trajectory points
    if "l_wrist_trajectory" not in student_history:
        student_history["l_wrist_trajectory"] = []
    if "r_wrist_trajectory" not in student_history:
        student_history["r_wrist_trajectory"] = []
        
    student_history["l_wrist_trajectory"].append(curr_l_wrist.tolist())
    student_history["r_wrist_trajectory"].append(curr_r_wrist.tolist())
    
    return wrist_velocity, movement_speed

def compute_object_hand_distance(current_landmarks, phone_boxes, frame_w, frame_h):
    """
    Computes minimum Euclidean distance between student's hands and any detected cell phone bounding box center.
    """
    l_wrist = current_landmarks[15]
    r_wrist = current_landmarks[16]
    
    # Convert normalized landmarks to absolute screen pixels
    lw_abs = np.array([l_wrist.x * frame_w, l_wrist.y * frame_h])
    rw_abs = np.array([r_wrist.x * frame_w, r_wrist.y * frame_h])
    
    min_dist = -1.0 # Default if no phones detected
    
    if len(phone_boxes) > 0:
        dists = []
        for box in phone_boxes:
            px1, py1, px2, py2 = box
            phone_center = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
            dists.append(np.linalg.norm(lw_abs - phone_center))
            dists.append(np.linalg.norm(rw_abs - phone_center))
        min_dist = float(min(dists))
        
    return min_dist

def compute_pose_stability(student_history, current_landmarks, config: FeatureConfig):
    """
    Computes pose stability (variation across landmarks over a rolling frame window).
    Lower value indicates higher stability.
    """
    # Flatten landmarks to a vector of coordinates
    current_vector = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks]).flatten()
    
    if "pose_window" not in student_history:
        student_history["pose_window"] = []
        
    student_history["pose_window"].append(current_vector)
    
    # Keep within sliding window size
    if len(student_history["pose_window"]) > config.stability_window:
        student_history["pose_window"].pop(0)
        
    # Calculate stability (std dev across window of frames)
    if len(student_history["pose_window"]) > 1:
        window_arr = np.array(student_history["pose_window"])
        std_devs = np.std(window_arr, axis=0)
        stability = float(np.mean(std_devs))
    else:
        stability = 0.0
        
    return stability

# --- Main pipeline ---

def run_feature_pipeline(config: FeatureConfig):
    # Prepare dependencies
    pose_model_path = download_pose_model(config)
    landmarker = initialize_pose_landmarker(pose_model_path, config)
    yolo_model = YOLO(config.yolo_model_name)
    
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error("Could not open input video source.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video loaded: {width}x{height} at {fps} FPS.")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output_video_path, fourcc, fps, (width, height))
    
    # Databases
    # Global tracking history for kinematics calculation
    tracking_history = {}
    
    # Feature vectors database structure:
    # { student_id: [ { frame: idx, head_yaw: val, ... }, ... ] }
    feature_database = {}
    
    frame_idx = 0
    import mediapipe as mp
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO Tracking
        # Class 0: person, Class 67: cell phone
        yolo_results = yolo_model.track(
            source=frame,
            conf=config.yolo_conf,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0, 67],
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        # Parse detections
        student_detections = []
        phone_boxes = []
        
        if yolo_results and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int) if yolo_results[0].boxes.id is not None else None
            
            for i, (bbox, cls_id) in enumerate(zip(boxes, class_ids)):
                if cls_id == 67: # Cell phone
                    phone_boxes.append(bbox)
                    # Draw phone bounding box
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "Phone", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif cls_id == 0 and track_ids is not None: # Student
                    student_detections.append((bbox, track_ids[i]))
                    
        # Process Pose and extract features for each student
        for bbox, student_id in student_detections:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            pose_result = landmarker.detect(mp_image)
            
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                
                # Compute Features
                yaw, pitch = compute_head_orientation(landmarks)
                body_rot, sh_angle, torso_angle = compute_body_angles(landmarks)
                
                if student_id not in tracking_history:
                    tracking_history[int(student_id)] = {}
                    
                wrist_vel, mov_speed = compute_kinematics(tracking_history[int(student_id)], landmarks, frame_idx, fps)
                hand_obj_dist = compute_object_hand_distance(landmarks, phone_boxes, width, height)
                pose_stability = compute_pose_stability(tracking_history[int(student_id)], landmarks, config)
                
                # Create Feature Vector
                features = {
                    "frame": int(frame_idx),
                    "head_yaw": yaw,
                    "head_pitch": pitch,
                    "body_rotation": body_rot,
                    "shoulder_angle": sh_angle,
                    "torso_angle": torso_angle,
                    "wrist_velocity": wrist_vel,
                    "movement_speed": mov_speed,
                    "object_hand_distance": hand_obj_dist,
                    "pose_stability": pose_stability
                }
                
                if int(student_id) not in feature_database:
                    feature_database[int(student_id)] = []
                feature_database[int(student_id)].append(features)
                
                # Visualize Features on frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Student {student_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Render feature statistics box beside student bounding box
                stats_y = y1 + 15
                cv2.putText(annotated_frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (x1 + 5, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Torso Ang: {torso_angle:.1f}", (x1 + 5, stats_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Wrist Vel: {wrist_vel:.3f}", (x1 + 5, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Stability: {pose_stability:.4f}", (x1 + 5, stats_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                if hand_obj_dist != -1.0:
                    cv2.putText(annotated_frame, f"Phone Dist: {hand_obj_dist:.1f}px", (x1 + 5, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                    
        out.write(annotated_frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"Processed frame {frame_idx}/{total_frames}...")
            
    cap.release()
    out.release()
    
    # Save Feature Vectors Database
    logger.info(f"Writing features database to JSON: {config.output_features_path}")
    with open(config.output_features_path, "w") as f:
        json.dump(feature_database, f, indent=4)
        
    logger.info(f"Module 4 complete. Output saved at: {config.output_video_path}")

def main():
    config = FeatureConfig()
    if not os.path.exists(config.video_path):
        logger.error(f"Input video file not found: {config.video_path}")
        return
        
    run_feature_pipeline(config)

if __name__ == "__main__":
    main()
