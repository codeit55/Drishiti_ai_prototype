#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 5: Rule-Based Behavior Detection

This script parses computed feature vectors from Module 4 and applies rule-based temporal
logic to flag cheating indicators (head turning, side glances, body rotation, phone/paper interaction,
looking away, suspicious leaning, and communication gestures).
"""

import os
import sys
import json
import logging
from dataclasses import dataclass

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_5.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Module5_Rules")

@dataclass
class RulesConfig:
    features_json_path: str = os.path.join(PROJECT_DIR, "output", "student_features.json")
    output_detected_path: str = os.path.join(PROJECT_DIR, "output", "detected_behaviours.json")
    fps: float = 30.0
    
    # Thresholds
    yaw_turn_threshold: float = 25.0          # Degrees
    pitch_down_threshold: float = 15.0        # Degrees (looking down)
    pitch_up_threshold: float = -15.0         # Degrees (looking up/straight)
    body_rot_threshold: float = 20.0          # Degrees
    wrist_vel_threshold: float = 1.2          # Units/sec
    phone_dist_threshold: float = 180.0        # Pixels
    torso_lean_threshold: float = 12.0        # Degrees

def analyze_student_behaviors(student_id, frames_features, config: RulesConfig):
    detections = []
    
    # Sort frames to guarantee temporal order
    frames_features.sort(key=lambda x: x["frame"])
    
    # Initialize rolling state variables
    yaw_history = []
    pitch_history = []
    torso_history = []
    
    # Helper to calculate confidence dynamically
    def get_confidence(val, threshold, max_offset=20.0):
        diff = abs(val) - threshold
        if diff < 0:
            return 0.0
        return float(min(1.0, 0.5 + (diff / max_offset) * 0.5))

    for idx, feat in enumerate(frames_features):
        frame = feat["frame"]
        timestamp = float(frame) / config.fps
        
        yaw = feat["head_yaw"]
        pitch = feat["head_pitch"]
        body_rot = feat["body_rotation"]
        wrist_vel = feat["wrist_velocity"]
        phone_dist = feat["object_hand_distance"]
        torso_angle = feat["torso_angle"]
        
        # Maintain sliding histories (last 3 seconds / 90 frames)
        yaw_history.append((frame, yaw))
        pitch_history.append((frame, pitch))
        torso_history.append((frame, torso_angle))
        if len(yaw_history) > 90:
            yaw_history.pop(0)
            pitch_history.pop(0)
            torso_history.pop(0)
            
        # 1. Head Turning
        if abs(yaw) > config.yaw_turn_threshold:
            conf = get_confidence(yaw, config.yaw_turn_threshold, 25.0)
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Head Turning",
                "timestamp": timestamp,
                "confidence": conf
            })
            
        # 2. Repeated Side Glances
        # Check for oscillating yaw patterns (transitions between left and right yaw)
        if len(yaw_history) >= 30:
            glance_transitions = 0
            last_sign = 0 # -1 for left, 1 for right
            for f, y in yaw_history:
                if y > 15.0 and last_sign != -1:
                    glance_transitions += 1
                    last_sign = -1
                elif y < -15.0 and last_sign != 1:
                    glance_transitions += 1
                    last_sign = 1
            if glance_transitions >= 3:
                conf = float(min(1.0, 0.6 + (glance_transitions - 3) * 0.1))
                detections.append({
                    "student_id": int(student_id),
                    "behaviour_name": "Repeated Side Glances",
                    "timestamp": timestamp,
                    "confidence": conf
                })
                
        # 3. Body Rotation
        if abs(body_rot) > config.body_rot_threshold:
            conf = get_confidence(body_rot, config.body_rot_threshold, 20.0)
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Body Rotation",
                "timestamp": timestamp,
                "confidence": conf
            })
            
        # 4. Hand Movement
        if wrist_vel > config.wrist_vel_threshold:
            conf = float(min(1.0, 0.5 + (wrist_vel - config.wrist_vel_threshold) / 2.0))
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Hand Movement",
                "timestamp": timestamp,
                "confidence": conf
            })
            
        # 5. Phone Interaction
        if phone_dist != -1.0 and phone_dist <= config.phone_dist_threshold:
            conf = float(min(1.0, 0.7 + ((config.phone_dist_threshold - phone_dist) / config.phone_dist_threshold) * 0.3))
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Phone Interaction",
                "timestamp": timestamp,
                "confidence": conf
            })
            
        # 6. Paper Interaction
        # Check if looking down continuously and hands show activity
        if len(pitch_history) >= 45:
            last_pitch_vals = [p for f, p in pitch_history[-45:]]
            if all(p > config.pitch_down_threshold for p in last_pitch_vals) and wrist_vel > 0.35:
                detections.append({
                    "student_id": int(student_id),
                    "behaviour_name": "Paper Interaction",
                    "timestamp": timestamp,
                    "confidence": 0.75
                })
                
        # 7. Looking Away
        # Flag if head is turned away (yaw > 40) or looking up/straight away (pitch < -15)
        if abs(yaw) > 40.0 or pitch < config.pitch_up_threshold:
            # Confirm duration criteria (last 30 frames / 1 second)
            if len(pitch_history) >= 30:
                last_yaw_vals = [abs(y) for f, y in yaw_history[-30:]]
                last_pitch_vals = [p for f, p in pitch_history[-30:]]
                if all(y > 40.0 for y in last_yaw_vals) or all(p < config.pitch_up_threshold for p in last_pitch_vals):
                    detections.append({
                        "student_id": int(student_id),
                        "behaviour_name": "Looking Away",
                        "timestamp": timestamp,
                        "confidence": 0.85
                    })
                    
        # 8. Suspicious Leaning
        if abs(torso_angle) > config.torso_lean_threshold:
            conf = get_confidence(torso_angle, config.torso_lean_threshold, 15.0)
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Suspicious Leaning",
                "timestamp": timestamp,
                "confidence": conf
            })
            
        # 9. Communication Gestures
        # Concurrently moving hands and turning body/head towards others
        if wrist_vel > 0.8 and abs(body_rot) > 15.0:
            detections.append({
                "student_id": int(student_id),
                "behaviour_name": "Communication Gestures",
                "timestamp": timestamp,
                "confidence": 0.78
            })
            
    return detections

def run_rules_pipeline(config: RulesConfig):
    logger.info("Loading student features database...")
    if not os.path.exists(config.features_json_path):
        logger.error(f"Features file not found at: {config.features_json_path}. Please execute Module 4 first.")
        return
        
    with open(config.features_json_path, "r") as f:
        feature_database = json.load(f)
        
    all_detections = []
    
    for student_id, frames_features in feature_database.items():
        logger.info(f"Analyzing behaviors for Student ID: {student_id}...")
        student_detections = analyze_student_behaviors(student_id, frames_features, config)
        all_detections.extend(student_detections)
        logger.info(f"Generated {len(student_detections)} behavior events for Student {student_id}.")
        
    # Sort detections by timestamp for chronological readability
    all_detections.sort(key=lambda x: (x["timestamp"], x["student_id"]))
    
    # Save detection logs
    logger.info(f"Writing behavioral detections to JSON: {config.output_detected_path}")
    with open(config.output_detected_path, "w") as f:
        json.dump(all_detections, f, indent=4)
        
    logger.info(f"Module 5 complete. Total behavior events detected: {len(all_detections)}")
    
    # Display sample output detections
    if all_detections:
        print("\n--- Sample Detected Behaviours (First 10 Events) ---")
        for i, det in enumerate(all_detections[:10]):
            print(f"[{i+1}] Student {det['student_id']}: {det['behaviour_name']} at {det['timestamp']:.2f}s (Conf: {det['confidence']:.2f})")
    else:
        print("\nNo suspicious behaviors detected based on the rules.")

def main():
    config = RulesConfig()
    run_rules_pipeline(config)

if __name__ == "__main__":
    main()
