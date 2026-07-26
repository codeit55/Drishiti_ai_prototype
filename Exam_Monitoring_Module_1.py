#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 1: System Initialization, Dependency Setup, and Hardware Verification

This script initializes the environment, directory structure, checks GPU availability,
loads YOLO11, configures robust logging, and verifies MediaPipe pose/mesh.
"""

import os
import sys
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass

def create_directory_structure(project_dir, subdirs):
    print("\n--- Step 1: Creating Project Directory Structure ---")
    os.makedirs(project_dir, exist_ok=True)
    for subdir in subdirs:
        path = os.path.join(project_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created/Verified directory: {path}")
    print("Directory setup completed successfully.")

def install_dependencies(dependencies):
    print("\n--- Step 2: Installing/Upgrading System Dependencies ---")
    for dep in dependencies:
        print(f"Installing/Updating {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-q", dep])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {dep}: {e}")
            raise
    print("All dependencies installed successfully.")

def check_gpu():
    print("\n--- Step 3: Hardware & GPU Verification ---")
    import torch
    gpu_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Device Name: {device_name}")
    else:
        print("Warning: Running on CPU. Performance might be slow for real-time inference.")
    return gpu_available

def verify_libraries():
    print("\n--- Step 4: Importing Core Libraries & Verifying Versions ---")
    import cv2
    import numpy as np
    import pandas as pd
    import mediapipe as mp
    import ultralytics
    
    print(f"OpenCV Version: {cv2.__version__}")
    try:
        print(f"MediaPipe Version: {mp.__version__}")
    except AttributeError:
        print("MediaPipe Version: Unknown")
    print(f"Ultralytics Version: {ultralytics.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")

def setup_logging(project_dir):
    print("\n--- Step 5: Setting Up Logging Infrastructure ---")
    log_file_path = os.path.join(project_dir, "logs", "system.log")
    
    # Setup custom formatter
    log_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger("ExamMonitor")
    logger.setLevel(logging.DEBUG)
    
    # File handler with rotation (max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate outputs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging infrastructure successfully initialized.")
    return logger

@dataclass
class ExamMonitorConfig:
    # Input source
    video_path: str = r"/Users/harshsharma/Downloads/WhatsApp Video 2026-07-25 at 10.59.18 PM.mp4"    
    # Model parameters
    yolo_model_name: str = "yolo11n.pt"  # Pre-trained YOLO11 Nano model
    confidence_threshold: float = 0.40
    
    # MediaPipe settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Behavioral detection thresholds
    head_turn_threshold: float = 15.0 # Degrees
    gaze_deviation_threshold: float = 0.15 # Normalized ratio
    
    # Storage settings
    project_dir: str = "exam_monitoring_system"
    output_dir: str = os.path.join("exam_monitoring_system", "output")
    log_dir: str = os.path.join("exam_monitoring_system", "logs")
    
    # Logging flag
    verbose: bool = True

def verify_yolo(config, logger):
    print("\n--- Step 7: Loading and Verifying Pretrained YOLO11 ---")
    from ultralytics import YOLO
    import numpy as np
    
    try:
        model_path = os.path.join(config.project_dir, "models", config.yolo_model_name)
        logger.info(f"Loading YOLO11 model: {config.yolo_model_name}")
        
        # Initialize YOLO11 model
        model = YOLO(config.yolo_model_name)
        model.save(model_path) # Cache copy to models directory
        logger.info("YOLO11 loaded successfully. Running dummy verification inference...")
        
        # Create a blank image to run inference on to verify model functionality
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        
        logger.info(f"Inference check successful. Obtained {len(results)} results object(s).")
        print("YOLO11 loaded and verified successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load or verify YOLO11: {str(e)}", exc_info=True)
        raise

def verify_mediapipe(config, logger):
    print("\n--- Step 8: Verifying MediaPipe Pose & Face Mesh Solutions ---")
    import numpy as np
    
    try:
        try:
            import importlib
            mp_pose = importlib.import_module('mediapipe.solutions.pose')
            mp_face_mesh = importlib.import_module('mediapipe.solutions.face_mesh')
            
            pose = mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=config.min_detection_confidence
            )
            
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=config.min_detection_confidence
            )
            
            # Verification using dummy frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Process dummy frames
            pose.process(dummy_frame)
            face_mesh.process(dummy_frame)
            
            logger.info("MediaPipe Pose and FaceMesh (Legacy Solutions) initialized and processed dummy frames successfully.")
            print("MediaPipe Pose and FaceMesh verified successfully (Legacy API).")
            return pose, face_mesh
            
        except (ModuleNotFoundError, AttributeError):
            logger.info("Legacy MediaPipe solutions not found. Verifying using new Tasks API...")
            from mediapipe.tasks.python import vision
            
            # Verify Tasks API components are present
            if hasattr(vision, 'PoseLandmarker') and hasattr(vision, 'FaceLandmarker'):
                logger.info("MediaPipe Tasks API verified successfully. PoseLandmarker & FaceLandmarker are available.")
                print("MediaPipe Tasks API (PoseLandmarker & FaceLandmarker) verified successfully.")
                return vision.PoseLandmarker, vision.FaceLandmarker
            else:
                raise ImportError("MediaPipe Tasks API classes not found.")
                
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe models: {str(e)}", exc_info=True)
        raise

class VideoStreamVerifier:
    def __init__(self, path: str, logger):
        self.path = path
        self.logger = logger
        
    def check_file_exists(self) -> bool:
        exists = os.path.exists(self.path)
        if exists:
            self.logger.info(f"Input video file found at path: {self.path}")
        else:
            self.logger.warning(f"Input video file NOT found at: {self.path}")
        return exists
        
    def verify_stream(self) -> bool:
        import cv2
        if not self.check_file_exists():
            return False
            
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.logger.error("Could not open input video stream even though file exists.")
            return False
            
        # Try to read the first frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.logger.info(f"Video stream verified. Frame dimensions: {frame.shape}")
            return True
            
        self.logger.error("Could not read a frame from input video stream.")
        return False

def main():
    PROJECT_DIR = "exam_monitoring_system"
    SUBDIRS = ["data", "models", "logs", "output"]
    DEPENDENCIES = ["ultralytics", "mediapipe", "opencv-python", "numpy", "pandas"]
    
    # 1. Directories setup
    create_directory_structure(PROJECT_DIR, SUBDIRS)
    
    # 2. Package install
    install_dependencies(DEPENDENCIES)
    
    # 3. GPU check
    check_gpu()
    
    # 4. Library imports verification
    verify_libraries()
    
    # 5. Logging setup
    logger = setup_logging(PROJECT_DIR)
    
    # 6. Configuration setup
    config = ExamMonitorConfig()
    logger.info(f"Loaded config parameters: {config}")
    
    # 7. YOLOv11 Load
    verify_yolo(config, logger)
    
    # 8. MediaPipe Verification
    verify_mediapipe(config, logger)
    
    # 9. Video Verification & Error Handling
    print("\n--- Step 9: Verifying Input Video Path & Code Execution Stability ---")
    verifier = VideoStreamVerifier(config.video_path, logger)
    video_is_valid = verifier.verify_stream()
    
    if video_is_valid:
        print("Verification completed: Video is valid and ready for processing in Module 2.")
    else:
        print("WARNING: Video verification failed. Make sure the file path is correct or accessible.")

if __name__ == "__main__":
    main()
