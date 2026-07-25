#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 2: YOLOv11 + ByteTrack Integration

This script loads a pretrained YOLO11 model, runs detection and multi-object tracking (ByteTrack)
on students and cell phones, draws bounding boxes & tracking IDs, saves the output video,
and provides a display helper for Google Colab/HTML presentation environments.
"""

import os
import sys
import logging
import subprocess
from dataclasses import dataclass
from ultralytics import YOLO

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_2.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Module2_Tracking")

@dataclass
class TrackingConfig:
    # Source configuration
    video_path: str = r"C:\Users\rishi\Downloads\WhatsApp Video 2026-07-25 at 10.59.18 PM.mp4"
    
    # Model parameters
    model_name: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"  # Built-in ByteTrack configuration in ultralytics
    
    # Class IDs for detection (from COCO dataset):
    # 0 = person (student)
    # 67 = cell phone
    classes_of_interest = [0, 67]
    
    # Thresholds
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    
    # Save configuration
    output_dir: str = os.path.join(PROJECT_DIR, "output")
    verbose: bool = True

def check_video_source(video_path: str) -> bool:
    if not os.path.exists(video_path):
        logger.error(f"Target video path not found: {video_path}")
        return False
    logger.info(f"Target video found: {video_path}")
    return True

def run_bytetrack(config: TrackingConfig):
    logger.info("--- Starting YOLO11 + ByteTrack Integration ---")
    
    # Load YOLO11 Model
    logger.info(f"Initializing model {config.model_name}...")
    model = YOLO(config.model_name)
    
    # Set output run name
    run_name = "bytetrack_run"
    
    logger.info(f"Running tracking on source: {config.video_path}")
    
    # Run tracking using model.track()
    # This natively handles detection, ByteTrack logic, tracking IDs maintenance,
    # and automatically saves the annotated video output (using ultralytics backend).
    results = model.track(
        source=config.video_path,
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        tracker=config.tracker_config,
        persist=True,
        save=True,
        classes=config.classes_of_interest,
        project=os.path.abspath(config.output_dir),
        name=run_name,
        exist_ok=True,
        verbose=config.verbose
    )
    
    logger.info("Tracking completed successfully.")
    
    # Locating the saved video
    filename = os.path.basename(config.video_path)
    base_name, _ = os.path.splitext(filename)
    
    # Detect output formats
    saved_dir = os.path.abspath(os.path.join(config.output_dir, run_name))
    fallback_dir = os.path.abspath(os.path.join("runs", "detect", config.output_dir, run_name))
    
    possible_files = [
        os.path.join(saved_dir, f"{base_name}.avi"),
        os.path.join(saved_dir, f"{base_name}.mp4"),
        os.path.join(fallback_dir, f"{base_name}.avi"),
        os.path.join(fallback_dir, f"{base_name}.mp4")
    ]
    
    saved_path = None
    for pf in possible_files:
        if os.path.exists(pf):
            saved_path = pf
            break
            
    if saved_path:
        logger.info(f"Annotated tracking video saved at: {saved_path}")
        return saved_path
    else:
        logger.error(f"Could not locate the saved output video from Ultralytics track function. Checked paths: {possible_files}")
        return None

def convert_to_h264_mp4(input_path: str):
    """
    Transcode the saved output video to H264 MP4 so it is compatible
    with modern browser engines and Google Colab HTML players.
    """
    if not input_path:
        return None
        
    output_mp4_path = input_path.replace(".avi", "_h264.mp4")
    if not output_mp4_path.endswith(".mp4"):
        output_mp4_path += "_h264.mp4"
        
    logger.info("Transcoding annotated tracking video to H.264 MP4 for web display...")
    
    # Run FFMPEG command
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-f", "mp4",
        output_mp4_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Transcoded video successfully saved to: {output_mp4_path}")
        return output_mp4_path
    except Exception as e:
        logger.warning(f"FFMPEG conversion failed: {e}. Video display in Google Colab may be limited.")
        return input_path

def display_colab_video(video_path: str):
    """
    Outputs HTML element for displaying video in Colab if executed in Colab environment.
    """
    if not video_path or not os.path.exists(video_path):
        return
        
    # Check if running in Google Colab
    in_colab = 'google.colab' in sys.modules
    
    if in_colab:
        logger.info("Google Colab detected. Embedding HTML Video Player...")
        from IPython.display import HTML
        import base64
        
        video_file = open(video_path, "r+b").read()
        video_url = f"data:video/mp4;base64,{base64.b64encode(video_file).decode()}"
        
        # Display embedded video
        display(HTML(f"""
        <video width="640" height="480" controls>
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """))
    else:
        logger.info("Local environment detected. Video HTML output skipped. "
                    f"You can view the output video manually at: {video_path}")

def main():
    config = TrackingConfig()
    
    # Verify input video source
    if not check_video_source(config.video_path):
        logger.error("Exiting tracking task: Input video path is invalid.")
        return
        
    # Run detection and tracking
    saved_path = run_bytetrack(config)
    
    if saved_path:
        # Transcode video to H264 MP4 for web display compatibility
        web_ready_path = convert_to_h264_mp4(saved_path)
        
        # Render display in Google Colab
        display_colab_video(web_ready_path)

if __name__ == "__main__":
    main()
