#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 7: Complete Logging System

This module provides a reusable, production-grade logging class that ingests
detected behaviour events and risk score data from Modules 5 and 6, consolidates
all relevant metadata (timestamp, frame number, student ID, behaviour, confidence,
risk score, phone detection, body/head rotation), and writes structured historical
logs to a rotating CSV file.
"""

import os
import sys
import csv
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_7.log")
_file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
_file_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger = logging.getLogger("Module7_Logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# CSV Output Column definitions
CSV_COLUMNS = [
    "log_id",
    "timestamp_seconds",
    "frame_number",
    "student_id",
    "behaviour_name",
    "confidence",
    "risk_score",
    "risk_label",
    "phone_detected",
    "body_rotation",
    "head_yaw",
    "head_pitch",
    "explanation",
    "created_at"
]

@dataclass
class LoggerConfig:
    behaviours_json_path: str = os.path.join(PROJECT_DIR, "output", "detected_behaviours.json")
    risk_scores_json_path: str = os.path.join(PROJECT_DIR, "output", "risk_scores.json")
    features_json_path: str = os.path.join(PROJECT_DIR, "output", "student_features.json")
    output_csv_path: str = os.path.join(PROJECT_DIR, "output", "exam_monitoring_log.csv")
    fps: float = 30.0

class ExamMonitoringLogger:
    """
    Production-grade reusable logger for the Exam Monitoring System.
    Consolidates events from detected_behaviours, risk_scores, and student_features
    into a fully structured CSV log file.
    """
    def __init__(self, config: LoggerConfig):
        self.config = config
        self._log_id = 0
        self._csv_file = None
        self._csv_writer = None
        logger.info("ExamMonitoringLogger initialized.")

    def open(self):
        """Open the CSV writer and write the header row."""
        self._csv_file = open(self.config.output_csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._csv_writer.writeheader()
        logger.info(f"CSV log file opened at: {self.config.output_csv_path}")

    def close(self):
        """Flush and close the CSV writer."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            logger.info("CSV log file closed and flushed.")

    def write_event(
        self,
        timestamp: float,
        student_id: int,
        behaviour_name: str,
        confidence: float,
        risk_score: float,
        risk_label: str,
        phone_detected: bool,
        body_rotation: float,
        head_yaw: float,
        head_pitch: float,
        explanation: str = ""
    ):
        """Write a single event row into the CSV log."""
        self._log_id += 1
        frame_number = int(timestamp * self.config.fps)
        row = {
            "log_id":             self._log_id,
            "timestamp_seconds":  round(timestamp, 3),
            "frame_number":       frame_number,
            "student_id":         student_id,
            "behaviour_name":     behaviour_name,
            "confidence":         round(confidence, 4),
            "risk_score":         round(risk_score, 2),
            "risk_label":         risk_label,
            "phone_detected":     int(phone_detected),
            "body_rotation":      round(body_rotation, 4),
            "head_yaw":           round(head_yaw, 4),
            "head_pitch":         round(head_pitch, 4),
            "explanation":        explanation,
            "created_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._csv_writer.writerow(row)

    def build_feature_lookup(self, features_db: Dict) -> Dict:
        """
        Build a lookup map of (student_id, frame) → feature dict
        for fast join with behaviour events.
        """
        lookup = {}
        for sid, frames in features_db.items():
            for feat in frames:
                key = (int(sid), int(feat["frame"]))
                lookup[key] = feat
        return lookup

    def build_risk_lookup(self, risk_db: Dict) -> Dict:
        """
        Build a lookup map of (student_id, timestamp) → risk event dict.
        """
        lookup = {}
        for sid, summary in risk_db.items():
            for entry in summary.get("score_history", []):
                key = (int(sid), round(entry["timestamp"], 3))
                lookup[key] = entry
        return lookup

    def process_and_log(self):
        """
        Full pipeline: load data sources, join them, and write structured CSV logs.
        """
        # Load data sources
        if not os.path.exists(self.config.behaviours_json_path):
            logger.error(f"Behaviours file not found: {self.config.behaviours_json_path}")
            return
        if not os.path.exists(self.config.risk_scores_json_path):
            logger.error(f"Risk scores file not found: {self.config.risk_scores_json_path}")
            return
        if not os.path.exists(self.config.features_json_path):
            logger.error(f"Features file not found: {self.config.features_json_path}")
            return

        with open(self.config.behaviours_json_path, "r") as f:
            behaviours = json.load(f)
        with open(self.config.risk_scores_json_path, "r") as f:
            risk_data = json.load(f)
        with open(self.config.features_json_path, "r") as f:
            features_db = json.load(f)

        logger.info(f"Loaded {len(behaviours)} behaviour events.")
        feature_lookup = self.build_feature_lookup(features_db)
        risk_lookup = self.build_risk_lookup(risk_data)

        # Sort chronologically
        behaviours.sort(key=lambda x: (x["timestamp"], x["student_id"]))

        self.open()
        logged_count = 0

        for det in behaviours:
            sid = det["student_id"]
            ts = det["timestamp"]
            frame_num = int(ts * self.config.fps)
            behaviour = det["behaviour_name"]
            confidence = det["confidence"]

            # Look up matching feature for body/head data
            feat = feature_lookup.get((sid, frame_num), {})
            body_rotation = feat.get("body_rotation", 0.0)
            head_yaw = feat.get("head_yaw", 0.0)
            head_pitch = feat.get("head_pitch", 0.0)
            phone_dist = feat.get("object_hand_distance", -1.0)
            phone_detected = (phone_dist != -1.0 and phone_dist <= 180.0)

            # Look up risk score at this timestamp
            risk_key = (sid, round(ts, 3))
            risk_entry = risk_lookup.get(risk_key, {})
            risk_score = risk_entry.get("score_after", 0.0)
            risk_label = risk_entry.get("risk_label", "Safe")
            explanation = risk_entry.get("explanation", f"{behaviour} detected.")

            self.write_event(
                timestamp=ts,
                student_id=sid,
                behaviour_name=behaviour,
                confidence=confidence,
                risk_score=risk_score,
                risk_label=risk_label,
                phone_detected=phone_detected,
                body_rotation=body_rotation,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
                explanation=explanation
            )
            logged_count += 1

        self.close()
        logger.info(f"Module 7 complete. Logged {logged_count} events to: {self.config.output_csv_path}")

        # Preview first 10 rows
        print(f"\n--- CSV Log Preview (First 10 Rows) ---")
        with open(self.config.output_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                print(f"  [{row['log_id']}] T={row['timestamp_seconds']}s | Student {row['student_id']} | "
                      f"{row['behaviour_name']} | Conf={row['confidence']} | "
                      f"Risk={row['risk_score']} ({row['risk_label']}) | Phone={row['phone_detected']}")

def main():
    config = LoggerConfig()
    event_logger = ExamMonitoringLogger(config)
    event_logger.process_and_log()

if __name__ == "__main__":
    main()
