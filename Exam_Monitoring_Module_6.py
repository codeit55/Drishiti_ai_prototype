#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 6: Explainable Rolling Risk Scoring Engine

This script processes detected behaviour events from Module 5, computes a rolling cumulative
risk score (0-100) per student, assigns a risk label (Safe/Low/Medium/High/Critical),
generates human-readable explanations for each score change, and visualizes the live
risk score per student frame-by-frame.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Configure local directory structure
PROJECT_DIR = "exam_monitoring_system"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)

# Logging Setup
log_file_path = os.path.join(PROJECT_DIR, "logs", "module_6.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Module6_RiskScoring")

# Risk Score Labels
RISK_LABELS = {
    (0, 20):   "Safe",
    (20, 40):  "Low",
    (40, 60):  "Medium",
    (60, 80):  "High",
    (80, 101): "Critical"
}

# Behaviour Weights (out of 100 max total)
BEHAVIOUR_WEIGHTS = {
    "Head Turning":             8.0,
    "Repeated Side Glances":   15.0,
    "Body Rotation":            7.0,
    "Hand Movement":            5.0,
    "Phone Interaction":       25.0,
    "Paper Interaction":        6.0,
    "Looking Away":            10.0,
    "Suspicious Leaning":       6.0,
    "Communication Gestures":  12.0
}

# Temporal duration bonus multipliers (behaviours persisting longer are more suspicious)
DURATION_MULTIPLIER = {
    "Phone Interaction":       1.5,
    "Repeated Side Glances":   1.4,
    "Looking Away":            1.3,
    "Communication Gestures":  1.2,
    "Head Turning":            1.1,
    "Body Rotation":           1.0,
    "Hand Movement":           1.0,
    "Suspicious Leaning":      1.0,
    "Paper Interaction":       1.0
}

@dataclass
class RiskScoringConfig:
    behaviours_json_path: str = os.path.join(PROJECT_DIR, "output", "detected_behaviours.json")
    output_risk_path: str = os.path.join(PROJECT_DIR, "output", "risk_scores.json")
    fps: float = 30.0
    
    # Score decay per second when no behaviour is detected (gradual cooldown)
    decay_rate_per_frame: float = 0.05
    
    # Repeated behaviour bonus within a rolling window
    repeat_window_seconds: float = 10.0
    repeat_bonus: float = 0.3 # 30% bonus per repeat occurrence

def get_risk_label(score: float) -> str:
    for (low, high), label in RISK_LABELS.items():
        if low <= score < high:
            return label
    return "Critical"

def build_explanation(behaviour: str, score_before: float, score_after: float,
                      confidence: float, repeat_count: int, duration_bonus: float) -> str:
    """
    Build a human-readable explanation of why the risk score changed.
    """
    delta = score_after - score_before
    direction = "increased" if delta > 0 else "decreased"
    parts = [
        f"Score {direction} by {abs(delta):.2f} points due to '{behaviour}'",
        f"(confidence: {confidence:.2f})"
    ]
    if repeat_count > 1:
        parts.append(f"with {repeat_count}x repetition bonus")
    if duration_bonus > 0:
        parts.append(f"and temporal duration weight x{DURATION_MULTIPLIER.get(behaviour, 1.0):.1f}")
    parts.append(f"→ Risk Level: {get_risk_label(score_after)}")
    return " ".join(parts)

class StudentRiskTracker:
    """
    Maintains rolling risk score state per student.
    Tracks score history, explanation history, and per-behaviour frequencies.
    """
    def __init__(self, student_id: int, config: RiskScoringConfig):
        self.student_id = student_id
        self.config = config
        self.current_score = 0.0
        self.score_history: List[Dict] = []
        self.behaviour_timestamps: Dict[str, List[float]] = {}
        
    def update(self, behaviour: str, timestamp: float, confidence: float):
        # Track behaviour occurrences in rolling window
        window = self.config.repeat_window_seconds
        if behaviour not in self.behaviour_timestamps:
            self.behaviour_timestamps[behaviour] = []
        self.behaviour_timestamps[behaviour].append(timestamp)
        
        # Filter to rolling window
        self.behaviour_timestamps[behaviour] = [
            t for t in self.behaviour_timestamps[behaviour]
            if timestamp - t <= window
        ]
        repeat_count = len(self.behaviour_timestamps[behaviour])
        
        # Compute base increment
        weight = BEHAVIOUR_WEIGHTS.get(behaviour, 5.0)
        duration_mult = DURATION_MULTIPLIER.get(behaviour, 1.0)
        repeat_mult = 1.0 + (repeat_count - 1) * self.config.repeat_bonus
        increment = weight * confidence * duration_mult * repeat_mult / 10.0  # Normalize to frame-level contribution
        
        score_before = self.current_score
        self.current_score = min(100.0, self.current_score + increment)
        
        explanation = build_explanation(
            behaviour, score_before, self.current_score,
            confidence, repeat_count, duration_mult
        )
        
        self.score_history.append({
            "student_id": self.student_id,
            "timestamp": timestamp,
            "behaviour": behaviour,
            "confidence": confidence,
            "score_before": round(score_before, 2),
            "score_after": round(self.current_score, 2),
            "risk_label": get_risk_label(self.current_score),
            "explanation": explanation
        })
        
    def apply_decay(self, timestamp: float, last_timestamp: float):
        """Gradually reduce risk score when no behaviour is detected."""
        elapsed_frames = max(0, (timestamp - last_timestamp) * self.config.fps)
        decay = elapsed_frames * self.config.decay_rate_per_frame
        self.current_score = max(0.0, self.current_score - decay)

    def get_summary(self) -> Dict:
        if not self.score_history:
            return {
                "student_id": self.student_id,
                "final_score": 0.0,
                "risk_label": "Safe",
                "peak_score": 0.0,
                "total_events": 0,
                "score_history": []
            }
        peak = max(e["score_after"] for e in self.score_history)
        return {
            "student_id": self.student_id,
            "final_score": round(self.current_score, 2),
            "risk_label": get_risk_label(self.current_score),
            "peak_score": round(peak, 2),
            "total_events": len(self.score_history),
            "score_history": self.score_history
        }

def visualize_risk_scores_to_video(risk_summaries: Dict, config: RiskScoringConfig):
    """
    Generates a simple risk score visualization by rendering risk level charts
    per student onto output frames saved as a PNG summary.
    """
    try:
        import cv2
        import numpy as np

        # Frame parameters
        frame_w, frame_h = 900, 600
        background = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        background[:] = (20, 20, 30)  # Dark background

        # Title
        cv2.putText(background, "Exam Monitoring - Student Risk Score Summary",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Color mapping per label
        label_colors = {
            "Safe":     (0, 200, 0),
            "Low":      (0, 200, 200),
            "Medium":   (0, 200, 255),
            "High":     (0, 100, 255),
            "Critical": (0, 0, 255)
        }

        y_start = 80
        bar_height = 28
        bar_max_w = 500
        x_label = 30
        x_bar = 200

        for i, (sid, summary) in enumerate(sorted(risk_summaries.items(), key=lambda x: -x[1]["peak_score"])):
            if i >= 15:  # Limit display to top 15 students
                break
            score = summary["peak_score"]
            label = summary["risk_label"]
            color = label_colors.get(label, (200, 200, 200))

            y = y_start + i * (bar_height + 8)

            # Student label
            cv2.putText(background, f"Student {sid}:",
                        (x_label, y + bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            # Risk bar
            filled_w = int((score / 100.0) * bar_max_w)
            cv2.rectangle(background, (x_bar, y), (x_bar + bar_max_w, y + bar_height), (60, 60, 60), -1)
            if filled_w > 0:
                cv2.rectangle(background, (x_bar, y), (x_bar + filled_w, y + bar_height), color, -1)

            # Score text
            cv2.putText(background, f"{score:.1f} - {label}",
                        (x_bar + filled_w + 8, y + bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Save visualization image
        output_img_path = os.path.join(PROJECT_DIR, "output", "risk_score_visualization.png")
        cv2.imwrite(output_img_path, background)
        logger.info(f"Risk score visualization saved at: {output_img_path}")
    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")

def run_risk_scoring(config: RiskScoringConfig):
    logger.info("Loading detected behaviours database...")
    if not os.path.exists(config.behaviours_json_path):
        logger.error(f"Behaviours file not found at: {config.behaviours_json_path}. Please run Module 5 first.")
        return

    with open(config.behaviours_json_path, "r") as f:
        all_detections = json.load(f)

    # Group detections by student ID
    student_detections: Dict[int, List] = {}
    for det in all_detections:
        sid = det["student_id"]
        if sid not in student_detections:
            student_detections[sid] = []
        student_detections[sid].append(det)

    # Initialize trackers per student
    trackers: Dict[int, StudentRiskTracker] = {
        sid: StudentRiskTracker(sid, config)
        for sid in student_detections
    }

    # Process detections chronologically per student
    for sid, detections in student_detections.items():
        detections.sort(key=lambda x: x["timestamp"])
        last_ts = 0.0
        for det in detections:
            ts = det["timestamp"]
            trackers[sid].apply_decay(ts, last_ts)
            trackers[sid].update(det["behaviour_name"], ts, det["confidence"])
            last_ts = ts
        logger.info(f"Student {sid}: Final Score = {trackers[sid].current_score:.2f} ({get_risk_label(trackers[sid].current_score)})")

    # Build summaries
    risk_summaries = {sid: t.get_summary() for sid, t in trackers.items()}

    # Save risk scores JSON
    logger.info(f"Writing risk scores to: {config.output_risk_path}")
    with open(config.output_risk_path, "w") as f:
        json.dump(risk_summaries, f, indent=4)

    # Visualize risk score bars
    visualize_risk_scores_to_video(risk_summaries, config)

    # Print final leaderboard
    print("\n--- Risk Score Summary (Sorted by Peak Score) ---")
    sorted_students = sorted(risk_summaries.items(), key=lambda x: -x[1]["peak_score"])
    for sid, summary in sorted_students:
        print(f"  Student {sid:>3} | Peak: {summary['peak_score']:>6.2f} | Final: {summary['final_score']:>6.2f} | {summary['risk_label']}")

    total_events = sum(s["total_events"] for s in risk_summaries.values())
    logger.info(f"Module 6 complete. Processed {total_events} events across {len(trackers)} students.")

def main():
    config = RiskScoringConfig()
    run_risk_scoring(config)

if __name__ == "__main__":
    main()
