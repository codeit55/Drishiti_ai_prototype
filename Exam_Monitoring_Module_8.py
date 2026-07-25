#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI-Based Human Exam Behaviour Monitoring System
Module 8: Streamlit Dashboard

Professional interactive dashboard displaying live video replay, student tracking data,
risk scores, alerts, behaviour timelines, event logs, statistics, and heatmaps.
Run with: streamlit run Exam_Monitoring_Module_8.py
"""

import os
import sys
import json
import subprocess
import importlib

# Auto-install Streamlit and dependencies
def ensure_packages(pkgs):
    for pkg in pkgs:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_packages(["streamlit", "pandas", "numpy", "matplotlib", "opencv-python"])

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────── PATHS ───────────────────────────
PROJECT_DIR        = "exam_monitoring_system"
VIDEO_PATH         = r"C:\Users\rishi\Downloads\WhatsApp Video 2026-07-25 at 10.59.18 PM.mp4"
TRACKED_VIDEO_PATH = os.path.join(PROJECT_DIR, "output", "tracked_pose_output.mp4")
BEHAVIOURS_JSON    = os.path.join(PROJECT_DIR, "output", "detected_behaviours.json")
RISK_JSON          = os.path.join(PROJECT_DIR, "output", "risk_scores.json")
FEATURES_JSON      = os.path.join(PROJECT_DIR, "output", "student_features.json")
LOG_CSV            = os.path.join(PROJECT_DIR, "output", "exam_monitoring_log.csv")

# ─────────────────────────── PAGE CONFIG ─────────────────────
st.set_page_config(
    page_title="Drishti AI – Exam Monitoring Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ──────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0e1117; }
    .stMetric { background: #1c2333; border-radius: 10px; padding: 12px; border: 1px solid #2d3748; }
    .alert-critical { background: #450a0a; border-left: 4px solid #ef4444; padding: 10px; border-radius: 6px; margin: 4px 0; }
    .alert-high     { background: #431407; border-left: 4px solid #f97316; padding: 10px; border-radius: 6px; margin: 4px 0; }
    .alert-medium   { background: #422006; border-left: 4px solid #eab308; padding: 10px; border-radius: 6px; margin: 4px 0; }
    .alert-safe     { background: #052e16; border-left: 4px solid #22c55e; padding: 10px; border-radius: 6px; margin: 4px 0; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #93c5fd; border-bottom: 1px solid #2d3748; padding-bottom: 6px; margin-bottom: 12px; }
    .student-card   { background: #1c2333; border-radius: 10px; padding: 14px; border: 1px solid #2d3748; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── DATA LOADERS ────────────────────
@st.cache_data
def load_behaviours():
    if not os.path.exists(BEHAVIOURS_JSON):
        return pd.DataFrame()
    with open(BEHAVIOURS_JSON) as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def load_risk_scores():
    if not os.path.exists(RISK_JSON):
        return {}
    with open(RISK_JSON) as f:
        return json.load(f)

@st.cache_data
def load_features():
    if not os.path.exists(FEATURES_JSON):
        return {}
    with open(FEATURES_JSON) as f:
        return json.load(f)

@st.cache_data
def load_log_csv():
    if not os.path.exists(LOG_CSV):
        return pd.DataFrame()
    return pd.read_csv(LOG_CSV)

@st.cache_data
def get_video_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

@st.cache_data
def get_total_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps

# ─────────────────────────── RISK COLOUR ─────────────────────
def label_color(label: str) -> str:
    return {"Critical": "#ef4444", "High": "#f97316",
            "Medium": "#eab308", "Low": "#3b82f6", "Safe": "#22c55e"}.get(label, "#94a3b8")

def score_to_label(score: float) -> str:
    if score >= 80: return "Critical"
    if score >= 60: return "High"
    if score >= 40: return "Medium"
    if score >= 20: return "Low"
    return "Safe"

# ═══════════════════════════ SIDEBAR ═════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Drishti AI")
    st.caption("Exam Behaviour Monitoring System")
    st.divider()

    df_beh = load_behaviours()
    risk_db = load_risk_scores()
    df_log = load_log_csv()

    all_students = sorted(df_beh["student_id"].unique().tolist()) if not df_beh.empty else []

    st.markdown("### 🎛️ Filters")
    selected_students = st.multiselect("Filter by Student ID", options=all_students, default=all_students[:5] if len(all_students) >= 5 else all_students)

    st.divider()

    all_behaviours = sorted(df_beh["behaviour_name"].unique().tolist()) if not df_beh.empty else []
    selected_behaviours = st.multiselect("Filter by Behaviour", options=all_behaviours, default=all_behaviours)

    st.divider()
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    show_replay = st.checkbox("Show Video Replay", value=True)

# ═══════════════════════════ HEADER ══════════════════════════
st.markdown("# 🎓 Drishti AI – Exam Monitoring Dashboard")
st.markdown("*Real-time behaviour analysis, risk scoring, and anomaly detection*")
st.divider()

# ─────────────────────────── FILTERED DATA ───────────────────
df_filtered = df_beh.copy()
if not df_filtered.empty and selected_students:
    df_filtered = df_filtered[df_filtered["student_id"].isin(selected_students)]
if not df_filtered.empty and selected_behaviours:
    df_filtered = df_filtered[df_filtered["behaviour_name"].isin(selected_behaviours)]
if not df_filtered.empty:
    df_filtered = df_filtered[df_filtered["confidence"] >= min_conf]

# ═══════════════════════════ TOP KPI METRICS ═════════════════
total_events     = len(df_filtered)
total_students   = df_filtered["student_id"].nunique() if not df_filtered.empty else 0
critical_count   = sum(1 for sid, s in risk_db.items() if int(sid) in selected_students and s.get("risk_label") == "Critical")
avg_risk         = np.mean([s["final_score"] for sid, s in risk_db.items() if int(sid) in selected_students]) if risk_db else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("📋 Total Events", f"{total_events:,}")
col2.metric("👥 Students Monitored", total_students)
col3.metric("🚨 Critical Alerts", critical_count)
col4.metric("📊 Avg Risk Score", f"{avg_risk:.1f}/100")

st.divider()

# ═══════════════════════════ VIDEO + RISK PANEL ═══════════════
col_video, col_risk = st.columns([3, 2])

with col_video:
    st.markdown('<div class="section-header">🎬 Video Replay</div>', unsafe_allow_html=True)
    video_source = TRACKED_VIDEO_PATH if os.path.exists(TRACKED_VIDEO_PATH) else VIDEO_PATH
    if show_replay and os.path.exists(video_source):
        total_frames, fps = get_total_frames(video_source)
        frame_idx = st.slider("Seek Frame", 0, max(0, total_frames - 1), 0, key="frame_seeker")
        timestamp_display = frame_idx / fps if fps > 0 else 0.0
        st.caption(f"Timestamp: {timestamp_display:.2f}s  |  Frame: {frame_idx}/{total_frames}")
        frame_img = get_video_frame(video_source, frame_idx)
        if frame_img is not None:
            st.image(frame_img, use_container_width=True, caption="Tracked Output Feed")
        else:
            st.warning("Frame could not be loaded.")

        # Filter events at this frame
        if not df_filtered.empty:
            frame_ts = frame_idx / fps
            nearby = df_filtered[(df_filtered["timestamp"] >= frame_ts - 0.5) &
                                 (df_filtered["timestamp"] <= frame_ts + 0.5)]
            if not nearby.empty:
                st.caption("**Events at this timestamp:**")
                for _, row in nearby.iterrows():
                    st.markdown(f"- 🔴 **Student {row['student_id']}**: `{row['behaviour_name']}` (Conf: {row['confidence']:.2f})")
    else:
        st.info("Enable 'Show Video Replay' to display frames.")

with col_risk:
    st.markdown('<div class="section-header">🏆 Student Risk Scores</div>', unsafe_allow_html=True)
    if risk_db:
        sorted_risk = sorted(
            [(int(sid), s) for sid, s in risk_db.items() if int(sid) in selected_students],
            key=lambda x: -x[1]["peak_score"]
        )
        for sid, summary in sorted_risk:
            label = summary["risk_label"]
            score = summary["peak_score"]
            color = label_color(label)
            pct = score / 100.0
            st.markdown(f"""
            <div class="student-card">
                <strong style="color:{color};">Student {sid}</strong>
                <span style="float:right;color:{color};font-weight:700;">{label}</span><br>
                <div style="background:#2d3748;border-radius:4px;height:8px;margin:6px 0;">
                    <div style="background:{color};width:{pct*100:.1f}%;height:8px;border-radius:4px;"></div>
                </div>
                <small>Peak: {score:.1f} &nbsp;|&nbsp; Events: {summary['total_events']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Risk scores not yet generated. Run Module 6 first.")

st.divider()

# ═══════════════════════════ ALERTS ══════════════════════════
st.markdown('<div class="section-header">🚨 Active Alerts</div>', unsafe_allow_html=True)
if not df_filtered.empty:
    high_risk = df_filtered[df_filtered["confidence"] >= 0.8].sort_values("timestamp", ascending=False).head(20)
    if not high_risk.empty:
        for _, row in high_risk.iterrows():
            rid = row["student_id"]
            label = risk_db.get(str(rid), {}).get("risk_label", "Unknown")
            css_class = f"alert-{label.lower()}" if label.lower() in ["critical","high","medium","safe"] else "alert-safe"
            st.markdown(f"""
            <div class="{css_class}">
                🔔 <strong>Student {rid}</strong> — {row['behaviour_name']}
                &nbsp;|&nbsp; T={row['timestamp']:.2f}s
                &nbsp;|&nbsp; Conf: {row['confidence']:.2f}
                &nbsp;|&nbsp; <span style="color:{label_color(label)}">{label}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No high-confidence alerts detected at current threshold.")
else:
    st.info("No data loaded.")

st.divider()

# ═══════════════════════════ TIMELINE ════════════════════════
st.markdown('<div class="section-header">📅 Behaviour Timeline</div>', unsafe_allow_html=True)
if not df_filtered.empty:
    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0e1117")
    ax.set_facecolor("#1c2333")

    behaviour_names = df_filtered["behaviour_name"].unique()
    cmap = cm.get_cmap("tab10", len(behaviour_names))
    b_to_idx = {b: i for i, b in enumerate(behaviour_names)}

    for _, row in df_filtered.iterrows():
        b_idx = b_to_idx[row["behaviour_name"]]
        ax.scatter(row["timestamp"], row["student_id"],
                   c=[cmap(b_idx)], s=12, alpha=0.7, zorder=2)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(b_to_idx[b]),
                           markersize=7, label=b) for b in behaviour_names]
    ax.legend(handles=handles, loc="upper right", fontsize=7, facecolor="#1c2333", labelcolor="white")
    ax.set_xlabel("Timestamp (s)", color="white")
    ax.set_ylabel("Student ID", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Behaviour Events Over Time", color="white", fontsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()

# ═══════════════════════════ CHARTS ══════════════════════════
col_freq, col_heat = st.columns(2)

with col_freq:
    st.markdown('<div class="section-header">📊 Behaviour Frequency</div>', unsafe_allow_html=True)
    if not df_filtered.empty:
        freq = df_filtered["behaviour_name"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
        ax2.set_facecolor("#1c2333")
        colors = plt.cm.Set2(np.linspace(0, 1, len(freq)))
        bars = ax2.barh(freq.index.tolist(), freq.values.tolist(), color=colors)
        ax2.set_xlabel("Count", color="white")
        ax2.tick_params(colors="white")
        ax2.set_title("Behaviour Frequency", color="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

with col_heat:
    st.markdown('<div class="section-header">🌡️ Student × Behaviour Heatmap</div>', unsafe_allow_html=True)
    if not df_filtered.empty:
        pivot = df_filtered.groupby(["student_id", "behaviour_name"]).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
        ax3.set_facecolor("#1c2333")
        im = ax3.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right", fontsize=7, color="white")
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels([str(s) for s in pivot.index.tolist()], fontsize=8, color="white")
        plt.colorbar(im, ax=ax3, shrink=0.8)
        ax3.set_title("Events per Student × Behaviour", color="white")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

st.divider()

# ═══════════════════════════ EVENT LOGS TABLE ═════════════════
st.markdown('<div class="section-header">📋 Event Logs</div>', unsafe_allow_html=True)
if not df_log.empty:
    log_filtered = df_log.copy()
    if selected_students:
        log_filtered = log_filtered[log_filtered["student_id"].isin(selected_students)]
    if selected_behaviours:
        log_filtered = log_filtered[log_filtered["behaviour_name"].isin(selected_behaviours)]
    log_filtered = log_filtered[log_filtered["confidence"] >= min_conf]

    display_cols = ["log_id", "timestamp_seconds", "frame_number", "student_id",
                    "behaviour_name", "confidence", "risk_score", "risk_label",
                    "phone_detected", "body_rotation", "head_yaw", "head_pitch"]
    display_cols = [c for c in display_cols if c in log_filtered.columns]

    st.dataframe(
        log_filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=300
    )

    csv_bytes = log_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", data=csv_bytes, file_name="filtered_log.csv", mime="text/csv")
else:
    st.info("No log data found. Run Module 7 first.")

st.divider()

# ═══════════════════════════ STATISTICS ══════════════════════
st.markdown('<div class="section-header">📈 Statistics</div>', unsafe_allow_html=True)
if not df_filtered.empty:
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown("**Behaviour Breakdown**")
        freq_df = df_filtered["behaviour_name"].value_counts().reset_index()
        freq_df.columns = ["Behaviour", "Count"]
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    with stat_cols[1]:
        st.markdown("**Confidence Distribution**")
        fig4, ax4 = plt.subplots(figsize=(4, 3), facecolor="#0e1117")
        ax4.set_facecolor("#1c2333")
        ax4.hist(df_filtered["confidence"], bins=20, color="#3b82f6", edgecolor="#1c2333")
        ax4.set_xlabel("Confidence", color="white")
        ax4.set_ylabel("Count", color="white")
        ax4.tick_params(colors="white")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#2d3748")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    with stat_cols[2]:
        st.markdown("**Risk Label Distribution**")
        if risk_db:
            labels = [s["risk_label"] for sid, s in risk_db.items() if int(sid) in selected_students]
            label_counts = pd.Series(labels).value_counts()
            fig5, ax5 = plt.subplots(figsize=(4, 3), facecolor="#0e1117")
            ax5.set_facecolor("#0e1117")
            colors = [label_color(l) for l in label_counts.index]
            ax5.pie(label_counts.values, labels=label_counts.index, colors=colors,
                    autopct='%1.0f%%', textprops={"color": "white"})
            ax5.set_title("Risk Distribution", color="white")
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

st.divider()
st.caption("Drishti AI – Exam Behaviour Monitoring System | Modules 1–7 Pipeline")
