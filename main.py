import base64
import functools
import io
import json
import os
import queue
import re
import socket
import sqlite3
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# Must be the absolute first Streamlit command executed
st.set_page_config(
    page_title="CyberVision",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_IMPORT_ERROR = ""
except Exception as _yolo_import_exc:
    YOLO = None
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = f"{type(_yolo_import_exc).__name__}: {_yolo_import_exc}"

try:
    import yara
    YARA_AVAILABLE = True
except Exception:
    yara = None
    YARA_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    from google.cloud import firestore, storage
    from google.oauth2 import service_account as gcp_service_account
    GCP_AVAILABLE = True
except Exception:
    storage = None
    firestore = None
    gcp_service_account = None
    GCP_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception:
    av = None
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object
    WEBRTC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False


def is_cloud_environment() -> bool:
    """Detect a hosted environment (Streamlit Community Cloud, Render, Cloud Run,
    Heroku, HF Spaces, ...). A hosted server has no physical camera, so
    cv2.VideoCapture can never see the *visitor's* webcam there. In that case the
    app captures the visitor's own browser camera over WebRTC instead.
    """
    force_webrtc = os.environ.get("CYBERVISION_FORCE_WEBRTC", "").strip().lower()
    if force_webrtc in ("1", "true", "yes"):
        return True

    force_local = os.environ.get("CYBERVISION_FORCE_LOCAL_CAMERA", "").strip().lower()
    if force_local in ("1", "true", "yes"):
        return False

    if os.path.exists("/mount/src"):
        return True

    if os.environ.get("HOME") == "/home/appuser":
        return True

    for env_name in ("RENDER", "K_SERVICE", "DYNO", "RAILWAY_ENVIRONMENT", "SPACE_ID", "FLY_APP_NAME"):
        if os.environ.get(env_name):
            return True

    return False


def local_camera_available() -> bool:
    """Best-effort probe for a real, locally-attached camera device."""
    try:
        probe = cv2.VideoCapture(CONFIG["CAMERA_SOURCES"][0])
        opened = probe.isOpened()
        probe.release()
        return opened
    except Exception:
        return False


def _writable_data_dir() -> str:
    """Hosted file systems are often read-only next to the source. Fall back to
    the OS temp dir so the SQLite buffer and pending uploads always work."""
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        probe = os.path.join(base, ".cv_write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return base
    except Exception:
        return tempfile.gettempdir()


_DATA_DIR = _writable_data_dir()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG: Dict[str, Any] = {
    "APP_NAME": "CyberVision",
    "APP_SUBTITLE": "Adaptive Context Optimization at the Edge",
    "VLM_MODEL": "moondream",
    "CAMERA_INDEX": 0,
    "CAMERA_SOURCES": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 3x3 Grid Sources (local mode only)
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 300,
    "JPEG_QUALITY": 70,
    "FRAME_DELAY_SEC": 0.0,
    "ANALYZE_EVERY_N_FRAMES": 12,
    "CPU_THRESHOLD": 95,
    "RAM_THRESHOLD": 90,
    "ALERT_HISTORY_MAX": 100,
    "METRICS_HISTORY_MAX": 150,
    "LATENCY_HISTORY_MAX": 100,
    "TOAST_DISPLAY_SECONDS": 6.0,
    "CLOUD_UPLOAD_MIN_INTERVAL_SEC": 3.0,
    "YARA_RULE_PATH": "hazard_rules.yar",
    # GCP Credentials & Bucket Configuration
    "GCP_BUCKET": "cybervision_history",
    "GCP_PROJECT_ID": "c5c505d599a4735f61c6c6896dd3f47a2c28337b",
    "DB_FILE": os.path.join(_DATA_DIR, "cybervision_buffer.db"),
    "PENDING_UPLOADS_DIR": os.path.join(_DATA_DIR, "pending_gcp_uploads"),
    "YOLO_MODEL_PATH": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "YOLO_dataset.pt",
    ),
    "YOLO_CONFIDENCE": 0.45,
    "YOLO_IMAGE_SIZE": 160,       # Reduced from 320 to accelerate CPU inference
    "YOLO_EVERY_N_FRAMES": 6,     # Run real inference every 6th tick; reuse last result otherwise
    "VISUAL_DETECT_EVERY_N_FRAMES": 3,  # HSV/YCrCb color analysis is CPU-heavy; don't run it every tick
}

SEVERITY_STYLE = {
    "NORMAL": {"score": 0, "icon": ":material/check_circle:", "color": "#30d158"},
    "MEDIUM": {"score": 1, "icon": ":material/warning:", "color": "#ffd60a"},
    "HIGH": {"score": 2, "icon": ":material/error:", "color": "#ff9500"},
    "CRITICAL": {"score": 3, "icon": ":material/dangerous:", "color": "#ff3b30"},
}

VLM_RESULT_QUEUE: queue.Queue = queue.Queue()
YOLO_RESULT_QUEUE: queue.Queue = queue.Queue()


# =============================================================================
# OFFLINE SQLITE BUFFER & GCP FIRESTORE SYNC THREAD
# =============================================================================
def init_db() -> None:
    with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                severity TEXT,
                description TEXT,
                camera_id TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def save_event_locally(timestamp: float, severity: str, description: str, camera_id: str = "CAM_01") -> None:
    with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pending_events (timestamp, severity, description, camera_id, synced)
            VALUES (?, ?, ?, ?, 0)
        """, (timestamp, severity, description, camera_id))
        conn.commit()


_WIFI_CHECK_CACHE: Dict[str, Any] = {"checked_at": 0.0, "online": False}
_WIFI_CHECK_TTL_SECONDS = 15.0


def is_wifi_connected(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    # Cached for a few seconds: this is called on every rerun (sidebar, settings
    # panel, sync thread).
    now = time.time()
    if now - _WIFI_CHECK_CACHE["checked_at"] < _WIFI_CHECK_TTL_SECONDS:
        return bool(_WIFI_CHECK_CACHE["online"])

    try:
        with socket.create_connection((host, port), timeout=timeout):
            online = True
    except OSError:
        online = False

    _WIFI_CHECK_CACHE["checked_at"] = now
    _WIFI_CHECK_CACHE["online"] = online
    return online


def sync_worker_loop() -> None:
    while True:
        if is_wifi_connected() and GCP_AVAILABLE and firestore is not None:
            try:
                _, db_client = get_gcp_clients()
                if db_client is None:
                    raise RuntimeError("GCP credentials not configured")
                with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, timestamp, severity, description, camera_id FROM pending_events WHERE synced = 0")
                    unsynced_rows = cursor.fetchall()

                    for row in unsynced_rows:
                        event_id, ts, sev, desc, cam_id = row
                        doc_ref = db_client.collection("hazard_events").document(f"event_{event_id}")
                        doc_ref.set({
                            "timestamp": ts,
                            "severity": sev,
                            "description": desc,
                            "camera_id": cam_id,
                            "synced_at": firestore.SERVER_TIMESTAMP
                        })
                        cursor.execute("UPDATE pending_events SET synced = 1 WHERE id = ?", (event_id,))
                    conn.commit()
            except Exception:
                pass

            flush_pending_cloud_uploads()
        time.sleep(10)


@st.cache_resource(show_spinner=False)
def _start_sync_thread_once() -> bool:
    # WEB FIX: a hosted app serves many sessions from one process. The old code
    # started a new sync thread per browser session; this starts exactly one.
    init_db()
    threading.Thread(target=sync_worker_loop, daemon=True, name="cybervision-sync").start()
    return True


def start_sync_thread() -> None:
    _start_sync_thread_once()


# =============================================================================
# SESSION STATE & THEME
# =============================================================================
def init_session_state() -> None:
    defaults = {
        "dark_mode": False,
        "sound_enabled": True,
        "camera_running": False,
        "last_frame": None,
        "last_frame_rgb": None,
        "frames_processed": 0,
        "last_analyzed_frame": 0,
        "inference_count": 0,
        "alert_history": deque(maxlen=CONFIG["ALERT_HISTORY_MAX"]),
        "metrics_history": deque(maxlen=CONFIG["METRICS_HISTORY_MAX"]),
        "latency_history": deque(maxlen=CONFIG["LATENCY_HISTORY_MAX"]),
        "system_status": "STANDBY",
        "vlm_context": "READY",
        "last_error": "",
        "last_public_notice": "",
        "cloud_sync_enabled": False,
        "fps": 0.0,
        "last_frame_time": time.time(),
        "latest_detection": None,
        "yolo_severity": "NORMAL",
        "yolo_detector": None,
        "yolo_detection": None,
        "yolo_detection_in_progress": False,
        "last_vlm_candidate_frame": None,
        "latest_vlm_text": "",
        "alarm_active": False,
        "vlm_in_progress": False,
        "last_triggered_alert_ts": 0.0,
        "last_cloud_upload_ts": 0.0,
        "active_hazard_toast": None,
        "active_page": "Dashboard",
        "webrtc_ctx": None,
        "_frame_processing_busy": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "use_webrtc" not in st.session_state:
        needs_browser_camera = is_cloud_environment() or not local_camera_available()
        st.session_state.use_webrtc = needs_browser_camera and WEBRTC_AVAILABLE
        st.session_state.webrtc_unavailable_on_cloud = needs_browser_camera and not WEBRTC_AVAILABLE


def inject_custom_css() -> None:
    # NOTE: intentionally NOT wrapped in @st.cache_data. Caching a function that
    # emits elements relies on cached-element replay, which is fragile; the CSS
    # block is tiny so it is simply re-emitted on each run.
    bg = "#fff7ed"
    panel_warm = "#fff3e6"
    text = "#1f2937"
    muted = "#6b7280"
    accent = "#f97316"
    accent_dark = "#c2410c"

    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {{
            background-color: {bg} !important;
            background:
                radial-gradient(circle at top left, rgba(249,115,22,0.18), transparent 32%),
                linear-gradient(135deg, {bg} 0%, #ffffff 45%, #fff1e6 100%) !important;
            color: {text} !important;
        }}

        /* STABILIZE LIVE VIDEO CONTAINER TO PREVENT DISAPPEARING/BLINKING */
        div[data-testid="stFragment"],
        [data-testid="stFragment"] > div,
        div[data-testid="stElementContainer"],
        div[data-testid="stImage"],
        div[data-testid="stImage"] img,
        [data-stale="true"],
        [data-stale="true"] * {{
            opacity: 1 !important;
            visibility: visible !important;
            transition: none !important;
            filter: none !important;
            animation: none !important;
        }}

        .main .block-container {{
            padding-top: 0.4rem;
            padding-bottom: 1rem;
            max-width: 1600px;
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 0.55rem !important;
        }}

        [data-testid="stAlert"] {{
            padding: 0.5rem 0.9rem !important;
            margin-bottom: 0 !important;
        }}

        div[data-baseweb="select"] {{
            border-radius: 14px !important;
        }}

        div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.95) !important;
            border: none !important;
            color: {accent_dark} !important;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {panel_warm} 100%) !important;
            border-right: none !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {accent_dark} !important;
            font-weight: 900 !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text};
        }}

        .muted {{
            color: {muted};
        }}

        .hero {{
            position: relative;
            padding: 0.45rem 1.1rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,237,213,0.96)),
                radial-gradient(circle at top right, rgba(249,115,22,0.28), transparent 35%);
            border: none;
            border-radius: 20px;
            margin-bottom: 0.4rem;
            box-shadow: 0 10px 26px rgba(249,115,22,0.13);
            overflow: hidden;
        }}

        .hero-title {{
            font-size: 1.55rem;
            font-weight: 900;
            margin: 0.15rem 0 0 0;
            color: {accent_dark};
            letter-spacing: -0.03em;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: rgba(249,115,22,0.12);
            border: none;
            color: {accent_dark};
            font-weight: 900;
            font-size: 0.7rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.94);
            border: none !important;
            border-radius: 12px;
            padding: 0.3rem 0.5rem;
            height: 84px;
            min-height: 84px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 6px 14px rgba(249,115,22,0.08);
        }}

        div[data-testid="stMetricValue"] {{
            color: {accent_dark};
            font-weight: 900;
            font-size: 1.25rem !important;
        }}

        div[data-testid="stMetricDelta"] {{
            background: rgba(107, 114, 128, 0.12) !important;
            border-radius: 999px !important;
            padding: 0.05rem 0.5rem !important;
            width: fit-content;
        }}

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] svg,
        div[data-testid="stMetricDelta"] span {{
            color: {muted} !important;
            fill: {muted} !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 14px;
            font-weight: 800;
            border: 1px solid #fb923c;
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            color: {accent_dark};
            box-shadow: 0 8px 18px rgba(249,115,22,0.12);
            transition: all 0.18s ease-in-out;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: linear-gradient(135deg, #fed7aa, #fdba74);
            color: #7c2d12;
            border-color: {accent};
            transform: translateY(-1px);
        }}

        .hazard-toast {{
            position: fixed;
            top: 4.5rem;
            right: 1.1rem;
            z-index: 1000000;
            background: #ffffff;
            border: none;
            border-left: 7px solid {accent};
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
            padding: 1.1rem 1.3rem;
            width: auto;
            min-width: 340px;
            max-width: 420px;
        }}

        .hazard-toast .toast-severity {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 900;
            font-size: 1.15rem;
            color: #1f2937;
        }}

        .hazard-toast .toast-title {{
            font-weight: 700;
            font-size: 0.95rem;
            color: #1f2937;
            margin-top: 0.1rem;
        }}

        .hazard-toast .toast-line {{
            font-weight: 600;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.4rem;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: break-word;
        }}

        .hazard-toast .toast-detection {{
            color: #2563eb;
        }}

        .hazard-toast .toast-action {{
            color: #1f2937;
            font-weight: 800;
        }}

        .hazard-toast .toast-meta {{
            color: {muted};
            font-size: 0.8rem;
            margin-top: 0.45rem;
        }}

        .stTabs {{
            margin-top: -0.3rem !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.2rem !important;
        }}

        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 0.4rem !important;
        }}

        [data-testid="stSidebarContent"] {{
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }}

        [data-testid="stSidebar"] div:has(.st-key-sidebar_bottom_block):not(.st-key-sidebar_bottom_block) {{
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
        }}

        [data-testid="stSidebarUserContent"] {{
            padding-bottom: 0.6rem !important;
        }}

        .st-key-sidebar_bottom_block {{
            margin-top: auto !important;
            flex: 0 0 auto !important;
            position: sticky !important;
            bottom: 0 !important;
            background: {panel_warm} !important;
            padding-top: 0.3rem;
            z-index: 5;
        }}

        .brand-block {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.2rem 0 1.1rem 0;
        }}

        .brand-icon {{
            font-size: 1.7rem;
            line-height: 1;
        }}

        .brand-name {{
            font-weight: 900;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            color: {accent_dark};
            line-height: 1.1;
        }}

        .brand-subtitle {{
            font-size: 0.65rem;
            color: {muted};
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .nav-section-label {{
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {muted};
            margin: 0.6rem 0 0.5rem 0.1rem;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            justify-content: flex-start !important;
            text-align: left !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: 700 !important;
            color: {text} !important;
            padding: 0.5rem 0.7rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button > div {{
            display: flex !important;
            width: 100% !important;
            justify-content: space-between !important;
            align-items: center !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(249,115,22,0.10) !important;
            color: {accent_dark} !important;
            transform: none !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {accent}, {accent_dark}) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 8px 18px rgba(249,115,22,0.30) !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {accent_dark}, #9a3412) !important;
            color: #ffffff !important;
        }}

        .sidebar-footer-divider {{
            height: 1px;
            background: rgba(249,115,22,0.18);
            margin: 1.2rem 0 0.9rem 0;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricValue"],
        .st-key-ram_metric_high [data-testid="stMetricValue"],
        .st-key-cpu_metric_high [data-testid="stMetricDelta"],
        .st-key-ram_metric_high [data-testid="stMetricDelta"] {{
            color: #ff3b30 !important;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_high [data-testid="stMetricDelta"] svg {{
            fill: #ff3b30 !important;
        }}

        .st-key-severity_metric_normal [data-testid="stMetricValue"],
        .st-key-severity_metric_normal [data-testid="stMetricValue"] * {{
            color: #16a34a !important;
        }}
        .st-key-severity_metric_medium [data-testid="stMetricValue"],
        .st-key-severity_metric_medium [data-testid="stMetricValue"] * {{
            color: #eab308 !important;
        }}
        .st-key-severity_metric_high [data-testid="stMetricValue"],
        .st-key-severity_metric_high [data-testid="stMetricValue"] *,
        .st-key-severity_metric_critical [data-testid="stMetricValue"],
        .st-key-severity_metric_critical [data-testid="stMetricValue"] * {{
            color: #dc2626 !important;
        }}

        .status-footer {{
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 1rem;
        }}

        .status-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: {muted};
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
            box-shadow: 0 0 0 3px rgba(0,0,0,0.03);
        }}

        .profile-card {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.55rem 0.6rem;
            border-radius: 14px;
            background: rgba(249,115,22,0.08);
            border: none;
            margin-bottom: 0.5rem;
        }}

        .profile-avatar {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            object-fit: cover;
            flex-shrink: 0;
        }}

        .profile-avatar-fallback {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, {accent}, {accent_dark});
            color: #ffffff;
            font-weight: 800;
            font-size: 0.9rem;
        }}

        .profile-text {{
            min-width: 0;
        }}

        .profile-name {{
            font-weight: 800;
            font-size: 0.82rem;
            color: {text};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .profile-email {{
            font-size: 0.68rem;
            color: {muted};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .dash-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.35rem 0.1rem 0.9rem 0.1rem;
            flex-wrap: wrap;
            gap: 0.6rem;
        }}

        .dash-header-title {{
            font-weight: 900;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            color: {text};
            line-height: 1.15;
        }}

        .dash-header-subtitle {{
            font-size: 0.85rem;
            color: {muted};
            font-weight: 500;
        }}

        .status-pill-live {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            border: none;
            font-weight: 800;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .status-pill-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }}

        /* Panels (video, stats, settings): no outline, no box, like backup.py */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent !important;
            border: none !important;
            outline: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }}

        [data-testid="stExpander"],
        [data-testid="stExpander"] details {{
            border: none !important;
            outline: none !important;
        }}

        .stats-panel-title {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {muted};
            margin-bottom: 0.4rem;
        }}

        .video-status-bar {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 0.5rem 0.2rem 0.1rem 0.2rem;
            font-size: 0.8rem;
        }}

        .video-status-bar b {{
            color: {accent_dark};
        }}

        /* Browser-camera (WebRTC) connector: kept VISIBLE and compact. Hiding it
           (0px height / opacity 0) stops browsers from showing the camera
           permission prompt, which is why the feed never appeared. */
        .st-key-webrtc_transport {{
            min-height: 3rem;
            overflow: visible !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# AUDIO & NOTIFICATION DISPATCHER
# =============================================================================
@functools.lru_cache(maxsize=4)
def build_beep_wav_base64(duration: float = 0.45, freq: int = 1100, sample_rate: int = 44100) -> str:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    envelope = np.linspace(1, 0.15, tone.size)
    audio = (tone * envelope * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@functools.lru_cache(maxsize=4)
def build_siren_wav_base64(
    duration: float = 1.6,
    low_freq: int = 900,
    high_freq: int = 1500,
    sample_rate: int = 44100,
) -> str:
    # lru_cache: this used to be rebuilt (numpy + base64 of ~280 KB) on every
    # 100 ms tick while CRITICAL.
    segment_len = max(1, int(sample_rate * duration / 4))
    t_seg = np.linspace(0, duration / 4, segment_len, False)

    def tone(freq: int) -> np.ndarray:
        return np.sin(freq * t_seg * 2 * np.pi)

    wave_data = np.concatenate([tone(high_freq), tone(low_freq), tone(high_freq), tone(low_freq)])
    audio = (wave_data * 32767 * 0.9).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def play_critical_alarm(is_critical: bool) -> None:
    """Plays continuous siren sound while severity remains CRITICAL."""
    st.session_state.alarm_active = bool(is_critical) and st.session_state.sound_enabled

    if not st.session_state.alarm_active:
        return

    siren_b64 = build_siren_wav_base64()
    st.markdown(
        f"""
        <audio autoplay loop style="display:none;">
            <source src="data:audio/wav;base64,{siren_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def dispatch_hazard_alerts(severity_str: str, hazard_title: str) -> None:
    severity_norm = str(severity_str).strip().upper()

    if severity_norm not in ["MEDIUM", "HIGH", "CRITICAL"]:
        return

    now = time.time()
    if now - st.session_state.get("last_triggered_alert_ts", 0) <= 3.0:
        return

    st.session_state.last_triggered_alert_ts = now
    hazard_norm = str(hazard_title).strip().upper()

    if "FIRE" in hazard_norm:
        alert_title = "FIRE DETECTED"
        hazard_word = "fire"
    else:
        alert_title = "SMOKE DETECTED"
        hazard_word = "smoke"

    st.session_state.active_hazard_toast = {
        "severity": severity_norm,
        "alert_title": alert_title,
        "detection_line": f"The system detected clear signs of {hazard_word}.",
        "action_line": "Please evacuate from the area immediately.",
        "checked_time": datetime.now().strftime("%H:%M:%S"),
        "triggered_at": now,
        # CRITICAL uses the looping siren instead of the one-shot beep.
        "beep": severity_norm != "CRITICAL",
    }


def render_custom_hazard_toast() -> None:
    toast_data = st.session_state.get("active_hazard_toast")
    if not toast_data:
        return

    if time.time() - toast_data["triggered_at"] > CONFIG["TOAST_DISPLAY_SECONDS"]:
        return

    # WEB FIX: the beep used to be emitted once, inside a fragment that re-runs
    # every 100 ms, so the <audio> element was removed before it finished
    # playing. It is now rendered for as long as the toast is visible; the unique
    # id makes the browser re-mount (and so replay) it for each new alert.
    audio_html = ""
    if toast_data.get("beep") and st.session_state.sound_enabled:
        beep_id = int(toast_data["triggered_at"] * 1000)
        audio_html = (
            f'<audio id="beep-{beep_id}" autoplay style="display:none;">'
            f'<source src="data:audio/wav;base64,{build_beep_wav_base64()}" type="audio/wav">'
            f"</audio>"
        )

    st.markdown(
        f"""
        <div class="hazard-toast">
            <div class="toast-title">{toast_data['alert_title']}</div>
            <div class="toast-line toast-detection">{toast_data['detection_line']}</div>
            <div class="toast-line toast-action">{toast_data['action_line']}</div>
            <div class="toast-meta">Last checked: {toast_data['checked_time']}</div>
            {audio_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SYSTEM MONITORING
# =============================================================================
class SystemMonitor:
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        try:
            cpu = float(psutil.cpu_percent(interval=0.01))
            ram = float(psutil.virtual_memory().percent)

            metrics = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": cpu,
                "ram": ram,
            }

            st.session_state.metrics_history.append(metrics)
            return metrics
        except Exception:
            return {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": 0.0,
                "ram": 0.0,
            }


class ResourceGovernor:
    @staticmethod
    def check_resource_pressure() -> None:
        metrics = SystemMonitor.get_metrics()
        cpu = metrics["cpu"]
        ram = metrics["ram"]

        if cpu >= CONFIG["CPU_THRESHOLD"] or ram >= CONFIG["RAM_THRESHOLD"]:
            st.session_state.system_status = "RESOURCE_PRESSURE"
            st.session_state.vlm_context = "REDUCED"
            st.session_state.last_public_notice = (
                f"Resource pressure detected. CPU={cpu:.1f}%, RAM={ram:.1f}%. "
                "Using reduced context."
            )
        else:
            st.session_state.system_status = "HEALTHY"
            st.session_state.vlm_context = "ACTIVE"
            st.session_state.last_public_notice = ""

    @staticmethod
    def recommended_token_budget() -> int:
        metrics = SystemMonitor.get_metrics()
        worst = max(metrics["cpu"], metrics["ram"])

        if worst >= 85:
            return 128
        if worst >= 70:
            return 256
        if worst >= 50:
            return 512

        return 1024


# =============================================================================
# BROWSER CAMERA (WEBRTC)
# =============================================================================
if WEBRTC_AVAILABLE:
    class BrowserCameraProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._latest_frame: Optional[np.ndarray] = None
            self._overlay: Optional[Dict[str, Any]] = None

        def set_overlay(self, yolo_result: Dict[str, Any], src_w: int, src_h: int, hud: Tuple[str, str]) -> None:
            with self._lock:
                self._overlay = {"yolo": yolo_result, "w": src_w, "h": src_h, "hud": hud}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                self._latest_frame = img
                overlay = self._overlay

            out = img
            if overlay is not None:
                try:
                    out = self._draw_overlay(img, overlay)
                except Exception:
                    out = img
            return av.VideoFrame.from_ndarray(out, format="bgr24")

        @staticmethod
        def _draw_overlay(img: np.ndarray, overlay: Dict[str, Any]) -> np.ndarray:
            # Detections are computed on a resized copy (overlay["w"] x overlay["h"]);
            # scale them back onto the full-size frame the browser receives.
            out = img.copy()
            h, w = out.shape[:2]
            sx = w / max(overlay["w"], 1)
            sy = h / max(overlay["h"], 1)

            for det in (overlay["yolo"] or {}).get("detections", []):
                x1, y1, x2, y2 = det["box"]
                p1 = (int(x1 * sx), int(y1 * sy))
                p2 = (int(x2 * sx), int(y2 * sy))
                label = f"{det['class']} {det['confidence']:.0%}"
                cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
                cv2.putText(out, label, (p1[0], max(p1[1] - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            shade = out.copy()
            cv2.rectangle(shade, (10, 10), (430, 82), (0, 0, 0), -1)
            cv2.addWeighted(shade, 0.50, out, 0.50, 0, out)
            cv2.putText(out, overlay["hud"][0], (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
            cv2.putText(out, overlay["hud"][1], (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
            return out

        def get_latest_frame(self) -> Optional[np.ndarray]:
            with self._lock:
                return None if self._latest_frame is None else self._latest_frame.copy()
else:
    BrowserCameraProcessor = None


def _get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _get_cloudflare_turn_credentials() -> Tuple[Optional[str], Optional[str]]:
    key_id = _get_secret("CLOUDFLARE_TURN_KEY_ID")
    api_token = _get_secret("CLOUDFLARE_TURN_KEY_API_TOKEN")
    return key_id, api_token


def _get_metered_credentials() -> Tuple[Optional[str], Optional[str]]:
    api_key = _get_secret("METERED_API_KEY")
    domain = _get_secret("METERED_DOMAIN")
    return api_key, domain


# =============================================================================
# GOOGLE CLOUD CREDENTIALS
# =============================================================================
def _get_gcp_service_account_info() -> Optional[dict]:
    raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass

    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    local_key_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "service_account.json"
    )
    if os.path.isfile(local_key_path):
        try:
            with open(local_key_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return None


@st.cache_resource(show_spinner=False)
def get_gcp_clients() -> Tuple[Optional[Any], Optional[Any]]:
    if not GCP_AVAILABLE:
        return None, None

    info = _get_gcp_service_account_info()
    project_id = CONFIG["GCP_PROJECT_ID"]

    try:
        if info and gcp_service_account is not None:
            credentials = gcp_service_account.Credentials.from_service_account_info(info)
            project_id = info.get("project_id", project_id)
            storage_client = storage.Client(credentials=credentials, project=project_id)
            db_client = firestore.Client(credentials=credentials, project=project_id)
        else:
            storage_client = storage.Client(project=project_id)
            db_client = firestore.Client(project=project_id)
        return storage_client, db_client
    except Exception:
        return None, None


STUN_ONLY_ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
ICE_SERVER_DIAGNOSTICS: Dict[str, str] = {"tier": "unknown", "detail": ""}


def _fetch_cloudflare_ice_servers(key_id: str, api_token: str) -> Optional[list]:
    if not REQUESTS_AVAILABLE:
        ICE_SERVER_DIAGNOSTICS["detail"] = "`requests` package is not installed/importable."
        return None
    try:
        resp = requests.post(
            f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={"ttl": 86400},
            timeout=5,
        )
        resp.raise_for_status()
        ice_servers = resp.json().get("iceServers") or []
        port_53_pattern = re.compile(r":53(?:\?|$)")
        filtered = []
        for server in ice_servers:
            urls = server.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            urls = [u for u in urls if not port_53_pattern.search(u)]
            if urls:
                item = dict(server)
                item["urls"] = urls
                filtered.append(item)
        if not filtered:
            ICE_SERVER_DIAGNOSTICS["detail"] = (
                "Cloudflare returned iceServers but every entry was filtered out."
            )
            return None
        return filtered
    except Exception as exc:
        ICE_SERVER_DIAGNOSTICS["detail"] = f"Cloudflare TURN request failed: {exc}"
        return None


def _fetch_metered_ice_servers(api_key: str, domain: str) -> Optional[list]:
    if not REQUESTS_AVAILABLE:
        ICE_SERVER_DIAGNOSTICS["detail"] = "`requests` package is not installed/importable."
        return None
    try:
        resp = requests.get(
            f"https://{domain}/api/v1/turn/credentials",
            params={"apiKey": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        ice_servers = resp.json()
        return ice_servers or None
    except Exception as exc:
        ICE_SERVER_DIAGNOSTICS["detail"] = f"Metered TURN request failed: {exc}"
        return None


def get_ice_servers() -> list:
    cache_key = "_ice_servers_cache"
    diag_key = "_ice_server_diagnostics_cache"
    ttl_seconds = 3000

    if diag_key in st.session_state:
        ICE_SERVER_DIAGNOSTICS.update(st.session_state[diag_key])

    cached = st.session_state.get(cache_key)
    if cached and (time.time() - cached["fetched_at"]) < ttl_seconds:
        return cached["servers"]

    cf_key_id, cf_api_token = _get_cloudflare_turn_credentials()
    ice_servers = None
    if cf_key_id and cf_api_token:
        ice_servers = _fetch_cloudflare_ice_servers(cf_key_id, cf_api_token)
        if ice_servers:
            ICE_SERVER_DIAGNOSTICS.update(tier="cloudflare", detail="TURN credentials minted OK.")
    else:
        ICE_SERVER_DIAGNOSTICS["detail"] = (
            "CLOUDFLARE_TURN_KEY_ID / CLOUDFLARE_TURN_KEY_API_TOKEN not set "
            "(checked env vars and st.secrets)."
        )

    if not ice_servers:
        metered_key, metered_domain = _get_metered_credentials()
        if metered_key and metered_domain:
            ice_servers = _fetch_metered_ice_servers(metered_key, metered_domain)
            if ice_servers:
                ICE_SERVER_DIAGNOSTICS.update(tier="metered", detail="TURN credentials fetched OK.")

    if not ice_servers:
        ICE_SERVER_DIAGNOSTICS["tier"] = "stun-only"
        ice_servers = STUN_ONLY_ICE_SERVERS

    st.session_state[cache_key] = {"servers": ice_servers, "fetched_at": time.time()}
    st.session_state[diag_key] = dict(ICE_SERVER_DIAGNOSTICS)
    return ice_servers


def _describe_ice_servers(ice_servers: list) -> str:
    urls: list = []
    has_turn_credential = False
    for server in ice_servers:
        server_urls = server.get("urls", [])
        if isinstance(server_urls, str):
            server_urls = [server_urls]
        urls.extend(server_urls)
        if server.get("credential"):
            has_turn_credential = True

    if any("cloudflare.com" in u for u in urls) and has_turn_credential:
        return ":material/check_circle: Browser camera relay: Cloudflare TURN configured."
    if has_turn_credential:
        return ":material/check_circle: Browser camera relay: TURN relay configured."
    return ":material/info: Browser camera relay: STUN only (may fail behind strict networks)."


def render_browser_camera_widget() -> None:
    """Mounts the WebRTC connector that streams the VISITOR's webcam to the server.

    Fixes for "camera doesn't display when started":
      * SENDRECV instead of SENDONLY. In SENDONLY mode streamlit-webrtc does not
        run the video processor, so BrowserCameraProcessor.recv() was never called
        and the app waited forever for a first frame.
      * The widget is visible (it used to be collapsed to 0px with opacity 0, which
        stops browsers from showing the camera permission prompt).
      * Removed the unsupported `media_toggle_controls` argument.
    """
    if not WEBRTC_AVAILABLE:
        st.error(
            "Browser camera mode requires the `streamlit-webrtc` and `av` packages, "
            "which are listed in requirements.txt but failed to import. Check the "
            "deployment logs — this is usually a missing system library (see "
            "packages.txt: libavformat-dev, libavdevice-dev, libgl1)."
        )
        return

    ice_servers = get_ice_servers()

    with st.container(key="webrtc_transport"):
        ctx = webrtc_streamer(
            key="cybervision-browser-camera",
            mode=WebRtcMode.SENDRECV,
            desired_playing_state=True,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},
                },
                "audio": False,
            },
            video_processor_factory=BrowserCameraProcessor,
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
                "style": {"width": "100%", "borderRadius": "12px"},
            },
            async_processing=True,
        )

    st.session_state.webrtc_ctx = ctx

    state = getattr(ctx, "state", None)
    if state is not None and getattr(state, "playing", False):
        st.caption(":material/videocam: Browser camera connected.")
    else:
        st.caption(
            ":material/videocam: Allow camera access when your browser asks. "
            "If no prompt appears, press START above."
        )
    ice_note = _describe_ice_servers(ice_servers)
    if ice_note.startswith(":material/info:") and not (state is not None and getattr(state, "playing", False)):
        st.warning(
            "No TURN relay is configured, so a hosted app usually cannot connect to your "
            "browser camera. Add CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN "
            "(or METERED_API_KEY + METERED_DOMAIN) to your Streamlit secrets and reboot the app.",
            icon=":material/warning:",
        )
    st.caption(ice_note)


# =============================================================================
# MULTI-CAMERA HANDLING & 3x3 MATRIX (LOCAL DEVICES ONLY)
# =============================================================================
@st.cache_resource(show_spinner=False)
def _get_camera_read_executor(n_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max(1, n_workers))


@st.cache_resource(show_spinner=False)
def get_camera_caps() -> Dict[int, cv2.VideoCapture]:
    caps = {}

    def _init_cam(args):
        idx, src = args
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return idx, cap
        else:
            cap.release()
            return idx, None

    with ThreadPoolExecutor(max_workers=len(CONFIG["CAMERA_SOURCES"])) as executor:
        results = executor.map(_init_cam, enumerate(CONFIG["CAMERA_SOURCES"]))
        for idx, cap in results:
            if cap is not None:
                caps[idx] = cap
    return caps


def release_camera() -> None:
    if st.session_state.get("use_webrtc"):
        return
    try:
        caps = get_camera_caps()
        for cap in caps.values():
            if cap and cap.isOpened():
                cap.release()
        get_camera_caps.clear()
    except Exception:
        pass


def create_blank_tile(width: int = 320, height: int = 240, label: str = "NO CAMERA DETECTED") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(frame, label, (text_x, text_y), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (40, 40, 40), 1)
    return frame


_BLANK_TILE_CACHE: Dict[Tuple[int, int, str], np.ndarray] = {}


def get_cached_blank_tile(width: int, height: int, label: str) -> np.ndarray:
    key = (width, height, label)
    tile = _BLANK_TILE_CACHE.get(key)
    if tile is None:
        tile = create_blank_tile(width, height, label)
        _BLANK_TILE_CACHE[key] = tile
    return tile.copy()


def construct_3x3_grid(active_frames: Dict[int, np.ndarray], tile_w: int = 320, tile_h: int = 240) -> np.ndarray:
    tiles = []
    for idx in range(9):
        cam_key = f"CAM_{idx+1:02d}"
        if idx in active_frames and active_frames[idx] is not None:
            tile = cv2.resize(active_frames[idx], (tile_w, tile_h))
            cv2.putText(tile, f"{cam_key} [LIVE]", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (0, 255, 0), 1)
        else:
            tile = get_cached_blank_tile(tile_w, tile_h, f"{cam_key}: NO CAMERA DETECTED")
        tiles.append(tile)

    row1 = np.hstack([tiles[0], tiles[1], tiles[2]])
    row2 = np.hstack([tiles[3], tiles[4], tiles[5]])
    row3 = np.hstack([tiles[6], tiles[7], tiles[8]])

    return np.vstack([row1, row2, row3])


class VideoCaptureManager:
    @staticmethod
    def capture_active_frames() -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        if st.session_state.get("use_webrtc"):
            return VideoCaptureManager._capture_from_browser()
        return VideoCaptureManager._capture_from_local_devices()

    @staticmethod
    def _capture_from_browser() -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        ctx = st.session_state.get("webrtc_ctx")
        if ctx is None or ctx.video_processor is None:
            st.session_state.last_error = "Waiting for browser camera permission..."
            return {}, None

        frame = ctx.video_processor.get_latest_frame()
        if frame is None:
            st.session_state.last_error = "Waiting for first frame from browser camera..."
            return {}, None

        # Keep the camera's real aspect ratio (the old fixed 640x300 resize
        # squashed 4:3 webcams).
        h, w = frame.shape[:2]
        target_w = CONFIG["FRAME_WIDTH"]
        target_h = max(1, int(round(h * target_w / max(w, 1))))
        primary_frame = cv2.resize(frame, (target_w, target_h))

        if str(st.session_state.get("last_error", "")).startswith("Waiting for"):
            st.session_state.last_error = ""

        return {0: primary_frame}, primary_frame

    @staticmethod
    def _capture_from_local_devices() -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        caps = get_camera_caps()
        active_frames = {}
        primary_frame = None

        if not caps:
            st.session_state.last_error = "No camera streams open."
            return {}, None

        def _read_cam(item):
            idx, cap = item
            if cap and cap.isOpened():
                cap.grab()
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    return idx, frame
            return idx, None

        if len(caps) == 1:
            results = [_read_cam(item) for item in caps.items()]
        else:
            executor = _get_camera_read_executor(len(caps))
            results = list(executor.map(_read_cam, caps.items()))

        for idx, frame in results:
            if frame is not None:
                active_frames[idx] = frame
                if primary_frame is None:
                    primary_frame = cv2.resize(frame, (CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]))

        return active_frames, primary_frame

    @staticmethod
    def placeholder_frame(message: str, subtext: str = "") -> np.ndarray:
        frame = np.zeros(
            (CONFIG["FRAME_HEIGHT"], CONFIG["FRAME_WIDTH"], 3),
            dtype=np.uint8
        )
        frame[:] = (22, 28, 55)

        cv2.putText(
            frame,
            message,
            (55, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 170),
            2,
        )

        if subtext:
            cv2.putText(
                frame,
                subtext,
                (55, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# =============================================================================
# IMAGE ENCODING
# =============================================================================
def frame_to_base64_jpeg(frame: np.ndarray) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG["JPEG_QUALITY"]]
    ok, buffer = cv2.imencode(".jpg", frame, encode_param)

    if not ok:
        raise ValueError("Failed to encode frame.")

    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# YOLO FIRE & SMOKE DETECTION
# =============================================================================
def get_yolo_debug_info() -> Dict[str, Any]:
    model_path = CONFIG["YOLO_MODEL_PATH"]
    model_dir = os.path.dirname(model_path)

    try:
        dir_listing = os.listdir(model_dir) if os.path.isdir(model_dir) else []
    except Exception as exc:
        dir_listing = [f"<could not list directory: {exc}>"]

    info = {
        "YOLO_AVAILABLE (ultralytics imported)": YOLO_AVAILABLE,
        "ultralytics import error": YOLO_IMPORT_ERROR or "(none)",
        "cwd": os.getcwd(),
        "expected model path": model_path,
        "os.path.exists(model_path)": os.path.exists(model_path),
        "os.path.isfile(model_path)": os.path.isfile(model_path),
        "files in that directory": dir_listing,
        "load_yolo_model() result": None,
        "load exception": "(not attempted)",
    }

    if YOLO_AVAILABLE and os.path.exists(model_path):
        try:
            model = load_yolo_model()
            info["load_yolo_model() result"] = "loaded OK" if model is not None else "returned None"
            info["load exception"] = _yolo_state()["error"] or "(none)"
        except Exception as exc:
            info["load_yolo_model() result"] = "raised an exception"
            info["load exception"] = f"{type(exc).__name__}: {exc}"

    return info


@st.cache_resource(show_spinner=False)
def _yolo_state() -> Dict[str, Any]:
    return {"error": ""}


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    state = _yolo_state()

    if not YOLO_AVAILABLE:
        state["error"] = (
            "The 'ultralytics' package failed to import"
            + (f" ({YOLO_IMPORT_ERROR})" if YOLO_IMPORT_ERROR else "")
            + ". Check it's in requirements.txt and installed in this environment."
        )
        return None

    model_path = CONFIG["YOLO_MODEL_PATH"]
    if not os.path.exists(model_path):
        state["error"] = f"YOLO model file not found at: {model_path}"
        return None

    try:
        model = YOLO(model_path)
        state["error"] = ""
        return model
    except Exception as exc:
        state["error"] = f"YOLO failed to load ({type(exc).__name__}): {exc}"
        return None


def get_yolo_load_error() -> str:
    load_yolo_model()
    return _yolo_state()["error"]


def is_yolo_model_available() -> bool:
    if not YOLO_AVAILABLE:
        return False
    model_path = CONFIG["YOLO_MODEL_PATH"]
    if not os.path.isfile(model_path):
        return False
    return load_yolo_model() is not None


class YOLOFireSmokeDetector:
    def __init__(self) -> None:
        self.model = load_yolo_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        empty = {
            "detected": False,
            "detections": [],
            "highest_confidence": 0.0,
            "classes": [],
            "annotated_frame": frame,
        }

        if frame is None or self.model is None:
            return empty

        try:
            results = self.model(
                frame,
                imgsz=CONFIG["YOLO_IMAGE_SIZE"],
                conf=CONFIG["YOLO_CONFIDENCE"],
                verbose=False,
                device="cpu",
            )

            if not results:
                return empty

            result = results[0]
            detections = []
            names = result.names if hasattr(result, "names") else self.model.names

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    if isinstance(names, dict):
                        class_name = str(names.get(cls_id, cls_id)).lower()
                    else:
                        class_name = str(names[cls_id]).lower()

                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": [int(v) for v in xyxy],
                    })

            annotated = result.plot() if detections else frame

            return {
                "detected": bool(detections),
                "detections": detections,
                "highest_confidence": (
                    max(d["confidence"] for d in detections)
                    if detections else 0.0
                ),
                "classes": sorted(set(d["class"] for d in detections)),
                "annotated_frame": annotated,
            }

        except Exception as exc:
            empty["error"] = f"YOLO detection issue: {exc}"
            return empty

    def _async_worker(self, frame: np.ndarray) -> None:
        result = self.detect(frame)
        YOLO_RESULT_QUEUE.put(result)

    def trigger_async_detect(self, frame: np.ndarray) -> None:
        if st.session_state.get("yolo_detection_in_progress", False):
            return
        st.session_state.yolo_detection_in_progress = True
        thread = threading.Thread(
            target=self._async_worker,
            args=(frame.copy(),),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def drain_worker_queue() -> None:
        while not YOLO_RESULT_QUEUE.empty():
            try:
                result = YOLO_RESULT_QUEUE.get_nowait()
            except queue.Empty:
                break
            st.session_state.yolo_detection_in_progress = False
            st.session_state.yolo_detection = result
            st.session_state.inference_count += 1
            err = result.get("error")
            if err:
                st.session_state.last_error = err

    @staticmethod
    def draw_boxes_on_frame(frame: np.ndarray, yolo_result: Dict[str, Any]) -> np.ndarray:
        if not yolo_result or not yolo_result.get("detections"):
            return frame

        annotated = frame.copy()
        for det in yolo_result["detections"]:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class']} {det['confidence']:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated

    @staticmethod
    def hazard_severity(yolo_result: Dict[str, Any]) -> str:
        if not yolo_result.get("detected"):
            return "NORMAL"

        highest = "NORMAL"
        for detection in yolo_result.get("detections", []):
            class_name = detection["class"].lower()
            confidence = detection["confidence"]

            if "fire" in class_name:
                candidate = "CRITICAL" if confidence >= 0.50 else "HIGH"
            elif "smoke" in class_name:
                candidate = "HIGH" if confidence >= 0.30 else "MEDIUM"
            else:
                continue

            if SEVERITY_STYLE[candidate]["score"] > SEVERITY_STYLE[highest]["score"]:
                highest = candidate

        return highest


# =============================================================================
# VISUAL FALLBACK DETECTION
# =============================================================================
class VisualFireSmokeDetector:
    @staticmethod
    def detect(frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {
                "visual_severity": "NORMAL",
                "visual_keyword": "",
                "fire_ratio": 0.0,
                "smoke_ratio": 0.0,
            }

        bgr = frame
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        b, g, r = cv2.split(bgr)
        h, s, v = cv2.split(hsv)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        red_or_orange_hue = ((h <= 18) | (h >= 172))
        bright_saturated = (s >= 160) & (v >= 200)
        rgb_dominance = (r > 190) & (r > g * 1.25) & (r > b * 1.85) & (g > 60)
        non_skin = skin_mask == 0

        fire_pixels = red_or_orange_hue & bright_saturated & rgb_dominance & non_skin
        fire_mask = fire_pixels.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

        low_saturation = s < 35
        mid_brightness = (v > 110) & (v < 200)
        gray_dominance = (
            (np.abs(r.astype(int) - g.astype(int)) < 18)
            & (np.abs(g.astype(int) - b.astype(int)) < 18)
        )

        smoke_pixels = low_saturation & mid_brightness & gray_dominance
        smoke_mask = smoke_pixels.astype(np.uint8) * 255

        total_pixels = frame.shape[0] * frame.shape[1]
        fire_ratio = float(cv2.countNonZero(fire_mask)) / total_pixels
        smoke_ratio = float(cv2.countNonZero(smoke_mask)) / total_pixels

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fire_mask,
            connectivity=8,
        )

        largest_fire_area = 0
        if num_labels > 1:
            largest_fire_area = int(stats[1:, cv2.CC_STAT_AREA].max())

        largest_fire_ratio = largest_fire_area / total_pixels
        keyword = ""

        if fire_ratio >= 0.050 and largest_fire_ratio >= 0.040:
            severity = "CRITICAL"
            keyword = "visual_fire_critical"
        elif fire_ratio >= 0.025 and largest_fire_ratio >= 0.020:
            severity = "HIGH"
            keyword = "visual_fire_high"
        elif fire_ratio >= 0.010:
            severity = "MEDIUM"
            keyword = "visual_fire_medium"
        elif smoke_ratio >= 0.35:
            severity = "CRITICAL"
            keyword = "visual_smoke_critical"
        elif smoke_ratio >= 0.20:
            severity = "HIGH"
            keyword = "visual_smoke_high"
        elif smoke_ratio >= 0.10:
            severity = "MEDIUM"
            keyword = "visual_smoke_medium"
        else:
            severity = "NORMAL"
            keyword = ""

        return {
            "visual_severity": severity,
            "visual_keyword": keyword,
            "fire_ratio": fire_ratio,
            "smoke_ratio": smoke_ratio,
        }


# =============================================================================
# OLLAMA VLM INFERENCE
# =============================================================================
class VLMInference:
    @staticmethod
    def _async_worker(frame: np.ndarray, token_budget: int, host: Optional[str]) -> None:
        try:
            image_b64 = frame_to_base64_jpeg(frame)
            client = ollama.Client(host=host) if host else ollama.Client()
            response = client.chat(
                model=CONFIG["VLM_MODEL"],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze this camera frame for fire and smoke only. "
                            "Return exactly one category: NORMAL, MEDIUM, HIGH, or CRITICAL. "
                            "Use NORMAL when there is no visible fire or smoke. "
                            "Use MEDIUM only for light smoke or unclear early warning. "
                            "Use HIGH for visible flame, burning object, or clear smoke. "
                            "Use CRITICAL only for active fire, heavy smoke, visible flames, explosion, or immediate danger. "
                            "Do not classify people, skin, red clothing, walls, posters, or warm lighting as fire. "
                            "Start with the category, then one short reason."
                        ),
                        "images": [image_b64],
                    }
                ],
                options={
                    "num_ctx": token_budget,
                    "num_predict": 80,
                    "temperature": 0.0,
                },
            )

            text = response.get("message", {}).get("content", "").strip()
            if text:
                VLM_RESULT_QUEUE.put({"success": True, "text": text, "error": ""})
            else:
                VLM_RESULT_QUEUE.put({"success": False, "text": "", "error": "Ollama returned empty response."})
        except Exception as exc:
            VLM_RESULT_QUEUE.put({"success": False, "text": "", "error": f"Ollama backend issue: {exc}"})

    @staticmethod
    def trigger_async_inference(frame: np.ndarray) -> None:
        if not OLLAMA_AVAILABLE:
            st.session_state.last_error = "Ollama Python package is not installed."
            return

        if st.session_state.get("vlm_in_progress", False):
            return

        # WEB FIX: Ollama on "localhost" only exists on your own machine. On a
        # hosted deployment, set OLLAMA_HOST (env var or secret) to a reachable
        # server (e.g. http://my-vps:11434); without it the VLM step is skipped
        # and detection continues with YOLO + the OpenCV fallback.
        host = _get_secret("OLLAMA_HOST")
        if is_cloud_environment() and not host:
            return

        st.session_state.vlm_in_progress = True
        token_budget = ResourceGovernor.recommended_token_budget()
        thread = threading.Thread(
            target=VLMInference._async_worker,
            args=(frame.copy(), token_budget, host),
            daemon=True
        )
        thread.start()

    @staticmethod
    def drain_worker_queue() -> None:
        while not VLM_RESULT_QUEUE.empty():
            try:
                res = VLM_RESULT_QUEUE.get_nowait()
                st.session_state.vlm_in_progress = False
                if res["success"]:
                    st.session_state.latest_vlm_text = res["text"]
                    st.session_state.inference_count += 1
                    st.session_state.last_error = ""
                else:
                    st.session_state.last_error = res["error"]
            except queue.Empty:
                break


# =============================================================================
# YARA VERIFICATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def compile_yara_rules():
    if not YARA_AVAILABLE:
        return None

    if not os.path.exists(CONFIG["YARA_RULE_PATH"]):
        return None

    return yara.compile(filepath=CONFIG["YARA_RULE_PATH"])


class YARAVerifier:
    @staticmethod
    def verify(vlm_text: str, visual_result: Dict[str, Any]) -> Dict[str, Any]:
        yara_severity = "NORMAL"
        matched_rule = "no_yara_match"
        matched_source = "YARA_TEXT"

        if YARA_AVAILABLE and vlm_text:
            try:
                rules = compile_yara_rules()
                matches = rules.match(data=vlm_text.encode("utf-8", errors="ignore")) if rules else []

                if matches:
                    highest_score = -1
                    for match in matches:
                        severity = match.meta.get("severity", "NORMAL")
                        score = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["NORMAL"])["score"]

                        if score > highest_score:
                            highest_score = score
                            yara_severity = severity
                            matched_rule = match.rule

            except Exception:
                yara_severity = "NORMAL"
                matched_rule = "yara_verification_unavailable"
        else:
            matched_rule = "yara_not_installed" if not YARA_AVAILABLE else "no_vlm_text"

        visual_severity = visual_result.get("visual_severity", "NORMAL")
        yara_score = SEVERITY_STYLE[yara_severity]["score"]
        visual_score = SEVERITY_STYLE[visual_severity]["score"]

        if visual_score > yara_score:
            final_severity = visual_severity
            matched_source = "OPENCV_VISUAL_FALLBACK"
            matched_keyword = visual_result.get("visual_keyword", "visual_detection")
        else:
            final_severity = yara_severity
            matched_keyword = matched_rule

        is_hazard = final_severity != "NORMAL"
        confidence = {
            "NORMAL": 0.10,
            "MEDIUM": 0.64,
            "HIGH": 0.82,
            "CRITICAL": 0.92,
        }[final_severity]

        return {
            "severity": final_severity,
            "is_hazard": is_hazard,
            "confidence": confidence,
            "matched_keyword": matched_keyword,
            "matched_source": matched_source,
            "yara_severity": yara_severity,
            "visual_severity": visual_severity,
            "fire_ratio": visual_result.get("fire_ratio", 0.0),
            "smoke_ratio": visual_result.get("smoke_ratio", 0.0),
            "yara_available": YARA_AVAILABLE,
        }


# =============================================================================
# GOOGLE CLOUD FORENSIC BACKUP & LOCAL BUFFER
# =============================================================================
def _queue_image_for_later_upload(frame: np.ndarray, alert: Dict[str, Any]) -> None:
    try:
        pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
        os.makedirs(pending_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        severity = alert.get("severity", "UNKNOWN")
        base_name = f"{severity}_{stamp}"

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return

        with open(os.path.join(pending_dir, f"{base_name}.jpg"), "wb") as f:
            f.write(buffer.tobytes())

        with open(os.path.join(pending_dir, f"{base_name}.json"), "w") as f:
            json.dump(alert, f, default=str)
    except Exception:
        pass


def flush_pending_cloud_uploads() -> None:
    if not GCP_AVAILABLE or storage is None:
        return
    if CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
        return

    pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
    if not os.path.isdir(pending_dir):
        return

    try:
        client, _ = get_gcp_clients()
        if client is None:
            return
        bucket = client.bucket(CONFIG["GCP_BUCKET"])
    except Exception:
        return

    for filename in sorted(os.listdir(pending_dir)):
        if not filename.endswith(".jpg"):
            continue

        base_name = filename[:-4]
        image_path = os.path.join(pending_dir, filename)
        meta_path = os.path.join(pending_dir, f"{base_name}.json")

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            bucket.blob(f"cybervision_alerts/{base_name}.jpg").upload_from_string(
                image_bytes, content_type="image/jpeg"
            )

            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta_text = f.read()
                bucket.blob(f"cybervision_alerts/{base_name}.json").upload_from_string(
                    meta_text, content_type="application/json"
                )
                os.remove(meta_path)

            os.remove(image_path)
        except Exception:
            continue


def upload_to_google_cloud_async(frame: np.ndarray, alert: Dict[str, Any], cloud_sync_enabled: bool) -> None:
    # WEB FIX: `cloud_sync_enabled` is read on the Streamlit script thread and
    # passed in. The old worker read st.session_state from a plain background
    # thread, which has no session context, so the cloud upload never ran.
    def _worker():
        save_event_locally(
            time.time(),
            alert.get("severity", "UNKNOWN"),
            alert.get("description", ""),
            camera_id="CAM_01"
        )

        if not cloud_sync_enabled or not GCP_AVAILABLE or storage is None:
            return

        if CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
            return

        if not is_wifi_connected():
            _queue_image_for_later_upload(frame, alert)
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            severity = alert.get("severity", "UNKNOWN")

            image_name = f"cybervision_alerts/{severity}_{timestamp}.jpg"
            log_name = f"cybervision_alerts/{severity}_{timestamp}.json"

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                return

            client, _ = get_gcp_clients()
            if client is None:
                _queue_image_for_later_upload(frame, alert)
                return
            bucket = client.bucket(CONFIG["GCP_BUCKET"])

            bucket.blob(image_name).upload_from_string(buffer.tobytes(), content_type="image/jpeg")
            bucket.blob(log_name).upload_from_string(
                json.dumps(alert, default=str), content_type="application/json"
            )
        except Exception:
            _queue_image_for_later_upload(frame, alert)

    threading.Thread(target=_worker, daemon=True).start()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def determine_hazard_title(latest: Dict[str, Any]) -> str:
    if not latest or latest.get("severity") == "NORMAL":
        return "Environment is Safe"

    classes = [c.strip().lower() for c in latest.get("yolo_classes", "").split(",") if c.strip()]
    description = latest.get("description", "").lower()
    keyword = str(latest.get("matched_keyword", "")).lower()

    has_fire = (
        any("fire" in item for item in classes)
        or "fire" in description
        or "fire" in keyword
        or latest.get("fire_ratio", 0) > 0.010
    )

    has_smoke = (
        any("smoke" in item for item in classes)
        or "smoke" in description
        or "smoke" in keyword
        or latest.get("smoke_ratio", 0) > 0.10
    )

    if has_fire:
        return "Fire Detected"
    if has_smoke:
        return "Smoke Detected"

    if "fire" in description or "fire" in keyword:
        return "Fire Detected"

    return "Smoke Detected"


# =============================================================================
# DETECTION PIPELINE
# =============================================================================
def run_detection_pipeline(frame: np.ndarray) -> Dict[str, Any]:
    if frame is None or frame.size == 0:
        default_alert = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "description": "NORMAL: Camera stream starting...",
            "success": True,
            "latency": 0.0,
            "verified": False,
            "yolo_detected": False,
            "yolo_classes": "",
            "yolo_confidence": 0.0,
            "severity": "NORMAL",
            "is_hazard": False,
            "confidence": 0.0,
            "matched_keyword": "none",
            "matched_source": "SYSTEM",
            "yara_severity": "NORMAL",
            "visual_severity": "NORMAL",
            "fire_ratio": 0.0,
            "smoke_ratio": 0.0,
            "yara_available": YARA_AVAILABLE,
        }
        play_critical_alarm(False)
        st.session_state.yolo_severity = "NORMAL"
        return {
            "annotated_frame": frame,
            "yolo": {"detected": False, "detections": []},
            "alert": default_alert,
        }

    ResourceGovernor.check_resource_pressure()
    start_time = time.time()

    VLMInference.drain_worker_queue()

    # 1. Primary Object Detection (YOLO)
    yolo_detector = st.session_state.get("yolo_detector")
    if yolo_detector is None:
        yolo_detector = YOLOFireSmokeDetector()
        st.session_state.yolo_detector = yolo_detector

    yolo_detector.drain_worker_queue()

    frame_number = st.session_state.frames_processed
    run_yolo_now = (
        st.session_state.get("yolo_detection") is None
        or frame_number % max(1, CONFIG["YOLO_EVERY_N_FRAMES"]) == 0
    )

    if run_yolo_now:
        yolo_detector.trigger_async_detect(frame)

    yolo_result = st.session_state.get("yolo_detection") or {
        "detected": False, "detections": [], "highest_confidence": 0.0,
        "classes": [], "annotated_frame": frame,
    }

    yolo_severity = yolo_detector.hazard_severity(yolo_result) if yolo_result.get("detected") else "NORMAL"
    st.session_state.yolo_severity = yolo_severity

    run_visual_now = (
        st.session_state.get("last_visual_result") is None
        or frame_number % max(1, CONFIG["VISUAL_DETECT_EVERY_N_FRAMES"]) == 0
    )
    if run_visual_now:
        visual_result = VisualFireSmokeDetector.detect(frame)
        st.session_state.last_visual_result = visual_result
    else:
        visual_result = st.session_state.last_visual_result

    candidate_hazard = yolo_result.get("detected", False)
    analyze_this_frame = (
        candidate_hazard
        and frame_number % max(1, CONFIG["ANALYZE_EVERY_N_FRAMES"]) == 0
    )

    if analyze_this_frame:
        VLMInference.trigger_async_inference(frame)

    vlm_text = st.session_state.get("latest_vlm_text", "")
    if not vlm_text:
        vlm_text = "NORMAL: Monitoring active."

    verdict = YARAVerifier.verify(vlm_text, visual_result)

    latency = time.time() - start_time
    st.session_state.latency_history.append(latency)

    public_description = vlm_text
    if yolo_result.get("detected"):
        detection_text = ", ".join(f"{d['class']} {d['confidence']:.0%}" for d in yolo_result["detections"])
        public_description = f"YOLO detected: {detection_text}. VLM: {vlm_text}"

    is_hazard = yolo_result.get("detected", False)
    confidence = yolo_result.get("highest_confidence", 0.0) if is_hazard else 0.0

    alert = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "description": public_description,
        "success": True,
        "latency": latency,
        "verified": is_hazard,
        "yolo_detected": is_hazard,
        "yolo_classes": ", ".join(yolo_result.get("classes", [])),
        "yolo_confidence": confidence,
        "severity": yolo_severity,
        "is_hazard": is_hazard,
        "confidence": confidence,
        "matched_keyword": verdict.get("matched_keyword", "none"),
        "matched_source": "YOLO_DETECTOR" if is_hazard else "SYSTEM",
        "yara_severity": verdict.get("yara_severity", "NORMAL"),
        "visual_severity": visual_result.get("visual_severity", "NORMAL"),
        "fire_ratio": visual_result.get("fire_ratio", 0.0),
        "smoke_ratio": visual_result.get("smoke_ratio", 0.0),
        "yara_available": YARA_AVAILABLE,
    }

    st.session_state.alert_history.append(alert)
    st.session_state.latest_detection = alert

    if yolo_severity != "NORMAL":
        hazard_title = determine_hazard_title(alert)
        dispatch_hazard_alerts(yolo_severity, hazard_title)

    play_critical_alarm(yolo_severity == "CRITICAL")

    # WEB FIX: throttle uploads. At ~10 ticks/second this used to write a
    # SQLite row + Firestore doc + bucket objects for every single frame.
    if is_hazard:
        now = time.time()
        if now - st.session_state.get("last_cloud_upload_ts", 0.0) >= CONFIG["CLOUD_UPLOAD_MIN_INTERVAL_SEC"]:
            st.session_state.last_cloud_upload_ts = now
            upload_to_google_cloud_async(
                frame, dict(alert), bool(st.session_state.cloud_sync_enabled)
            )

    return {
        "annotated_frame": yolo_result.get("annotated_frame", frame),
        "yolo": yolo_result,
        "alert": alert,
    }


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar_profile() -> None:
    st.markdown("<div class='sidebar-footer-divider'></div>", unsafe_allow_html=True)

    user_name = getattr(st.user, "name", None) or "Signed in"
    user_email = getattr(st.user, "email", "") or ""
    user_picture = getattr(st.user, "picture", None)

    if user_picture:
        avatar_html = f'<img class="profile-avatar" src="{user_picture}" alt="avatar" />'
    else:
        initial = (user_name or "?")[:1].upper()
        avatar_html = f'<div class="profile-avatar profile-avatar-fallback">{initial}</div>'

    st.markdown(
        f"""
        <div class="profile-card">
            {avatar_html}
            <div class="profile-text">
                <div class="profile-name">{user_name}</div>
                <div class="profile-email">{user_email}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Log out", icon=":material/logout:", use_container_width=True, key="logout_btn"):
        st.logout()


def render_login_page() -> None:
    inject_custom_css()

    st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])

    with mid:
        st.markdown(
            """
            <div style="text-align:center;">
                <div style="font-size:2.6rem;">🔥</div>
                <div style="font-weight:900; font-size:1.5rem; color:#c2410c; margin-top:0.2rem;">CyberVision</div>
                <div style="font-size:0.7rem; color:#6b7280; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1.4rem;">
                    Edge Fire &amp; Smoke AI
                </div>
                <p style="color:#6b7280; font-size:0.9rem; margin-bottom:1.4rem;">
                    Sign in to access the live monitoring dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Sign in with Google", icon=":material/login:", use_container_width=True, type="primary"):
            try:
                st.login("google")
            except Exception as exc:
                st.error(
                    "Google sign-in isn't configured yet. Add an [auth] / "
                    "[auth.google] section with your OAuth client credentials to "
                    f".streamlit/secrets.toml (see secrets.toml.example). Details: {exc}"
                )


def render_sidebar_nav() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-icon">🔥</div>
                <div>
                    <div class="brand-name">CyberVision</div>
                    <div class="brand-subtitle">Edge Fire &amp; Smoke AI</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cam_running = st.session_state.camera_running
        wifi_online = is_wifi_connected()

        st.markdown(
            f"""
            <div class="status-footer">
                <div class="status-row">
                    <span class="status-dot" style="background:{'#30d158' if cam_running else '#ff3b30'};"></span>
                    Camera {"running" if cam_running else "stopped"}
                </div>
                <div class="status-row">
                    <span class="status-dot" style="background:{'#30d158' if wifi_online else '#ffd60a'};"></span>
                    Network {"online" if wifi_online else "offline"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-footer-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='nav-section-label'>Workspace</div>", unsafe_allow_html=True)

        # "Export Reports" now lives inside the Data Logs tab.
        nav_items = [
            ("Dashboard", "dashboard"),
            ("Analytics", "bar_chart"),
            ("Data Logs", "table_chart"),
        ]

        for label, icon_name in nav_items:
            is_active = st.session_state.active_page == label
            if st.button(
                label,
                icon=f":material/{icon_name}:",
                key=f"nav_{label.replace(' ', '_').lower()}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_page = label
                st.rerun()

        with st.container(key="sidebar_bottom_block"):
            render_sidebar_profile()


def render_dashboard_settings_panel() -> None:
    with st.container():
        st.markdown("<div class='stats-panel-title'>Server Setup</div>", unsafe_allow_html=True)

        st.session_state.sound_enabled = st.toggle(
            "Alert sound",
            value=st.session_state.sound_enabled,
        )

        wifi_online = is_wifi_connected()
        sync_label = (
            "Data synced _to cloud_"
            if (st.session_state.cloud_sync_enabled and wifi_online)
            else "Data synced _locally_"
        )
        st.session_state.cloud_sync_enabled = st.toggle(
            sync_label,
            value=st.session_state.cloud_sync_enabled,
        )

        if st.session_state.cloud_sync_enabled:
            if not GCP_AVAILABLE:
                st.caption(":material/error: `google-cloud-firestore` / `google-cloud-storage` not installed.")
            elif CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
                st.caption(":material/warning: Set CONFIG[\"GCP_BUCKET\"] / GCP_PROJECT_ID in main.py.")
            elif _get_gcp_service_account_info() is None:
                st.caption(":material/warning: No GCP credentials found (secrets, env var, or key file).")
            else:
                st.caption(":material/check_circle: GCP credentials loaded.")

        timeout_option = st.selectbox(
            "Inference timeout target",
            ["1.0s", "2.0s", "3.0s", "5.0s"],
            index=1,
            key="inference_timeout_select"
        )
        CONFIG["INFERENCE_TIMEOUT"] = float(timeout_option.replace("s", ""))

        st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='stats-panel-title'>Resources</div>", unsafe_allow_html=True)

        metrics = SystemMonitor.get_metrics()
        cpu_high = metrics["cpu"] >= CONFIG["CPU_THRESHOLD"]
        ram_high = metrics["ram"] >= CONFIG["RAM_THRESHOLD"]
        cpu_delta = ":material/check_circle:" if metrics["cpu"] < 70 else ":material/warning:" if not cpu_high else ":material/error:"
        ram_delta = ":material/check_circle:" if metrics["ram"] < 70 else ":material/warning:" if not ram_high else ":material/error:"

        m1, m2 = st.columns(2)
        with m1:
            with st.container(key="cpu_metric_high" if cpu_high else "cpu_metric_normal"):
                st.metric("CPU", f"{metrics['cpu']:.1f}%", cpu_delta)
        with m2:
            with st.container(key="ram_metric_high" if ram_high else "ram_metric_normal"):
                st.metric("RAM", f"{metrics['ram']:.1f}%", ram_delta)

        if st.session_state.latency_history:
            avg_latency = sum(st.session_state.latency_history) / len(st.session_state.latency_history)
            st.metric("Avg Latency", f"{avg_latency * 1000:.0f} ms")
        else:
            st.metric("Avg Latency", "—")

        if st.session_state.last_error:
            st.info("Model/YARA backend notice: Active fallback running.")

        if not is_yolo_model_available():
            with st.expander(":material/error: YOLO not active — details", expanded=True):
                for label, value in get_yolo_debug_info().items():
                    st.text(f"{label}: {value}")


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================
def render_dashboard_header() -> None:
    cam_running = st.session_state.camera_running
    yolo_ready = is_yolo_model_available()

    if cam_running and yolo_ready:
        pill_text, pill_color = "YOLO ACTIVE", "#16a34a"
    elif cam_running:
        pill_text, pill_color = "FALLBACK MODE", "#ca8a04"
    else:
        pill_text, pill_color = "STANDBY", "#6b7280"

    # (The old markup had two stray closing </div> tags that broke the layout.)
    st.markdown(
        f"""
        <div class="dash-header">
            <div>
                <div class="dash-header-title">Live Camera Feeds</div>
                <div class="dash-header-subtitle">Real-time AI fire &amp; smoke detection and alerting.</div>
            </div>
            <div class="status-pill-live" style="background:{pill_color}1f; color:{pill_color};">
                <span class="status-pill-dot" style="background:{pill_color};"></span>
                {pill_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_stats_body() -> None:
    latest = st.session_state.get("latest_detection") or {}

    severity = str(st.session_state.get("yolo_severity", "NORMAL")).upper()
    if severity not in SEVERITY_STYLE:
        severity = "NORMAL"

    total = len(st.session_state.alert_history)
    hazards = sum(1 for item in st.session_state.alert_history if item["is_hazard"])
    confidence = (latest.get("confidence", 0) * 100 if latest else 0)

    with st.container():
        st.markdown("<div class='stats-panel-title'>Active Monitoring</div>", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            with st.container(key=f"severity_metric_{severity.lower()}"):
                st.metric("Severity", severity)
        r1c2.metric("Confidence", f"{confidence:.0f}%")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Frames", st.session_state.frames_processed)
        r2c2.metric("Hazards", hazards, f"of {total}")


def render_live_stats_panel() -> None:
    refresh = 0.1 if st.session_state.get("camera_running") else None
    st.fragment(_render_live_stats_body, run_every=refresh)()


def render_video_status_bar() -> None:
    fps = st.session_state.get("fps", 0.0)
    avg_latency = (
        sum(st.session_state.latency_history) / len(st.session_state.latency_history)
        if st.session_state.latency_history else 0.0
    )
    resolution = f"{CONFIG['FRAME_WIDTH']}x{CONFIG['FRAME_HEIGHT']}"

    st.markdown(
        f"""
        <div class="video-status-bar muted">
            <span>RESOLUTION: <b>{resolution}</b></span>
            <span>INFERENCE: <b>{avg_latency * 1000:.0f} ms</b></span>
            <span>FPS: <b>{fps:.1f}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resource_trend_preview() -> None:
    metrics_df = pd.DataFrame(list(st.session_state.metrics_history))
    if metrics_df.empty:
        st.info("No resource metrics yet.")
        return
    metrics_df = metrics_df.set_index("time")
    st.line_chart(metrics_df[["cpu", "ram"]], height=180)


# =============================================================================
# LIVE VIDEO STREAM
# =============================================================================
def _show_frame(slot, image, **kwargs) -> None:
    """Draw an image into a slot; works on old and new Streamlit versions."""
    try:
        slot.image(image, width="stretch", **kwargs)
    except Exception:
        slot.image(image, use_container_width=True, **kwargs)


@st.fragment(run_every=0.1)
def live_camera_fragment() -> None:
    # The frame/status slots are created INSIDE the fragment. A placeholder
    # created outside and passed in raises StreamlitInvalidLayoutContextError when
    # the fragment reruns on its own (run_every).
    frame_slot = st.empty()
    status_slot = st.empty()

    def _redraw_last_frame() -> None:
        last = st.session_state.get("last_frame_rgb")
        if last is not None:
            _show_frame(frame_slot, last, channels="RGB", output_format="JPEG")

    render_custom_hazard_toast()

    if not st.session_state.get("camera_running", False):
        return

    if st.session_state.get("_frame_processing_busy"):
        if not st.session_state.get("use_webrtc"):
            _redraw_last_frame()
        return

    st.session_state._frame_processing_busy = True
    try:
        use_webrtc = bool(st.session_state.get("use_webrtc"))
        active_frames, primary_frame = VideoCaptureManager.capture_active_frames()

        if primary_frame is None and not active_frames:
            # WEB FIX: while the browser camera is still connecting, show a clear
            # "starting" tile (not a blank area) and an info message (not an error).
            if not use_webrtc:
                if st.session_state.get("last_frame_rgb") is not None:
                    _redraw_last_frame()
                else:
                    _show_frame(
                        frame_slot,
                        VideoCaptureManager.placeholder_frame(
                            "Starting camera...",
                            "Allow camera access in your browser.",
                        ),
                        channels="RGB",
                        output_format="JPEG",
                    )
            message = st.session_state.get("last_error") or "Camera unavailable."
            if use_webrtc:
                status_slot.info(message)
            else:
                status_slot.error(message)
            return

        current_time = time.time()
        last_time = st.session_state.get("last_frame_time", current_time)
        fps = 1 / max(current_time - last_time, 0.001)

        st.session_state.fps = fps
        st.session_state.last_frame_time = current_time
        st.session_state.frames_processed += 1

        pipeline_result = run_detection_pipeline(primary_frame)
        yolo_result = pipeline_result.get("yolo", {})

        if 0 in active_frames and primary_frame is not None:
            active_frames[0] = YOLOFireSmokeDetector.draw_boxes_on_frame(primary_frame, yolo_result)

        if use_webrtc:
            # A visitor has exactly one webcam: show it full-width instead of one
            # live tile surrounded by eight "NO CAMERA" tiles.
            display = active_frames[0].copy()
        else:
            display = construct_3x3_grid(active_frames)

        severity = st.session_state.get("yolo_severity", "NORMAL")
        yolo_status = "DETECTED" if yolo_result.get("detected") else "CLEAR"

        hud_line_1 = f"FPS: {fps:.1f} | YOLO: {yolo_status}"
        hud_line_2 = f"INF: {st.session_state.get('inference_count', 0)} | STATUS: {severity}"

        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (430, 82), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, display, 0.50, 0, display)

        cv2.putText(display, hud_line_1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
        cv2.putText(display, hud_line_2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame_rgb = display_rgb

        if use_webrtc:
            # The visitor's own <video> element is the live display. Hand the
            # latest detections to the processor so boxes + HUD are drawn onto
            # the stream itself (no per-frame image upload to the browser).
            ctx = st.session_state.get("webrtc_ctx")
            processor = getattr(ctx, "video_processor", None)
            if processor is not None:
                processor.set_overlay(
                    yolo_result,
                    int(primary_frame.shape[1]),
                    int(primary_frame.shape[0]),
                    (hud_line_1, hud_line_2),
                )
            status_slot.empty()
        else:
            # JPEG encodes far faster and more consistently than the default PNG.
            _show_frame(frame_slot, display_rgb, channels="RGB", output_format="JPEG")
    finally:
        st.session_state._frame_processing_busy = False


def render_video_feed() -> None:
    header_col, btn_col1, btn_col2 = st.columns([6, 1, 1])

    with btn_col1:
        if st.button("Start", icon=":material/play_arrow:", use_container_width=True, disabled=st.session_state.camera_running, key="start_cam_btn"):
            st.session_state.camera_running = True
            st.session_state.last_error = ""
            st.session_state.last_vlm_candidate_frame = None
            st.session_state.latest_vlm_text = ""
            st.session_state.last_frame_time = time.time()
            st.session_state.last_frame_rgb = None
            st.session_state.latest_detection = None
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.alarm_active = False
            st.session_state.active_hazard_toast = None

            if st.session_state.get("yolo_detector") is None:
                st.session_state.yolo_detector = YOLOFireSmokeDetector()

            st.rerun()

    with btn_col2:
        if st.button("Stop", icon=":material/stop:", use_container_width=True, disabled=not st.session_state.camera_running, key="stop_cam_btn"):
            st.session_state.camera_running = False
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.alarm_active = False
            st.session_state.active_hazard_toast = None
            release_camera()
            st.rerun()

    if st.session_state.get("webrtc_unavailable_on_cloud"):
        st.error(
            "This deployment has no local camera device, and the browser-camera "
            "(streamlit-webrtc) packages failed to load — see the deployment logs. "
            "Detection cannot start until that's fixed."
        )

    if not st.session_state.camera_running:
        video_placeholder = st.empty()
        if st.session_state.get("last_frame_rgb") is not None:
            _show_frame(video_placeholder, st.session_state.last_frame_rgb)
        else:
            _show_frame(
                video_placeholder,
                VideoCaptureManager.placeholder_frame("Camera Stopped", "Click Start to begin monitoring."),
            )
        st.info("Camera is stopped. Detection paused.")
        return

    # The WebRTC connector is only mounted while the camera is running, so
    # pressing Stop also releases the visitor's webcam.
    if st.session_state.get("use_webrtc"):
        render_browser_camera_widget()

    live_camera_fragment()


# =============================================================================
# DATA LOGS (now includes Export Reports)
# =============================================================================
def _build_forensic_display_df() -> Optional[pd.DataFrame]:
    if not st.session_state.alert_history:
        return None

    df = pd.DataFrame(list(st.session_state.alert_history))

    display_cols = [
        "timestamp", "severity", "matched_source", "matched_keyword",
        "yara_severity", "visual_severity", "confidence", "latency",
        "fire_ratio", "smoke_ratio", "success", "description",
        "yolo_detected", "yolo_classes", "yolo_confidence",
    ]

    display_df = df[display_cols].copy()
    display_df["latency_ms"] = (display_df["latency"] * 1000).round(1)
    display_df["confidence"] = ((display_df["confidence"] * 100).round(1).astype(str) + "%")
    display_df = display_df.drop(columns=["latency"])
    return display_df


def render_data_logs_page() -> None:
    display_df = _build_forensic_display_df()

    header_col, export_col, reset_col = st.columns([4, 1.6, 1.2])
    with header_col:
        st.markdown("### Data Logs")
        st.markdown("<p class='muted'>Full forensic event history captured by the detection pipeline.</p>", unsafe_allow_html=True)

    with export_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if display_df is None:
            st.button(
                "Export report",
                icon=":material/download:",
                use_container_width=True,
                disabled=True,
                key="export_csv_disabled_btn",
                help="No events logged yet — nothing to export.",
            )
        else:
            st.download_button(
                "Export report",
                icon=":material/download:",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cybervision_forensic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_btn",
                help=f"Download {len(display_df)} logged events as CSV.",
            )

    with reset_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("Reset logs", icon=":material/restart_alt:", use_container_width=True, key="reset_logs_btn"):
            st.session_state.alert_history.clear()
            st.session_state.latency_history.clear()
            st.session_state.latest_detection = None
            st.session_state.yolo_severity = "NORMAL"
            st.toast("Logs reset.")
            st.rerun()

    if display_df is None:
        st.info("No events logged yet.")
        return

    st.caption(f"{len(display_df)} events ready to export.")
    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)


def render_charts() -> None:
    c1, c2 = st.columns(2)
    metrics_df = pd.DataFrame(list(st.session_state.metrics_history))

    if not metrics_df.empty:
        metrics_df = metrics_df.set_index("time")
        with c1:
            st.markdown("#### CPU/RAM Usage")
            st.line_chart(metrics_df[["cpu", "ram"]])
    else:
        c1.info("No resource metrics yet.")

    if st.session_state.alert_history:
        alert_df = pd.DataFrame(list(st.session_state.alert_history))
        severity_counts = alert_df["severity"].value_counts().reindex(["NORMAL", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        with c2:
            st.markdown("#### Severity Distribution")
            st.bar_chart(severity_counts)
    else:
        c2.info("No alert statistics yet.")


def render_analytics_page() -> None:
    st.markdown("### Analytics")
    st.markdown("<p class='muted'>Resource usage trends and hazard severity distribution.</p>", unsafe_allow_html=True)
    render_charts()


def render_dashboard_page() -> None:
    metrics = SystemMonitor.get_metrics()
    if metrics["cpu"] > 90.0 or metrics["ram"] > 88.0:
        st.error(f"Critical System Load! CPU: {metrics['cpu']}% | RAM: {metrics['ram']}%. System throttling.", icon=":material/error:")

    render_dashboard_header()

    video_col, stats_col = st.columns([2.6, 1], gap="medium")

    # Diagnostic switch: open the app URL with ?safe=1 to skip the camera / WebRTC
    # widget. If the dashboard loads in safe mode, the problem is in that widget.
    safe_mode = str(st.query_params.get("safe", "")) == "1"

    with video_col:
        with st.container():
            if safe_mode:
                st.info("Safe mode: camera and WebRTC are disabled for diagnostics.")
            else:
                render_video_feed()
            render_video_status_bar()

    with stats_col:
        render_live_stats_panel()
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        render_dashboard_settings_panel()

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Live Trends")
    render_resource_trend_preview()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    init_session_state()

    try:
        logged_in = bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        logged_in = False

    if not logged_in:
        render_login_page()
        return

    start_sync_thread()
    inject_custom_css()

    # Export Reports was merged into Data Logs; keep old session state valid.
    if st.session_state.active_page == "Export Reports":
        st.session_state.active_page = "Data Logs"

    render_sidebar_nav()

    page = st.session_state.active_page

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Data Logs":
        render_data_logs_page()
    else:
        st.session_state.active_page = "Dashboard"
        render_dashboard_page()


if __name__ == "__main__":
    main()