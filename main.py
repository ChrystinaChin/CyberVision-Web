import base64
import io
import json
import os
import queue
import re
import socket
import sqlite3
import threading
import time
import wave
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# Initialize psutil counter once globally for non-blocking reads
psutil.cpu_percent(interval=None)

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
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object
    RTCConfiguration = None
    WEBRTC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False


def is_cloud_environment() -> bool:
    """Detect a hosted environment (e.g. Streamlit Community Cloud) that has
    no physical camera device attached, so cv2.VideoCapture can never work
    there and the app should fall back to capturing the visitor's own
    browser camera over WebRTC instead.
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

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG: Dict[str, Any] = {
    "APP_NAME": "CyberVision",
    "APP_SUBTITLE": "Adaptive Context Optimization at the Edge",
    "VLM_MODEL": "moondream",
    "CAMERA_INDEX": 0,
    "CAMERA_SOURCES": [0],  # Primary webcam only; prevents duplicate/phantom camera feeds
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 300,
    "JPEG_QUALITY": 70,
    "FRAME_DELAY_SEC": 0.22,
    "ANALYZE_EVERY_N_FRAMES": 6,
    "CPU_THRESHOLD": 85,
    "RAM_THRESHOLD": 85,
    "ALERT_HISTORY_MAX": 100,
    "METRICS_HISTORY_MAX": 100,
    "LATENCY_HISTORY_MAX": 50,
    "TOAST_DISPLAY_SECONDS": 6.0,
    "YARA_RULE_PATH": "hazard_rules.yar",
    "GCP_BUCKET": "cybervision_history",
    "GCP_PROJECT_ID": "project-dc6771c8-2eaf-4371-b74",
    "DB_FILE": "cybervision_buffer.db",
    "PENDING_UPLOADS_DIR": "pending_gcp_uploads",
    "YOLO_MODEL_PATH": next(
        (
            p for p in (
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOLO_dataset.pt"),
            )
            if os.path.isfile(p)
        ),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"),
    ),
    "YOLO_CONFIDENCE": 0.15,
    "YOLO_IMAGE_SIZE": 640,
    "THROTTLE_MIN_INTERVAL": 0.75,
    "THROTTLE_YOLO_IMAGE_SIZE": 320,
}

SEVERITY_STYLE = {
    "NORMAL": {"score": 0, "icon": ":material/check_circle:", "color": "#30d158"},
    "MEDIUM": {"score": 1, "icon": ":material/warning:", "color": "#ffd60a"},
    "HIGH": {"score": 2, "icon": ":material/error:", "color": "#ff9500"},
    "CRITICAL": {"score": 3, "icon": ":material/dangerous:", "color": "#ff3b30"},
}

VLM_RESULT_QUEUE: queue.Queue = queue.Queue()


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


def save_event_locally(
    timestamp: float,
    severity: str,
    description: str,
    camera_id: str = "CAM_01",
) -> None:
    formatted_ts = datetime.fromtimestamp(timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
                INSERT INTO pending_events (timestamp, severity, description, camera_id, synced)
                VALUES (?, ?, ?, ?, 0)
            """,
            (formatted_ts, severity, description, camera_id),
        )
        conn.commit()


def _probe_network(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_wifi_connected(host: str = "8.8.8.8", port: int = 53, timeout: float = 2) -> bool:
    if _probe_network(host, port, timeout):
        return True
    if _probe_network("1.1.1.1", 53, timeout):
        return True
    return _probe_network("storage.googleapis.com", 443, timeout)


@st.cache_resource(show_spinner=False)
def get_network_monitor() -> Dict[str, Any]:
    state: Dict[str, Any] = {"online": is_wifi_connected(timeout=1.5), "checked_at": time.time()}

    def _loop() -> None:
        while True:
            time.sleep(3.0)
            try:
                state["online"] = is_wifi_connected(timeout=1.5)
                state["checked_at"] = time.time()
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True).start()
    return state


def is_network_online() -> bool:
    return bool(get_network_monitor()["online"])


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
            except Exception as exc:
                print(f"[GCP sync] Firestore sync failed: {exc!r}", flush=True)

            flush_pending_cloud_uploads()
        time.sleep(10)


@st.cache_resource(show_spinner=False)
def _start_sync_thread_once() -> bool:
    init_db()
    threading.Thread(target=sync_worker_loop, daemon=True).start()
    return True


def start_sync_thread() -> None:
    _start_sync_thread_once()
    st.session_state.sync_thread_started = True


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
        "fps": 0.0,
        "last_frame_time": time.time(),
        "latest_detection": None,
        "yolo_severity": "NORMAL",
        "yolo_detector": None,
        "yolo_detection": None,
        "last_vlm_candidate_frame": None,
        "latest_vlm_text": "",
        "alarm_active": False,
        "vlm_in_progress": False,
        "last_triggered_alert_ts": 0.0,
        "active_hazard_toast": None,
        "active_page": "Dashboard",
        "throttle_active": False,
        "last_detection_run_ts": 0.0,
        "webrtc_ctx": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "use_webrtc" not in st.session_state:
        needs_browser_camera = is_cloud_environment() or not local_camera_available()
        st.session_state.use_webrtc = needs_browser_camera and WEBRTC_AVAILABLE
        st.session_state.webrtc_unavailable_on_cloud = needs_browser_camera and not WEBRTC_AVAILABLE


@st.cache_data
def inject_custom_css() -> None:
    bg = "#fff7ed"
    panel_warm = "#fff3e6"
    text = "#1f2937"
    muted = "#6b7280"
    accent = "#f97316"
    accent_dark = "#c2410c"
    border = "#fed7aa"

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

        .st-key-webrtc_transport {{
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
        }}
        .st-key-webrtc_transport iframe {{
            height: 1px !important;
            min-height: 1px !important;
            max-height: 1px !important;
            opacity: 0 !important;
            pointer-events: none !important;
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
            border: 1px solid {border} !important;
            color: {accent_dark} !important;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {panel_warm} 100%) !important;
            border-right: 1px solid {border};
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
            border: 1px solid {border};
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
            border: 1px solid rgba(249,115,22,0.35);
            color: {accent_dark};
            font-weight: 900;
            font-size: 0.7rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.94);
            border: 1px solid {border};
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

        .stButton > button {{
            border-radius: 14px;
            font-weight: 800;
            border: 1px solid #fb923c;
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            color: {accent_dark};
            box-shadow: 0 8px 18px rgba(249,115,22,0.12);
            transition: all 0.18s ease-in-out;
        }}

        .stButton > button:hover {{
            background: linear-gradient(135deg, #fed7aa, #fdba74);
            color: #7c2d12;
            border-color: {accent};
            transform: translateY(-1px);
        }}

        div[data-testid="stElementContainer"]:has(.hazard-toast),
        div[data-testid="stMarkdown"]:has(.hazard-toast),
        div[data-testid="stMarkdownContainer"]:has(.hazard-toast) {{
            opacity: 1 !important;
            filter: none !important;
            transform: none !important;
            transition: none !important;
        }}

        .hazard-toast {{
            position: fixed !important;
            top: 4rem;
            right: 1.1rem;
            z-index: 2147483647 !important;
            opacity: 1 !important;
            isolation: isolate;
            background: #ffffff;
            border: 1px solid {border};
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
            border: 1px solid transparent !important;
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
            border: 1px solid {accent_dark} !important;
            box-shadow: 0 8px 18px rgba(249,115,22,0.30) !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {accent_dark}, #9a3412) !important;
            color: #ffffff !important;
        }}

        .sidebar-footer-divider {{
            border-top: 1px solid {border};
            margin: 1.2rem 0 0.9rem 0;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricValue"],
        .st-key-ram_metric_high [data-testid="stMetricValue"] {{
            color: #ff3b30 !important;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricDelta"],
        .st-key-ram_metric_high [data-testid="stMetricDelta"],
        .st-key-cpu_metric_high [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_high [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_high [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_high [data-testid="stMetricDelta"] svg {{
            color: #ff3b30 !important;
            fill: #ff3b30 !important;
        }}
        .st-key-cpu_metric_high [data-testid="stMetricDelta"],
        .st-key-ram_metric_high [data-testid="stMetricDelta"] {{
            background: rgba(255, 59, 48, 0.12) !important;
        }}

        .st-key-cpu_metric_normal [data-testid="stMetricDelta"],
        .st-key-ram_metric_normal [data-testid="stMetricDelta"],
        .st-key-cpu_metric_normal [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_normal [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_normal [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_normal [data-testid="stMetricDelta"] svg {{
            color: #16a34a !important;
            fill: #16a34a !important;
        }}
        .st-key-cpu_metric_normal [data-testid="stMetricDelta"],
        .st-key-ram_metric_normal [data-testid="stMetricDelta"] {{
            background: rgba(22, 163, 74, 0.12) !important;
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
            border: 1px solid {border};
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
            border: 1px solid;
            background: rgba(255,255,255,0.9);
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

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# AUDIO & NOTIFICATION DISPATCHER
# =============================================================================
def build_beep_wav_base64(duration: float = 0.35, freq: int = 1050, sample_rate: int = 44100) -> str:
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


def play_alert_sound() -> None:
    if not st.session_state.sound_enabled:
        return

    audio_b64 = build_beep_wav_base64(duration=0.45, freq=1100)
    audio_html = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def build_siren_wav_base64(
    duration: float = 1.6,
    low_freq: int = 900,
    high_freq: int = 1500,
    sample_rate: int = 44100,
) -> str:
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

    detection_line = f"The system detected clear signs of {hazard_word}."
    action_line = "Please evacuate from the area immediately."
    checked_time = datetime.now().strftime("%H:%M:%S")

    st.session_state.active_hazard_toast = {
        "severity": severity_norm,
        "alert_title": alert_title,
        "detection_line": detection_line,
        "action_line": action_line,
        "checked_time": checked_time,
        "triggered_at": now,
    }

    if severity_norm != "CRITICAL":
        play_alert_sound()


def render_custom_hazard_toast() -> None:
    toast_data = st.session_state.get("active_hazard_toast")
    if not toast_data:
        return

    if time.time() - toast_data["triggered_at"] > CONFIG["TOAST_DISPLAY_SECONDS"]:
        return

    st.markdown(
        f"""
        <div class="hazard-toast">
            <div class="toast-title">{toast_data['alert_title']}</div>
            <div class="toast-line toast-detection">{toast_data['detection_line']}</div>
            <div class="toast-line toast-action">{toast_data['action_line']}</div>
            <div class="toast-meta">Last checked: {toast_data['checked_time']}</div>
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
            cpu_val = float(psutil.cpu_percent(interval=None))
            ram_val = float(psutil.virtual_memory().percent)
            
            metrics = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": cpu_val,
                "ram": ram_val,
            }

            if "metrics_history" in st.session_state:
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
    from streamlit_webrtc import VideoHTMLAttributes

    class BrowserCameraProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._latest_frame: Optional[np.ndarray] = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                self._latest_frame = img
            return frame

        def get_latest_frame(self) -> Optional[np.ndarray]:
            with self._lock:
                return None if self._latest_frame is None else self._latest_frame.copy()
else:
    BrowserCameraProcessor = None
    VideoHTMLAttributes = None


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


_GCP_CLIENT_LOCK = threading.Lock()
_GCP_CLIENTS: Dict[str, Any] = {"storage": None, "firestore": None}


def gcp_credentials_configured() -> bool:
    return _get_gcp_service_account_info() is not None or bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )


def get_gcp_clients() -> Tuple[Optional[Any], Optional[Any]]:
    if not GCP_AVAILABLE:
        return None, None

    with _GCP_CLIENT_LOCK:
        if _GCP_CLIENTS["storage"] is not None and _GCP_CLIENTS["firestore"] is not None:
            return _GCP_CLIENTS["storage"], _GCP_CLIENTS["firestore"]

        info = _get_gcp_service_account_info()
        project_id = CONFIG["GCP_PROJECT_ID"]

        try:
            if info and gcp_service_account is not None:
                credentials = gcp_service_account.Credentials.from_service_account_info(info)
                project_id = info.get("project_id", project_id)
                storage_client = storage.Client(credentials=credentials, project=project_id)
                db_client = firestore.Client(
                    credentials=credentials, project=project_id, database="cybervision"
                )
            else:
                storage_client = storage.Client(project=project_id)
                db_client = firestore.Client(project=project_id, database="cybervision")
        except Exception as exc:
            print(f"[GCP] client init failed: {exc!r}", flush=True)
            return None, None

        _GCP_CLIENTS["storage"] = storage_client
        _GCP_CLIENTS["firestore"] = db_client
        return storage_client, db_client


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


def render_browser_camera_widget(playing: bool) -> None:
    if not WEBRTC_AVAILABLE:
        if playing:
            st.error(
                "Browser camera mode requires the `streamlit-webrtc` and `av` packages, "
                "which are listed in requirements.txt but failed to import."
            )
        return

    with st.container(key="webrtc_transport"):
        ctx = webrtc_streamer(
            key="cybervision-browser-camera",
            mode=WebRtcMode.SENDONLY,
            desired_playing_state=playing,
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=BrowserCameraProcessor,
            video_html_attrs=VideoHTMLAttributes(
                autoPlay=True, controls=False, muted=True, style={"display": "none"}
            ),
            media_toggle_controls=False,
            async_processing=True,
        )
    st.session_state.webrtc_ctx = ctx


# =============================================================================
# MULTI-CAMERA HANDLING & 3x3 MATRIX
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_camera_caps() -> Dict[int, cv2.VideoCapture]:
    caps = {}
    for idx, src in enumerate(CONFIG["CAMERA_SOURCES"]):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[idx] = cap
        else:
            cap.release()
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
            tile = create_blank_tile(tile_w, tile_h, label=f"{cam_key}: NO CAMERA DETECTED")
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

        primary_frame = cv2.resize(frame, (CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]))
        return {0: frame}, primary_frame

    @staticmethod
    def _capture_from_local_devices() -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        caps = get_camera_caps()
        active_frames = {}
        primary_frame = None

        if not caps:
            st.session_state.last_error = "No camera streams open."
            return {}, None

        for idx, cap in list(caps.items()):
            if cap.isOpened():
                for _ in range(2):
                    cap.grab()
                ret, frame = cap.read()
                if ret and frame is not None:
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
            (55, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 170),
            2,
        )

        if subtext:
            cv2.putText(
                frame,
                subtext,
                (55, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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

    def detect(self, frame: np.ndarray, imgsz: Optional[int] = None) -> Dict[str, Any]:
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
                imgsz=imgsz or CONFIG["YOLO_IMAGE_SIZE"],
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
            st.session_state.last_error = f"YOLO detection issue: {exc}"
            return empty

    @staticmethod
    def hazard_severity(yolo_result: Dict[str, Any]) -> str:
        if not yolo_result.get("detected"):
            return "NORMAL"

        highest = "NORMAL"
        for detection in yolo_result.get("detections", []):
            class_name = detection["class"].lower()
            confidence = detection["confidence"]

            if "fire" in class_name:
                if confidence >= 0.60:
                    candidate = "CRITICAL"
                elif confidence >= 0.30:
                    candidate = "HIGH"
                else:
                    candidate = "MEDIUM"
            elif "smoke" in class_name:
                if confidence >= 0.55:
                    candidate = "HIGH"
                else:
                    candidate = "MEDIUM"
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
        elif smoke_ratio >= 0.30:
            severity = "CRITICAL"
            keyword = "visual_smoke_critical"
        elif smoke_ratio >= 0.15:
            severity = "HIGH"
            keyword = "visual_smoke_high"
        elif smoke_ratio >= 0.06:
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
    def _async_worker(frame: np.ndarray, token_budget: int) -> None:
        try:
            image_b64 = frame_to_base64_jpeg(frame)
            response = ollama.chat(
                model=CONFIG["VLM_MODEL"],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze this camera frame specifically for FIRE and SMOKE. "
                            "Return exactly one category: NORMAL, MEDIUM, HIGH, or CRITICAL. "
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

        if st.session_state.get("throttle_active", False):
            return

        st.session_state.vlm_in_progress = True
        token_budget = ResourceGovernor.recommended_token_budget()
        thread = threading.Thread(
            target=VLMInference._async_worker,
            args=(frame.copy(), token_budget),
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
        st.session_state.last_error = "hazard_rules.yar not found."
        return None

    return yara.compile(filepath=CONFIG["YARA_RULE_PATH"])


class YARAVerifier:
    @staticmethod
    def verify(
        vlm_text: str,
        visual_result: Dict[str, Any],
        yolo_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        yolo_result = yolo_result or {}
        yara_severity = "NORMAL"
        matched_rule = "no_yara_match"

        if YARA_AVAILABLE and vlm_text:
            try:
                rules = compile_yara_rules()
                matches = rules.match(data=vlm_text.encode("utf-8", errors="ignore")) if rules else []
                if matches:
                    best_score = -1
                    for match in matches:
                        severity = str(match.meta.get("severity", "NORMAL")).upper()
                        score = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["NORMAL"])["score"]
                        if score > best_score:
                            best_score = score
                            yara_severity = severity
                            matched_rule = match.rule
            except Exception:
                yara_severity = "NORMAL"
                matched_rule = "yara_verification_unavailable"

        yolo_severity = YOLOFireSmokeDetector.hazard_severity(yolo_result)
        visual_severity = visual_result.get("visual_severity", "NORMAL")
        yolo_score = SEVERITY_STYLE[yolo_severity]["score"]
        yara_score = SEVERITY_STYLE[yara_severity]["score"]
        visual_score = SEVERITY_STYLE[visual_severity]["score"]

        final_severity = yolo_severity
        matched_source = "YOLO" if yolo_score > 0 else "YARA"
        matched_keyword = "yolo_detection" if yolo_score > 0 else matched_rule

        if yara_score > SEVERITY_STYLE[final_severity]["score"]:
            final_severity = yara_severity
            matched_source = "YOLO+YARA" if yolo_score > 0 else "YARA"
            matched_keyword = matched_rule

        if not yolo_result.get("model_available", False) and visual_score > SEVERITY_STYLE[final_severity]["score"]:
            final_severity = visual_severity
            matched_source = "VISUAL_FALLBACK"
            matched_keyword = visual_result.get("visual_keyword", "visual_detection")

        is_hazard = final_severity != "NORMAL"

        yolo_conf = float(yolo_result.get("highest_confidence", 0.0) or 0.0)
        if yolo_score > 0 and yolo_conf > 0:
            confidence = min(0.99, max(0.15, yolo_conf))
        else:
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
            "yolo_severity": yolo_severity,
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
    except Exception as exc:
        print(f"[GCP sync] Storage client failed: {exc!r}", flush=True)
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
        except Exception as exc:
            print(f"[GCP sync] Upload of {filename} failed: {exc!r}", flush=True)
            continue


def upload_to_google_cloud_async(frame: np.ndarray, alert: Dict[str, Any]) -> None:
    def _worker():
        save_event_locally(
            time.time(),
            alert.get("severity", "UNKNOWN"),
            alert.get("description", ""),
            camera_id="CAM_01"
        )

        if not GCP_AVAILABLE or storage is None:
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

            image_blob = bucket.blob(image_name)
            image_blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")

            log_blob = bucket.blob(log_name)
            log_blob.upload_from_string(str(alert), content_type="application/json")
        except Exception:
            _queue_image_for_later_upload(frame, alert)

    threading.Thread(target=_worker, daemon=True).start()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def determine_hazard_title(latest: Dict[str, Any]) -> str:
    if not latest or latest.get("severity") == "NORMAL":
        return "Environment is Safe"

    classes = [c.lower() for c in latest.get("yolo_classes", "").split(",") if c.strip()]
    description = latest.get("description", "").lower()
    keyword = str(latest.get("matched_keyword", "")).lower()

    if classes:
        return "Fire Detected" if any("fire" in item for item in classes) else "Smoke Detected"

    has_fire = (
        any("fire" in item for item in classes)
        or "fire" in description)