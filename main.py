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
import uuid
import logging
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# Prevent OpenCV helper operations from competing with the WebRTC encoder.
try:
    cv2.setNumThreads(1)
except Exception:
    pass

LOG = logging.getLogger("cybervision")
CLOUD_STATUS_LOCK = threading.Lock()
CLOUD_STATUS = {"last_success": "", "last_error": "", "last_attempt": "",
                "storage_success": "", "storage_error": "",
                "firestore_success": "", "firestore_error": ""}
CLOUD_SYNC_LOCK = threading.Lock()
YOLO_INFERENCE_LOCK = threading.Lock()
RESOURCE_LOAD_LOCK = threading.Lock()
RESOURCE_LOAD = {"level": 0, "cpu": 0.0, "ram": 0.0, "updated_at": 0.0}
# Google Cloud clients are created only on the Streamlit script thread.
# Background upload threads never call st.secrets / st.cache_resource.
CLOUD_CLIENTS_LOCK = threading.Lock()
CLOUD_CLIENTS = {"storage": None, "firestore": None}


def _update_cloud_status(success: bool, detail: str = "", service: str = "") -> None:
    # Workers must never access st.session_state or st.secrets.
    with CLOUD_STATUS_LOCK:
        key = "last_success" if success else "last_error"
        CLOUD_STATUS[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + (f" — {detail}" if detail else "")
        CLOUD_STATUS["last_attempt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if service in ("storage", "firestore"):
            CLOUD_STATUS[service + ("_success" if success else "_error")] = CLOUD_STATUS[key]
            if success:
                CLOUD_STATUS[service + "_error"] = ""
        # Do not erase an outstanding Firestore failure when Storage succeeds.
        if success and not CLOUD_STATUS["storage_error"] and not CLOUD_STATUS["firestore_error"]:
            CLOUD_STATUS["last_error"] = ""
    if not success:
        LOG.warning("Cloud sync: %s", detail)


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
    """Hosted server (no physical camera) -> use the visitor's browser camera."""
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
    """Hosted file systems are often read-only; fall back to the OS temp dir."""
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
    "CAMERA_SOURCES": [0],  # Primary webcam only; probing 9 devices causes phantom feeds/hangs
    "WEBRTC_ANALYSIS_WIDTH": 320,       # browser-camera frames are analysed at this width
    "WEBRTC_TICK_SEC": 0.75,            # alert/stat tick in browser-camera mode (draws no images)
    "WEBRTC_YOLO_MIN_INTERVAL_SEC": 1.25,  # YOLO stays off the real-time video path
    "WEBRTC_SAMPLE_INTERVAL_SEC": 0.25,  # maximum 4 sampled analysis frames / second
    "WEBRTC_STALL_WARNING_SEC": 6.0,  # show a reconnect action once frames stop
    "YOLO_CPU_THREADS": 2,  # leave CPU capacity for WebRTC video encoding
    "WEBRTC_YOLO_RESULT_TTL_SEC": 3.0,  # boxes older than this are dropped
    "WEBRTC_FRAME_MAX_AGE_SEC": 4.0,    # a frame older than this counts as "no signal"
    "SNAPSHOT_ANALYSIS_WIDTH": 640,
    "CAMERA_CONNECT_HINT_SEC": 15,
    "BUSY_FLAG_TIMEOUT_SEC": 8.0,
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
    "CLOUD_UPLOAD_MIN_INTERVAL_SEC": 10.0,  # bounded cloud writes per hazard episode
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
    "YOLO_CONFIDENCE": 0.22,  # model proposal threshold, then apply per-class thresholds
    "YOLO_FIRE_CONFIDENCE": 0.35,
    "YOLO_SMOKE_CONFIDENCE": 0.25,
    "YOLO_IMAGE_SIZE": 416,  # 160 px removed too much detail for diffuse smoke
    "YOLO_EVERY_N_FRAMES": 8,
    "VISUAL_DETECT_EVERY_N_FRAMES": 4,
}

SEVERITY_STYLE = {
    "NORMAL": {"score": 0, "icon": ":material/check_circle:", "color": "#30d158"},
    "MEDIUM": {"score": 1, "icon": ":material/warning:", "color": "#ffd60a"},
    "HIGH": {"score": 2, "icon": ":material/error:", "color": "#ff9500"},
    "CRITICAL": {"score": 3, "icon": ":material/dangerous:", "color": "#ff3b30"},
}

BROWSER_MODE_LIVE = "Live stream"
BROWSER_MODE_SNAPSHOT = "Snapshot"

VLM_RESULT_QUEUE: queue.Queue = queue.Queue()
YOLO_RESULT_QUEUE: queue.Queue = queue.Queue()

EMPTY_YOLO_RESULT: Dict[str, Any] = {
    "detected": False,
    "detections": [],
    "highest_confidence": 0.0,
    "classes": [],
}


# =============================================================================
# OFFLINE SQLITE BUFFER & GCP FIRESTORE SYNC THREAD
# =============================================================================
def init_db() -> None:
    with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS pending_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL, severity TEXT, description TEXT, camera_id TEXT,
            synced INTEGER DEFAULT 0, event_uuid TEXT
        )""")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_events)")}
        if "event_uuid" not in cols:  # migrate localhost-created databases
            conn.execute("ALTER TABLE pending_events ADD COLUMN event_uuid TEXT")


def save_event_locally(timestamp: float, severity: str, description: str,
                       camera_id: str = "CAM_01", event_uuid: Optional[str] = None,
                       cloud_enabled: bool = True) -> None:
    with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
        conn.execute("""INSERT INTO pending_events
            (timestamp, severity, description, camera_id, synced, event_uuid)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (timestamp, severity, description, camera_id, 0 if cloud_enabled else 1,
             event_uuid or uuid.uuid4().hex))


_WIFI_CHECK_CACHE: Dict[str, Any] = {"checked_at": 0.0, "online": False}
_WIFI_CHECK_TTL_SECONDS = 15.0


def is_wifi_connected(host: str = "storage.googleapis.com", port: int = 443,
                      timeout: float = 1.5) -> bool:
    """Advisory reachability only. NEVER gate cloud writes on a probe."""
    now = time.time()
    if now - _WIFI_CHECK_CACHE["checked_at"] < _WIFI_CHECK_TTL_SECONDS:
        return bool(_WIFI_CHECK_CACHE["online"])
    try:
        with socket.create_connection((host, port), timeout=timeout):
            online = True
    except OSError:
        online = False
    _WIFI_CHECK_CACHE.update(checked_at=now, online=online)
    return online


def get_cloud_runtime_clients() -> Tuple[Optional[Any], Optional[Any]]:
    """Only returns pre-initialized clients. Safe on background worker threads."""
    with CLOUD_CLIENTS_LOCK:
        return CLOUD_CLIENTS["storage"], CLOUD_CLIENTS["firestore"]


def pending_cloud_counts() -> Tuple[int, int]:
    """Number of Firestore rows and complete image/JSON pairs awaiting upload."""
    try:
        with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
            events = int(conn.execute(
                "SELECT COUNT(*) FROM pending_events WHERE synced=0"
            ).fetchone()[0])
    except sqlite3.Error:
        events = 0
    directory = CONFIG["PENDING_UPLOADS_DIR"]
    images = (sum(1 for name in os.listdir(directory) if name.endswith(".jpg")
                  and os.path.isfile(os.path.join(directory, name[:-4] + ".json")))
              if os.path.isdir(directory) else 0)
    return events, images


def sync_cloud_once() -> None:
    """Serialize retries. A failed GCS or Firestore write remains pending."""
    if not CLOUD_SYNC_LOCK.acquire(blocking=False):
        return
    try:
        pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
        has_images = os.path.isdir(pending_dir) and any(
            f.endswith(".jpg") for f in os.listdir(pending_dir))
        with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
            has_events = conn.execute(
                "SELECT EXISTS(SELECT 1 FROM pending_events WHERE synced=0)"
            ).fetchone()[0]
        if not has_images and not has_events:
            return
        if not GCP_AVAILABLE:
            _update_cloud_status(False, "Google Cloud SDK not installed.")
            return
        gcs_client, db_client = get_cloud_runtime_clients()
        if gcs_client is None or db_client is None:
            _update_cloud_status(False, "Cloud clients not ready; check google-cloud-firestore install and [gcp_service_account] in Streamlit secrets.")
            return
        # Event database is separate from GCS image backups: retry either independently.
        with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
            rows = conn.execute("""SELECT id, timestamp, severity, description,
                camera_id, event_uuid FROM pending_events WHERE synced = 0 ORDER BY id LIMIT 30""").fetchall()
            for event_id, ts, sev, desc, cam_id, event_uuid in rows:
                event_uuid = event_uuid or uuid.uuid4().hex
                conn.execute("UPDATE pending_events SET event_uuid = ? WHERE id = ?",
                             (event_uuid, event_id))
                try:
                    db_client.collection("hazard_events").document(event_uuid).set({
                        "event_id": event_uuid, "timestamp": ts, "severity": sev,
                        "description": desc, "camera_id": cam_id,
                        "synced_at": firestore.SERVER_TIMESTAMP,
                    })
                    conn.execute("UPDATE pending_events SET synced = 1 WHERE id = ?", (event_id,))
                    conn.commit()
                    _update_cloud_status(True, "Firestore event uploaded", service="firestore")
                except Exception as exc:
                    _update_cloud_status(False, f"Firestore: {type(exc).__name__}: {exc}", service="firestore")
                    break
        flush_pending_cloud_uploads(gcs_client)
    except Exception as exc:
        _update_cloud_status(False, f"Cloud sync: {type(exc).__name__}: {exc}")
    finally:
        CLOUD_SYNC_LOCK.release()


def sync_worker_loop() -> None:
    while True:
        try:
            sync_cloud_once()
        except Exception as exc:
            LOG.exception("Unexpected cloud worker failure: %s", exc)
        time.sleep(10)

@st.cache_resource(show_spinner=False)
def _start_sync_thread_once() -> bool:
    # A hosted app serves many sessions from one process; start exactly one thread.
    init_db()
    threading.Thread(target=sync_worker_loop, daemon=True, name="cybervision-sync").start()
    return True


def start_sync_thread() -> None:
    """Initialize cloud clients on Streamlit's script thread, then retry in background."""
    configure_cloud_from_secrets()
    gcs, db = get_gcp_clients() if GCP_AVAILABLE else (None, None)
    with CLOUD_CLIENTS_LOCK:
        CLOUD_CLIENTS.update(storage=gcs, firestore=db)
    # The old toggle was removed. Every confirmed hazard is queued regardless
    # of whether authentication or the network is currently available.
    st.session_state.cloud_sync_enabled = True
    if not GCP_AVAILABLE:
        _update_cloud_status(False,
            "Google Cloud SDK incomplete: install google-cloud-firestore, "
            "google-cloud-storage and google-auth.")
    elif gcs is None or db is None:
        LOG.warning("Cloud backup pending: %s", CLOUD_STATUS.get("last_error", "clients unavailable"))
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
        "last_snapshot_rgb": None,
        "_last_snapshot_sig": None,
        "last_resolution": "",
        "frames_processed": 0,
        "frames_received": 0,  # actual WebRTC callback frames, not analysis ticks
        "last_analyzed_frame": 0,
        "inference_count": 0,
        "alert_history": deque(maxlen=CONFIG["ALERT_HISTORY_MAX"]),
        "metrics_history": deque(maxlen=CONFIG["METRICS_HISTORY_MAX"]),
        "latency_history": deque(maxlen=CONFIG["LATENCY_HISTORY_MAX"]),
        "system_status": "STANDBY",
        "vlm_context": "READY",
        "last_error": "",
        "last_public_notice": "",
        # Automatic cloud-first event buffering, independent of connection state.
        # Missing/expired credentials should not silently discard hazard backups.
        "cloud_sync_enabled": True,
        "fps": 0.0,
        "last_frame_time": time.time(),
        "camera_started_at": 0.0,
        "latest_detection": None,
        "last_visual_result": None,
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
        "last_cloud_hazard_type": "",
        "last_pipeline_ts": 0.0,
        "visual_smoke_streak": 0,
        "visual_fire_streak": 0,
        "visual_prev_frame": None,
        "resource_throttle_level": 0,
        "active_hazard_toast": None,
        "active_page": "Dashboard",
        "browser_cam_mode": BROWSER_MODE_LIVE,
        "webrtc_ctx": None,
        "webrtc_connection_id": 0,
        "_frame_processing_busy": False,
        "_frame_processing_busy_since": 0.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Browser-mode selector was removed. Clear a stored Snapshot choice from
    # earlier deployments before render_video_feed() runs on this rerun.
    st.session_state.browser_cam_mode = BROWSER_MODE_LIVE

    if "use_webrtc" not in st.session_state:
        needs_browser_camera = is_cloud_environment() or not local_camera_available()
        st.session_state.use_webrtc = needs_browser_camera and WEBRTC_AVAILABLE
        st.session_state.webrtc_unavailable_on_cloud = needs_browser_camera and not WEBRTC_AVAILABLE


def inject_custom_css() -> None:
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

        /* Keep live elements fully visible while a fragment reruns */
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

        .st-key-cpu_metric_normal [data-testid="stMetricDelta"],
        .st-key-cpu_metric_normal [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_normal [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_normal [data-testid="stMetricDelta"],
        .st-key-ram_metric_normal [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_normal [data-testid="stMetricDelta"] svg {{
            color: #16a34a !important;
            fill: #16a34a !important;
        }}

        .st-key-cpu_metric_medium [data-testid="stMetricDelta"],
        .st-key-cpu_metric_medium [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_medium [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_medium [data-testid="stMetricDelta"],
        .st-key-ram_metric_medium [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_medium [data-testid="stMetricDelta"] svg {{
            color: #eab308 !important;
            fill: #eab308 !important;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricValue"],
        .st-key-ram_metric_high [data-testid="stMetricValue"],
        .st-key-cpu_metric_high [data-testid="stMetricDelta"],
        .st-key-cpu_metric_high [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_high [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_high [data-testid="stMetricDelta"],
        .st-key-ram_metric_high [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_high [data-testid="stMetricDelta"] svg {{
            color: #f59e0b !important;
            fill: #f59e0b !important;
        }}

        .st-key-cpu_metric_critical [data-testid="stMetricValue"],
        .st-key-ram_metric_critical [data-testid="stMetricValue"],
        .st-key-cpu_metric_critical [data-testid="stMetricDelta"],
        .st-key-cpu_metric_critical [data-testid="stMetricDelta"] span,
        .st-key-cpu_metric_critical [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_critical [data-testid="stMetricDelta"],
        .st-key-ram_metric_critical [data-testid="stMetricDelta"] span,
        .st-key-ram_metric_critical [data-testid="stMetricDelta"] svg {{
            color: #dc2626 !important;
            fill: #dc2626 !important;
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

        /* BROWSER CAMERA: the WebRTC <video> IS the live display. The processor
           draws the YOLO boxes + HUD onto the frames, so the browser plays a
           real video stream: no st.image churn, no flicker. */
        .st-key-webrtc_transport {{
            width: 100% !important;
        }}

        .st-key-webrtc_transport video {{
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            border-radius: 14px;
            background: #111827;
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
        ResourceGovernor.update_from_metrics(SystemMonitor.get_metrics())

    @staticmethod
    def update_from_metrics(metrics: Dict[str, Any]) -> int:
        """Update the shared governor for both WebRTC workers and script fragments.

        CPU >75% or RAM >80% -> medium; CPU >90% or RAM >90% ->
        critical. Lower values switch to normal. The governor changes actual
        frame sampling, YOLO size/interval and detection pipeline frequency.
        """
        cpu = float(metrics.get("cpu", 0.0))
        ram = float(metrics.get("ram", 0.0))
        if cpu >= 90 or ram >= 90:
            level = 2
        elif cpu >= 75 or ram >= 80:
            level = 1
        else:
            level = 0
        with RESOURCE_LOAD_LOCK:
            RESOURCE_LOAD.update(level=level, cpu=cpu, ram=ram, updated_at=time.monotonic())
        st.session_state.resource_throttle_level = level
        st.session_state.system_status = "RESOURCE_PRESSURE" if level else "HEALTHY"
        st.session_state.vlm_context = "REDUCED" if level else "ACTIVE"
        st.session_state.last_public_notice = (
            f"Adaptive throttling level {level}: CPU={cpu:.1f}%, RAM={ram:.1f}%. "
            "YOLO, frame sampling and analysis intervals adjusted."
            if level else ""
        )
        return level

    @staticmethod
    def level() -> int:
        with RESOURCE_LOAD_LOCK:
            return int(RESOURCE_LOAD["level"])

    @staticmethod
    def yolo_interval() -> float:
        return (CONFIG["WEBRTC_YOLO_MIN_INTERVAL_SEC"], 2.5, 4.0)[ResourceGovernor.level()]

    @staticmethod
    def pipeline_interval() -> float:
        return (0.75, 1.50, 2.5)[ResourceGovernor.level()]

    @staticmethod
    def yolo_image_size() -> int:
        # Each throttle level must use fewer pixels, not more.
        # Ultralytics rounds to a suitable stride internally.
        return (CONFIG["YOLO_IMAGE_SIZE"], 352, 320)[ResourceGovernor.level()]

    @staticmethod
    def recommended_token_budget() -> int:
        return (1024, 256, 128)[ResourceGovernor.level()]

def draw_browser_camera_hud(frame_bgr: np.ndarray, fps: float, yolo_detected: bool,
                            inference_count: int, severity: str,
                            throttle_level: int) -> np.ndarray:
    """Draw the live HUD in WebRTC frames. Never uses Streamlit session state.

    Small ROI-only alpha blend + 3 lines of text; safe for 480x360 browser
    video. Unlike an HTML overlay, this is displayed inside the returned feed.
    """
    height, width = frame_bgr.shape[:2]
    panel_w, panel_h = min(width - 4, 310), min(height - 4, 78)
    if panel_w <= 0 or panel_h <= 0:
        return frame_bgr
    x0, y0 = 3, 3
    roi = frame_bgr[y0:y0 + panel_h, x0:x0 + panel_w]
    dim = np.zeros_like(roi)
    cv2.addWeighted(dim, 0.60, roi, 0.40, 0, dst=roi)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), (255, 220, 0), 1)
    cv2.line(frame_bgr, (x0 + 2, y0 + panel_h - 3),
             (x0 + panel_w - 3, y0 + panel_h - 3), (255, 230, 0), 1)

    severity = str(severity).upper() if severity else "NORMAL"
    sev_color = ((0, 0, 255) if severity == "CRITICAL" else
                 (0, 175, 255) if severity == "HIGH" else
                 (0, 255, 255) if severity == "MEDIUM" else (0, 255, 0))
    cv2.putText(frame_bgr, f"FPS: {fps:.1f} | YOLO: {'DETECTED' if yolo_detected else 'CLEAR'}",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"INF: {inference_count} | STATUS: {severity}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, .43, sev_color, 1, cv2.LINE_AA)
    throttle_name = ('NORMAL', 'MEDIUM', 'CRITICAL')[max(0, min(2, int(throttle_level)))]
    throttle_color = ((0, 255, 0), (0, 255, 255), (0, 0, 255))[max(0, min(2, int(throttle_level)))]
    cv2.putText(frame_bgr, f"THROTTLE: L{throttle_level} ({throttle_name})",
                (10, 63), cv2.FONT_HERSHEY_SIMPLEX, .40, throttle_color, 1, cv2.LINE_AA)
    return frame_bgr


# =============================================================================
# BROWSER CAMERA (WEBRTC)
# =============================================================================
if WEBRTC_AVAILABLE:
    class BrowserCameraProcessor(VideoProcessorBase):
        """Keep WebRTC transmission independent of expensive AI inference.

        recv() is called for every video frame, but converts pixels only for an
        occasional detection sample, and hands that sample to a bounded worker.
        Frames without detection boxes are returned unmodified. This avoids
        constant PyAV frame creation, OpenCV work and extra CPU/memory pressure.
        No Streamlit calls occur on the WebRTC worker threads.
        """

        def __init__(self, detector=None) -> None:
            self._lock = threading.Lock()
            self._detector = detector
            self._latest_frame: Optional[np.ndarray] = None
            self._last_frame_ts = 0.0
            self._last_sample_ts = 0.0
            self._last_sample_attempt_ts = 0.0
            self.frames_received = 0
            self._yolo_result: Optional[Dict[str, Any]] = None
            self._yolo_result_ts = 0.0
            self._yolo_busy = False
            self._last_yolo_start = 0.0
            self._fps = 0.0
            self._last_tick = 0.0
            self._ended = False
            self._analysis_severity = "NORMAL"
            self._inference_count = 0
            self._last_decode_error_log = 0.0
            self._last_render_error_log = 0.0

        def _yolo_worker(self, img: np.ndarray) -> None:
            try:
                result = self._detector.detect(img)
            except Exception as exc:
                result = {**EMPTY_YOLO_RESULT, "error": str(exc)}
            finally:
                # Even if inference raises, allow the next scheduled sample.
                with self._lock:
                    if "result" in locals():
                        self._yolo_result = result
                        self._yolo_result_ts = time.monotonic()
                        self._inference_count += 1
                        if result.get("detected"):
                            self._analysis_severity = YOLOFireSmokeDetector.hazard_severity(result)
                    self._yolo_busy = False

        def set_analysis_severity(self, severity: str) -> None:
            """Called by Streamlit pipeline, not by a WebRTC worker."""
            with self._lock:
                self._analysis_severity = severity if severity in SEVERITY_STYLE else "NORMAL"

        def recv(self, frame):
            now = time.monotonic()
            # Apply adaptive sampling, not just YOLO throttling, when CPU/RAM is high.
            sample_interval = CONFIG["WEBRTC_SAMPLE_INTERVAL_SEC"] * (1, 2, 4)[ResourceGovernor.level()]
            with self._lock:
                previous = self._last_tick
                self._last_tick = now
                self._last_frame_ts = now
                self.frames_received += 1
                self._ended = False
                if previous and now > previous:
                    instant_fps = min(120.0, 1.0 / (now - previous))
                    self._fps = instant_fps if not self._fps else (0.85 * self._fps + 0.15 * instant_fps)
                needs_sample = now - self._last_sample_attempt_ts >= sample_interval
                if needs_sample:
                    self._last_sample_attempt_ts = now
                recent = now - self._yolo_result_ts <= CONFIG["WEBRTC_YOLO_RESULT_TTL_SEC"]
                yolo = self._yolo_result if recent else None
                hud_severity = self._analysis_severity
                inference_count = self._inference_count

            sample = None
            if needs_sample:
                try:
                    # Decode to ndarray only 4 times/s, not for every WebRTC frame.
                    sample = frame.to_ndarray(format="bgr24")
                except Exception as exc:
                    sample = None
                    if now - self._last_decode_error_log >= 15.0:
                        LOG.warning("WebRTC sample decode error: %s: %s", type(exc).__name__, exc)
                        self._last_decode_error_log = now

            start_yolo = False
            if sample is not None:
                with self._lock:
                    self._latest_frame = sample
                    # Sample timestamp reflects when the video was received, not
                    # when the Streamlit analysis fragment next polls it.
                    self._last_sample_ts = now
                    if (
                        self._detector is not None and self._detector.available
                        and not self._yolo_busy
                        and now - self._last_yolo_start >= ResourceGovernor.yolo_interval()
                    ):
                        self._yolo_busy = True
                        self._last_yolo_start = now
                        start_yolo = True
                if start_yolo:
                    try:
                        threading.Thread(target=self._yolo_worker, args=(sample,),
                                         daemon=True, name="cybervision-yolo").start()
                    except RuntimeError as exc:
                        # Resource-constrained hosts can reject new threads.
                        # Preserve the WebRTC stream even when YOLO cannot start.
                        with self._lock:
                            self._yolo_busy = False
                        LOG.warning("YOLO background thread unavailable: %s", exc)

            # Always render the status HUD, including between YOLO detections.
            # This is a lightweight 480x360 frame conversion; YOLO itself remains
            # on the bounded background worker and the WebRTC frame is still sent
            # if a render error occurs.
            try:
                img = sample if sample is not None else frame.to_ndarray(format="bgr24")
                out = img.copy()  # sampled image also belongs to asynchronous YOLO
                if yolo and yolo.get("detections"):
                    out = YOLOFireSmokeDetector.draw_boxes_on_frame(out, yolo)
                out = draw_browser_camera_hud(
                    out, self._fps, bool(yolo and yolo.get("detected")),
                    inference_count, hud_severity, ResourceGovernor.level()
                )
                new_frame = av.VideoFrame.from_ndarray(out, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                return new_frame
            except Exception as exc:
                if now - self._last_render_error_log >= 15.0:
                    LOG.warning("WebRTC HUD/box overlay error: %s: %s", type(exc).__name__, exc)
                    self._last_render_error_log = now
                return frame

        def snapshot(self):
            """Copy a recent sample and its YOLO result for Streamlit analysis."""
            with self._lock:
                now = time.monotonic()
                if (
                    self._ended or self._latest_frame is None
                    or now - self._last_sample_ts > CONFIG["WEBRTC_FRAME_MAX_AGE_SEC"]
                    or now - self._last_frame_ts > CONFIG["WEBRTC_FRAME_MAX_AGE_SEC"]
                ):
                    return None, None, 0.0
                fresh = now - self._yolo_result_ts <= CONFIG["WEBRTC_YOLO_RESULT_TTL_SEC"]
                result = self._yolo_result if fresh else None
                return self._latest_frame.copy(), result, self._fps

        def health(self) -> Dict[str, Any]:
            with self._lock:
                return {
                    "frames_received": self.frames_received,
                    "age_sec": (time.monotonic() - self._last_frame_ts)
                               if self._last_frame_ts else None,
                    "ended": self._ended,
                    "yolo_busy": self._yolo_busy,
                    "inference_count": self._inference_count,
                    "severity": self._analysis_severity,
                }

        def on_ended(self) -> None:
            # Streamlit user state is inaccessible from WebRTC callback threads.
            with self._lock:
                self._ended = True
                self._latest_frame = None
                self._yolo_result = None
            LOG.info("WebRTC track ended after %s received frames", self.frames_received)

else:
    BrowserCameraProcessor = None


def _get_browser_processor():
    """Find the live processor (the factory records it in a per-session dict)."""
    processor = (st.session_state.get("_cam_holder") or {}).get("proc")
    if processor is None:
        ctx = st.session_state.get("webrtc_ctx")
        processor = getattr(ctx, "video_processor", None)
    return processor


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
    """Read a *service account*, not the Google OAuth login client in [auth.google].

    Supports GCP_SERVICE_ACCOUNT_JSON (JSON string or TOML inline table),
    [gcp_service_account], [GCP_SERVICE_ACCOUNT], [service_account], and
    common alternative names. Do not log or display service-account keys.
    """
    raw = _get_secret("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            value = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if isinstance(value, dict):
                return _normalize_service_account(value)
        except (ValueError, TypeError) as exc:
            _update_cloud_status(False,
                f"GCP_SERVICE_ACCOUNT_JSON is not valid JSON/table: {type(exc).__name__}")
            return None
    try:
        for key in ("gcp_service_account", "GCP_SERVICE_ACCOUNT", "service_account",
                    "google_service_account", "google_cloud_service_account", "gcp"):
            if key in st.secrets:
                info = st.secrets[key]
                if isinstance(info, str):
                    info = json.loads(info)
                return _normalize_service_account(dict(info))
        # Flat service account TOML fields are supported too.
        if "private_key" in st.secrets and "client_email" in st.secrets:
            return _normalize_service_account(dict(st.secrets))
    except Exception as exc:
        _update_cloud_status(False,
            f"Cannot parse GCP service account section: {type(exc).__name__}")
        return None

    # Optional local-dev fallback; never commit the JSON file to a public repo.
    local_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service_account.json")
    if os.path.isfile(local_key_path):
        try:
            with open(local_key_path, encoding="utf-8") as keyfile:
                return _normalize_service_account(json.load(keyfile))
        except Exception as exc:
            _update_cloud_status(False,
                f"Cannot read local service account file: {type(exc).__name__}")
    return None


def _normalize_service_account(info: dict) -> dict:
    result = dict(info)
    key = result.get("private_key")
    if isinstance(key, str):
        result["private_key"] = key.replace("\\n", "\n")
    return result


def get_gcp_clients() -> Tuple[Optional[Any], Optional[Any]]:
    """Construct clients using service account secrets or deployed ADC.

    Do not cache failed initialization: installing a missing secret followed by
    a deployment restart must allow a clean retry. Background threads use
    get_cloud_runtime_clients() rather than consulting Streamlit secrets.
    """
    if not GCP_AVAILABLE:
        _update_cloud_status(False,
            "Missing cloud SDK. Add google-cloud-firestore, google-cloud-storage, google-auth to requirements.txt")
        return None, None
    # Reuse clients across Streamlit reruns; cloud threads use the same clients.
    cached_storage, cached_firestore = get_cloud_runtime_clients()
    if cached_storage is not None and cached_firestore is not None:
        return cached_storage, cached_firestore
    info = None
    try:
        info = _get_gcp_service_account_info()
        project = None
        credentials = None
        if info is not None:
            required = ("client_email", "private_key", "token_uri")
            missing = [key for key in required if not info.get(key)]
            if missing:
                raise ValueError("GCP service account is missing: " + ", ".join(missing))
            credentials = gcp_service_account.Credentials.from_service_account_info(info)
            project = info.get("project_id") or _get_secret("GCP_PROJECT_ID")
        else:
            # Application Default Credentials are only expected on Google Cloud
            # deployments or when an explicit credential path was configured.
            # Avoid an unnecessary metadata-server lookup on Streamlit Cloud.
            adc_configured = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                                  or os.environ.get("K_SERVICE")
                                  or os.environ.get("GAE_ENV"))
            if not adc_configured:
                raise RuntimeError(
                    "No GCP service account configured. Google login [auth.google] "
                    "does not grant Cloud Storage/Firestore access; add [gcp_service_account] "
                    "to Streamlit Secrets.")
            import google.auth
            credentials, detected_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            project = _get_secret("GCP_PROJECT_ID") or detected_project

        if not project:
            raise ValueError("GCP_PROJECT_ID or service-account project_id is required")
        # No hardcoded placeholder project ID: the credential's project is authoritative.
        clients = (storage.Client(credentials=credentials, project=project),
                   firestore.Client(credentials=credentials, project=project))
        return clients
    except Exception as exc:
        # Do not include private-key values or the entire credentials object.
        reason = f"GCP credential/client initialization failed: {type(exc).__name__}: {exc}"
        _update_cloud_status(False, reason)
        return None, None


def get_cloud_bucket_name() -> str:
    """Called by worker; bucket name resolved from secrets on script thread."""
    return str(CONFIG["GCP_BUCKET"]).strip()


def configure_cloud_from_secrets() -> None:
    """Load cloud settings only on Streamlit's script thread.

    Bucket may be at TOML root or alongside credentials in [gcp_service_account],
    so users can append a complete table after their existing [auth.google].
    """
    bucket = _get_secret("GCP_BUCKET") or _get_secret("GCS_BUCKET")
    if not bucket:
        try:
            for table_name in ("gcp_service_account", "GCP_SERVICE_ACCOUNT", "gcp"):
                table = st.secrets.get(table_name)
                if table:
                    bucket = table.get("bucket") or table.get("gcp_bucket")
                    if bucket:
                        break
        except Exception:
            pass
    if bucket:
        CONFIG["GCP_BUCKET"] = str(bucket).strip()

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
        payload = resp.json()
        ice_servers = payload.get("iceServers") if isinstance(payload, dict) else payload
        if isinstance(ice_servers, dict):
            ice_servers = [ice_servers]
        ice_servers = ice_servers or []

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
    """Mounts the WebRTC connector. Its <video> is the live, annotated display.

    SENDRECV: the browser sends its camera, the server-side processor draws the
    detections on each frame and sends the annotated video straight back.
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
    holder = st.session_state.setdefault("_cam_holder", {})

    # Create the detector here, on the script thread, and hand it to the
    # processor (the factory itself runs in a WebRTC worker thread).
    detector = st.session_state.get("yolo_detector")
    if detector is None:
        detector = YOLOFireSmokeDetector()
        st.session_state.yolo_detector = detector

    def _processor_factory():
        proc = BrowserCameraProcessor(detector)
        holder["proc"] = proc
        return proc

    with st.container(key="webrtc_transport"):
        ctx = webrtc_streamer(
            key=f"cybervision-browser-camera-v3-{st.session_state.get('webrtc_connection_id', 0)}",
            mode=WebRtcMode.SENDRECV,
            desired_playing_state=True,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480, "max": 480},
                    "height": {"ideal": 360, "max": 360},
                    "frameRate": {"ideal": 12, "max": 15},
                },
                "audio": False,
            },
            video_processor_factory=_processor_factory,
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
                "style": {
                    "width": "100%",
                    "maxWidth": "100%",
                    "borderRadius": "14px",
                },
            },
            async_processing=True,
        )

    st.session_state.webrtc_ctx = ctx

    state = getattr(ctx, "state", None)
    playing = bool(state is not None and getattr(state, "playing", False))
    ice_note = _describe_ice_servers(ice_servers)

    if ice_note.startswith(":material/info:") and not playing:
        st.warning(
            "No TURN relay is configured, so a hosted app usually cannot connect to your "
            "browser camera. Add CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN "
            "(or METERED_API_KEY + METERED_DOMAIN) to your Streamlit secrets and reboot the app.",
            icon=":material/warning:",
        )


# =============================================================================
# LOCAL CAMERA HANDLING & 3x3 MATRIX (localhost behaviour of the working project)
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
    def capture_local_frames() -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
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

        cv2.putText(frame, message, (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 170), 2)

        if subtext:
            cv2.putText(frame, subtext, (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

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
        # Torch otherwise tends to use every available CPU thread for YOLO,
        # starving browser WebRTC frame processing on small hosted instances.
        try:
            import torch
            torch.set_num_threads(int(CONFIG["YOLO_CPU_THREADS"]))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # Inter-op threads can be configured only once.
        except Exception as thread_exc:
            LOG.debug("Optional torch CPU thread limit unavailable: %s", thread_exc)
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
            # Ultralytics shares predictor state; protect the cached model across sessions.
            with YOLO_INFERENCE_LOCK:
                results = self.model(
                    frame,
                    imgsz=ResourceGovernor.yolo_image_size(),
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

                    # Model labels outside fire/smoke must not create hazards.
                    if "fire" in class_name:
                        threshold = CONFIG["YOLO_FIRE_CONFIDENCE"]
                    elif "smoke" in class_name:
                        threshold = CONFIG["YOLO_SMOKE_CONFIDENCE"]
                    else:
                        continue
                    if confidence < threshold:
                        continue
                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": [int(v) for v in xyxy],
                    })

            # Boxes are drawn only by the video renderer, not during inference.
            # This saves an OpenCV copy/draw per inference.
            return {
                "detected": bool(detections),
                "detections": detections,
                "highest_confidence": (
                    max(d["confidence"] for d in detections)
                    if detections else 0.0
                ),
                "classes": sorted(set(d["class"] for d in detections)),
                "annotated_frame": frame,
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
    def detect(frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {
                "visual_severity": "NORMAL",
                "visual_keyword": "",
                "fire_ratio": 0.0,
                "smoke_ratio": 0.0,
                "smoke_motion_ratio": 0.0,
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
        # A stationary gray wall also passes the color mask. Prefer moving gray
        # plumes in live mode rather than making a hazardous wall-only alert.
        smoke_motion_ratio = 0.0
        if previous_frame is not None and previous_frame.shape == frame.shape:
            diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY),
            )
            moving_smoke = (diff > 12) & smoke_pixels
            smoke_motion_ratio = float(np.count_nonzero(moving_smoke)) / max(1, frame.shape[0] * frame.shape[1])

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
            "smoke_motion_ratio": smoke_motion_ratio,
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

        # "localhost" Ollama only exists on your own machine. On a hosted
        # deployment set OLLAMA_HOST (env var or secret) to a reachable server;
        # without it the VLM step is skipped and detection continues with YOLO.
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
            "visual_keyword": visual_result.get("visual_keyword", ""),
            "fire_ratio": visual_result.get("fire_ratio", 0.0),
            "smoke_ratio": visual_result.get("smoke_ratio", 0.0),
            "smoke_motion_ratio": visual_result.get("smoke_motion_ratio", 0.0),
            "yara_available": YARA_AVAILABLE,
        }


# =============================================================================
# GOOGLE CLOUD FORENSIC BACKUP & LOCAL BUFFER
# =============================================================================
def _queue_image_for_later_upload(frame: np.ndarray, alert: Dict[str, Any]) -> bool:
    """Persist JPEG and JSON before handing control to a daemon thread."""
    pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
    try:
        os.makedirs(pending_dir, exist_ok=True)
        event_uuid = str(alert["event_id"])
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise ValueError("OpenCV JPEG encoding failed")
        image_path = os.path.join(pending_dir, f"{event_uuid}.jpg")
        meta_path = os.path.join(pending_dir, f"{event_uuid}.json")
        with open(image_path + ".tmp", "wb") as f:
            f.write(buffer.tobytes())
        os.replace(image_path + ".tmp", image_path)
        with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(alert, f, default=str)
        os.replace(meta_path + ".tmp", meta_path)
        return True
    except Exception as exc:
        _update_cloud_status(False, f"Local backup queue: {type(exc).__name__}: {exc}")
        return False


def flush_pending_cloud_uploads(client=None) -> None:
    if not GCP_AVAILABLE:
        return
    pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
    if not os.path.isdir(pending_dir):
        return
    if client is None:
        client, _ = get_cloud_runtime_clients()
    if client is None:
        _update_cloud_status(False, "Cloud Storage client not initialized", service="storage")
        return
    bucket = client.bucket(get_cloud_bucket_name())
    for filename in sorted(os.listdir(pending_dir)):
        if not filename.endswith(".jpg"):
            continue
        event_uuid = filename[:-4]
        image_path = os.path.join(pending_dir, filename)
        meta_path = os.path.join(pending_dir, event_uuid + ".json")
        # We only upload complete pairs to avoid a partially-written metadata file.
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            with open(meta_path, "rb") as f:
                meta_bytes = f.read()
            # Stable object keys + retry-safe overwrites.
            bucket.blob(f"cybervision_alerts/{event_uuid}.jpg").upload_from_string(
                image_bytes, content_type="image/jpeg", timeout=20)
            bucket.blob(f"cybervision_alerts/{event_uuid}.json").upload_from_string(
                meta_bytes, content_type="application/json", timeout=20)
            os.remove(image_path)
            os.remove(meta_path)
            _update_cloud_status(True, "Storage image and metadata uploaded", service="storage")
        except Exception as exc:
            _update_cloud_status(False, f"Cloud Storage: {type(exc).__name__}: {exc}", service="storage")
            # Keep both files for the next retry.
            break


def upload_to_google_cloud_async(frame: np.ndarray, alert: Dict[str, Any],
                                 cloud_sync_enabled: bool) -> None:
    """Durably queue a confirmed hazard; retry even after temporary auth failures."""
    event_uuid = uuid.uuid4().hex
    event = {**alert, "event_id": event_uuid, "camera_id": "CAM_01"}
    try:
        save_event_locally(time.time(), event.get("severity", "UNKNOWN"),
                           event.get("description", ""), event_uuid=event_uuid,
                           cloud_enabled=True)
    except Exception as exc:
        _update_cloud_status(False, f"SQLite event queue: {type(exc).__name__}: {exc}")
    # Queue image *even when credentials are not configured yet*. Previously
    # "cloud_sync_enabled=False" lost the image and marked the event synced.
    if not _queue_image_for_later_upload(frame, event):
        return
    gcs, db = get_cloud_runtime_clients()
    if gcs is None or db is None:
        LOG.warning("Cloud client not ready; event %s held in local retry queue", event_uuid)
        return
    threading.Thread(target=sync_cloud_once, daemon=True,
                     name="cybervision-upload").start()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def determine_hazard_title(latest: Dict[str, Any]) -> str:
    if not latest or latest.get("severity") == "NORMAL":
        return "Environment is Safe"
    hazard_type = str(latest.get("hazard_type", "")).upper()
    if hazard_type == "FIRE":
        return "Fire Detected"
    if hazard_type == "SMOKE":
        return "Smoke Detected"
    # Fallback for existing records.
    candidates = str(latest.get("yolo_classes", "")).lower()
    keyword = str(latest.get("matched_keyword", "")).lower()
    return "Fire Detected" if ("fire" in candidates or "fire" in keyword) else "Smoke Detected"


def resolve_hazard(yolo_result: Dict[str, Any], verdict: Dict[str, Any],
                   smoke_streak: int = 0, fire_streak: int = 0,
                   snapshot: bool = False) -> Tuple[str, str, float, str]:
    """Return (severity, type, confidence, source).

    YOLO is primary. Visual HSV heuristics are a fallback requiring 2 samples
    for live scenes, to reduce alerts from stationary gray backgrounds.
    """
    detections = yolo_result.get("detections", [])
    yolo_items = []
    for det in detections:
        label = str(det.get("class", "")).lower()
        confidence = float(det.get("confidence", 0))
        if "fire" in label and confidence >= CONFIG["YOLO_FIRE_CONFIDENCE"]:
            severity = "CRITICAL" if confidence >= 0.50 else "HIGH"
            yolo_items.append((SEVERITY_STYLE[severity]["score"], confidence, "FIRE", severity))
        elif "smoke" in label and confidence >= CONFIG["YOLO_SMOKE_CONFIDENCE"]:
            severity = "HIGH" if confidence >= 0.30 else "MEDIUM"
            yolo_items.append((SEVERITY_STYLE[severity]["score"], confidence, "SMOKE", severity))
    if yolo_items:
        _, confidence, hazard_type, severity = max(yolo_items)
        return severity, hazard_type, confidence, "YOLO_DETECTOR"

    visual_sev = verdict.get("visual_severity", "NORMAL")
    visual_type = str(verdict.get("visual_keyword", verdict.get("matched_keyword", ""))).lower()
    if visual_sev != "NORMAL":
        is_fire = "fire" in visual_type
        is_smoke = "smoke" in visual_type
        confirmed = snapshot or (fire_streak >= 2 if is_fire else smoke_streak >= 2)
        if is_smoke and not snapshot:
            # Avoid treating unmoving gray scenery as smoke in a live video.
            confirmed = confirmed and float(verdict.get("smoke_motion_ratio", 0.0)) >= 0.012
        if confirmed and (is_fire or is_smoke):
            return visual_sev, "FIRE" if is_fire else "SMOKE", float(verdict.get("confidence", 0.0)), "OPENCV_VISUAL_FALLBACK"
    return "NORMAL", "", 0.0, "SYSTEM"

# =============================================================================
# DETECTION PIPELINE
# =============================================================================
def run_detection_pipeline(
    frame: np.ndarray,
    sync: bool = False,
    yolo_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full detection pipeline on one frame.

    yolo_override: YOLO result already computed elsewhere (the browser-camera
        video processor); the pipeline then skips its own YOLO call.
    sync=True: snapshot photo, run YOLO right now.
    otherwise: YOLO runs in a background thread and is picked up on a later tick.
    """
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

    if yolo_override is not None:
        yolo_result = yolo_override
        st.session_state.yolo_detection = yolo_result
    elif sync:
        yolo_result = yolo_detector.detect(frame)
        st.session_state.yolo_detection = yolo_result
        st.session_state.inference_count += 1
        err = yolo_result.get("error")
        if err:
            st.session_state.last_error = err
    else:
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


    run_visual_now = (
        sync
        or st.session_state.get("last_visual_result") is None
        or frame_number % max(1, CONFIG["VISUAL_DETECT_EVERY_N_FRAMES"] *
                               (1 + ResourceGovernor.level())) == 0
    )
    if run_visual_now:
        visual_result = VisualFireSmokeDetector.detect(
            frame, st.session_state.get("visual_prev_frame") if not sync else None
        )
        st.session_state.visual_prev_frame = frame.copy()
        st.session_state.last_visual_result = visual_result
    else:
        visual_result = st.session_state.last_visual_result

    visual_keyword = str(visual_result.get("visual_keyword", ""))
    if run_visual_now:
        st.session_state.visual_smoke_streak = (
            st.session_state.get("visual_smoke_streak", 0) + 1 if "smoke" in visual_keyword else 0
        )
        st.session_state.visual_fire_streak = (
            st.session_state.get("visual_fire_streak", 0) + 1 if "fire" in visual_keyword else 0
        )
    candidate_hazard = bool(yolo_result.get("detected")) or visual_result.get("visual_severity") != "NORMAL"
    analyze_this_frame = candidate_hazard and (
        sync or frame_number % max(1, CONFIG["ANALYZE_EVERY_N_FRAMES"]) == 0
    )

    if analyze_this_frame:
        VLMInference.trigger_async_inference(frame)

    vlm_text = st.session_state.get("latest_vlm_text", "")
    if not vlm_text:
        vlm_text = "NORMAL: Monitoring active."

    verdict = YARAVerifier.verify(vlm_text if candidate_hazard else "", visual_result)
    severity, hazard_type, confidence, hazard_source = resolve_hazard(
        yolo_result, verdict,
        st.session_state.get("visual_smoke_streak", 0),
        st.session_state.get("visual_fire_streak", 0),
        snapshot=sync,
    )
    st.session_state.yolo_severity = severity  # existing dashboard metric name

    latency = time.time() - start_time
    st.session_state.latency_history.append(latency)

    public_description = vlm_text
    if yolo_result.get("detected"):
        detection_text = ", ".join(f"{d['class']} {d['confidence']:.0%}" for d in yolo_result["detections"])
        public_description = f"YOLO detected: {detection_text}. VLM: {vlm_text}"

    is_hazard = severity != "NORMAL"
    if is_hazard and hazard_source == "OPENCV_VISUAL_FALLBACK":
        public_description = (f"Visual analysis suggests {hazard_type.lower()} "
                              f"({visual_result.get('visual_keyword', 'visual')}). "
                              f"YOLO: no matching object. VLM: {vlm_text}")

    alert = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "description": public_description,
        "success": True,
        "latency": latency,
        "verified": hazard_source == "YOLO_DETECTOR",
        "yolo_detected": bool(yolo_result.get("detected")),
        "yolo_classes": ", ".join(yolo_result.get("classes", [])),
        "yolo_confidence": confidence,
        "severity": severity,
        "hazard_type": hazard_type,
        "is_hazard": is_hazard,
        "confidence": confidence,
        "matched_keyword": verdict.get("matched_keyword", "none"),
        "matched_source": hazard_source,
        "yara_severity": verdict.get("yara_severity", "NORMAL"),
        "visual_severity": visual_result.get("visual_severity", "NORMAL"),
        "fire_ratio": visual_result.get("fire_ratio", 0.0),
        "smoke_ratio": visual_result.get("smoke_ratio", 0.0),
        "smoke_motion_ratio": visual_result.get("smoke_motion_ratio", 0.0),
        "yara_available": YARA_AVAILABLE,
    }

    st.session_state.alert_history.append(alert)
    st.session_state.latest_detection = alert

    if is_hazard:
        hazard_title = determine_hazard_title(alert)
        dispatch_hazard_alerts(severity, hazard_title)

    play_critical_alarm(severity == "CRITICAL")

    # Throttle uploads: otherwise every tick writes a SQLite row + Firestore doc.
    if is_hazard:
        now = time.time()
        # New hazard type is a new event even within the upload cool-down.
        if (now - st.session_state.get("last_cloud_upload_ts", 0.0) >= CONFIG["CLOUD_UPLOAD_MIN_INTERVAL_SEC"]
                or hazard_type != st.session_state.get("last_cloud_hazard_type", "")):
            st.session_state.last_cloud_upload_ts = now
            st.session_state.last_cloud_hazard_type = hazard_type
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


def _metric_status_class(value: float) -> str:
    if value >= 90:
        return "critical"
    if value >= 70:
        return "medium"
    return "normal"


def _metric_delta_icon(value: float) -> str:
    if value >= 90:
        return ":material/error:"
    if value >= 70:
        return ":material/warning:"
    return ":material/check_circle:"


def render_dashboard_settings_panel() -> None:
    with st.container():
        st.markdown("<div class='stats-panel-title'>Server Setup</div>", unsafe_allow_html=True)

        # Hidden controls intentionally removed from the UI per design request:
        # - Browser camera mode
        # - Cloud/local backup toggle
        # - Inference throttle caption
        # Keep the underlying behavior enabled with sensible defaults.
        st.session_state.browser_cam_mode = BROWSER_MODE_LIVE

        st.session_state.sound_enabled = st.toggle(
            "Alert sound",
            value=st.session_state.sound_enabled,
        )

        # Cloud backup is automatic and configured in Streamlit secrets. Keep
        # diagnostics in server logs, not the Server Setup UI.

        timeout_option = st.selectbox(
            "Inference timeout target",
            ["1.0s", "2.0s", "3.0s", "5.0s"],
            index=1,
            key="inference_timeout_select"
        )
        CONFIG["INFERENCE_TIMEOUT"] = float(timeout_option.replace("s", ""))

        render_resource_metrics_panel()


def _render_resource_metrics_body() -> None:
    st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='stats-panel-title'>Resources</div>", unsafe_allow_html=True)

    metrics = SystemMonitor.get_metrics()
    # Update the global governor here too, including when pipeline ticks are
    # being skipped under pressure. This propagates to video worker threads.
    if st.session_state.get("camera_running", False):
        ResourceGovernor.update_from_metrics(metrics)
    cpu_class = _metric_status_class(metrics["cpu"])
    ram_class = _metric_status_class(metrics["ram"])
    cpu_delta = _metric_delta_icon(metrics["cpu"])
    ram_delta = _metric_delta_icon(metrics["ram"])

    m1, m2 = st.columns(2)
    with m1:
        with st.container(key=f"cpu_metric_{cpu_class}"):
            st.metric("CPU", f"{metrics['cpu']:.1f}%", cpu_delta)
    with m2:
        with st.container(key=f"ram_metric_{ram_class}"):
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


def render_resource_metrics_panel() -> None:
    # Fragments refresh metrics without remounting WebRTC or the setup controls.
    refresh = 3.0 if st.session_state.get("camera_running", False) else None
    st.fragment(_render_resource_metrics_body, run_every=refresh)()


# DASHBOARD FUNCTIONS RESTORED AFTER SERVER SETUP REFACTOR
def _browser_snapshot_mode() -> bool:
    return bool(
        st.session_state.get("use_webrtc")
        and st.session_state.get("browser_cam_mode") == BROWSER_MODE_SNAPSHOT
    )


def render_dashboard_header() -> None:
    cam_running = st.session_state.camera_running or _browser_snapshot_mode()
    yolo_ready = is_yolo_model_available()

    if cam_running and yolo_ready:
        pill_text, pill_color = "YOLO ACTIVE", "#16a34a"
    elif cam_running:
        pill_text, pill_color = "FALLBACK MODE", "#ca8a04"
    else:
        pill_text, pill_color = "STANDBY", "#6b7280"

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
        frames = (st.session_state.get("frames_received", 0)
                  if st.session_state.get("use_webrtc")
                  else st.session_state.frames_processed)
        r2c1.metric("Frames", int(frames))
        r2c2.metric("Hazards", hazards, f"of {total}")



def render_live_stats_panel() -> None:
    refresh = 1.0 if st.session_state.get("camera_running") else None
    st.fragment(_render_live_stats_body, run_every=refresh)()


def _render_video_status_bar_body() -> None:
    fps = st.session_state.get("fps", 0.0)
    avg_latency = (
        sum(st.session_state.latency_history) / len(st.session_state.latency_history)
        if st.session_state.latency_history else 0.0
    )

    resolution = st.session_state.get("last_resolution") or ""
    if not resolution:
        last_rgb = st.session_state.get("last_frame_rgb")
        if last_rgb is not None:
            resolution = f"{last_rgb.shape[1]}x{last_rgb.shape[0]}"
        else:
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


def render_video_status_bar() -> None:
    # Its own fragment, so it refreshes on its own while the camera runs.
    refresh = 1.0 if st.session_state.get("camera_running") else None
    st.fragment(_render_video_status_bar_body, run_every=refresh)()


def render_resource_trend_preview() -> None:
    metrics_df = pd.DataFrame(list(st.session_state.metrics_history))
    if metrics_df.empty:
        st.info("No resource metrics yet.")
        return
    metrics_df = metrics_df.set_index("time")
    st.line_chart(metrics_df[["cpu", "ram"]], height=180)


# =============================================================================
# LIVE VIDEO
# =============================================================================
def _show_frame(slot, image, **kwargs) -> None:
    """Draw an image into a slot; works on old and new Streamlit versions."""
    try:
        slot.image(image, width="stretch", **kwargs)
    except Exception:
        slot.image(image, use_container_width=True, **kwargs)


def live_camera_fragment() -> None:
    if st.session_state.get("use_webrtc"):
        # Browser camera: the WebRTC <video> shows the stream. This fragment
        # draws NO images; it only feeds alerts, stats and uploads.
        st.fragment(_live_webrtc_body, run_every=CONFIG["WEBRTC_TICK_SEC"])()
    else:
        # Local webcam: the exact flow of the working localhost project.
        st.fragment(_live_local_camera_body, run_every=0.1)()


def _connection_help_text() -> str:
    tier = ICE_SERVER_DIAGNOSTICS.get("tier", "unknown")
    detail = ICE_SERVER_DIAGNOSTICS.get("detail", "")

    processor = _get_browser_processor()
    if processor is None:
        stage = (
            "the browser has NOT connected to the server yet (no video processor was created, "
            "so the WebRTC handshake never finished)"
        )
    else:
        frames = int(getattr(processor, "frames_received", 0))
        if frames == 0:
            stage = "the connection opened but no frames have arrived yet"
        else:
            stage = f"frames arrived earlier ({frames} total) but stopped"

    return (
        "Camera stream stalled: " + stage + ". "
        f"Relay setup: {tier}{' - ' + detail if detail else ''}. "
        "A TURN credential response does not guarantee that WebRTC media remains connected. "
        "Press Reconnect Camera. If it persists, check Streamlit logs, network stability "
        "and available CPU; then allow camera access again if prompted."
    )


def _busy_flag_active() -> bool:
    """True if another tick is genuinely still processing. A flag older than
    BUSY_FLAG_TIMEOUT_SEC is treated as stale (e.g. an interrupted run)."""
    if not st.session_state.get("_frame_processing_busy"):
        return False
    since = st.session_state.get("_frame_processing_busy_since", 0.0)
    return (time.time() - since) < CONFIG["BUSY_FLAG_TIMEOUT_SEC"]


def _live_webrtc_body() -> None:
    status_slot = st.empty()
    render_custom_hazard_toast()

    if not st.session_state.get("camera_running", False):
        return

    if _busy_flag_active():
        return

    st.session_state._frame_processing_busy = True
    st.session_state._frame_processing_busy_since = time.time()
    try:
        processor = _get_browser_processor()
        frame, yolo, fps = (None, None, 0.0)
        if processor is not None:
            frame, yolo, fps = processor.snapshot()
            st.session_state.frames_received = int(processor.health().get("frames_received", 0))

        if frame is None:
            st.session_state.fps = 0.0
            waited = time.time() - (st.session_state.get("camera_started_at") or time.time())
            health = processor.health() if processor is not None else {}
            last_age = health.get("age_sec")
            stalled = bool(health.get("ended")) or (
                last_age is not None and last_age >= CONFIG["WEBRTC_STALL_WARNING_SEC"]
            )
            if stalled or waited >= CONFIG["CAMERA_CONNECT_HINT_SEC"]:
                status_slot.warning(_connection_help_text(), icon=":material/warning:")
                if st.button("Reconnect Camera", key="reconnect_browser_camera", type="primary"):
                    st.session_state.webrtc_connection_id += 1
                    st.session_state.get("_cam_holder", {}).clear()
                    st.session_state.webrtc_ctx = None
                    st.session_state.camera_started_at = time.time()
                    st.session_state.last_pipeline_ts = 0.0
                    st.session_state.fps = 0.0
                    st.rerun()
            else:
                status_slot.info("Connecting to browser camera…")
            return

        status_slot.empty()

        # Real throttling: skip expensive pipeline ticks under system pressure.
        tick = time.time()
        st.session_state.fps = fps
        if tick - st.session_state.get("last_pipeline_ts", 0) < ResourceGovernor.pipeline_interval():
            return
        st.session_state.last_pipeline_ts = tick
        h, w = frame.shape[:2]
        st.session_state.last_resolution = f"{w}x{h}"
        st.session_state.last_frame_time = tick
        st.session_state.frames_processed += 1

        target_w = CONFIG["WEBRTC_ANALYSIS_WIDTH"]
        target_h = max(1, int(round(h * target_w / max(w, 1))))
        small = cv2.resize(frame, (target_w, target_h))

        # YOLO boxes in `yolo` are in full-frame coordinates; the pipeline only
        # needs the detections/severity, not the pixels.
        run_detection_pipeline(small, yolo_override=yolo or dict(EMPTY_YOLO_RESULT))
        if processor is not None:
            processor.set_analysis_severity(st.session_state.get("yolo_severity", "NORMAL"))
            # WebRTC YOLO worker is the inference producer; the Streamlit-side
            # counter otherwise stays 0 even while detections occur.
            st.session_state.inference_count = int(processor.health().get("inference_count", 0))
    finally:
        st.session_state._frame_processing_busy = False


def _live_local_camera_body() -> None:
    # The slots are created INSIDE the fragment: a placeholder created outside
    # and passed in raises StreamlitInvalidLayoutContextError on newer Streamlit
    # when the fragment reruns on its own (run_every).
    frame_slot = st.empty()
    status_slot = st.empty()

    def _redraw_last_frame() -> None:
        last = st.session_state.get("last_frame_rgb")
        if last is not None:
            _show_frame(frame_slot, last, channels="RGB", output_format="JPEG")

    render_custom_hazard_toast()

    if not st.session_state.get("camera_running", False):
        return

    # A fragment rerun that draws nothing makes the previous image disappear.
    if _busy_flag_active():
        _redraw_last_frame()
        return

    st.session_state._frame_processing_busy = True
    st.session_state._frame_processing_busy_since = time.time()
    try:
        active_frames, primary_frame = VideoCaptureManager.capture_local_frames()

        if primary_frame is None and not active_frames:
            if st.session_state.get("last_frame_rgb") is not None:
                _redraw_last_frame()
            else:
                _show_frame(
                    frame_slot,
                    VideoCaptureManager.placeholder_frame("Starting camera...", "Opening local camera."),
                    channels="RGB",
                    output_format="JPEG",
                )
            status_slot.error(st.session_state.get("last_error") or "Camera unavailable.")
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

        _show_frame(frame_slot, display_rgb, channels="RGB", output_format="JPEG")
        status_slot.empty()
    finally:
        st.session_state._frame_processing_busy = False


def render_snapshot_camera() -> None:
    """Fallback camera: take a photo with the browser camera (st.camera_input)
    and analyse it. Needs no WebRTC / TURN, so it works on any hosted app."""
    st.session_state.camera_running = False
    st.session_state.get("_cam_holder", {}).clear()

    st.info(
        "Snapshot mode: allow camera access, press **Take Photo**, and the photo is "
        "analysed for fire and smoke. Take another photo to re-check."
    )

    shot = st.camera_input("Camera", key="snapshot_camera", label_visibility="collapsed")

    render_custom_hazard_toast()

    if shot is not None:
        data = shot.getvalue()
        sig = (len(data), hash(data))

        if st.session_state.get("_last_snapshot_sig") != sig:
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not read the photo. Try again.")
                return

            h, w = frame.shape[:2]
            target_w = CONFIG["SNAPSHOT_ANALYSIS_WIDTH"]
            if w > target_w:
                frame = cv2.resize(frame, (target_w, max(1, int(round(h * target_w / w)))))

            with st.spinner("Analysing photo..."):
                st.session_state.frames_processed += 1
                result = run_detection_pipeline(frame, sync=True)

            annotated = YOLOFireSmokeDetector.draw_boxes_on_frame(frame, result.get("yolo", {}))
            st.session_state.last_snapshot_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame_rgb = st.session_state.last_snapshot_rgb
            st.session_state._last_snapshot_sig = sig

    result_rgb = st.session_state.get("last_snapshot_rgb")
    if result_rgb is not None:
        st.markdown("<div class='stats-panel-title'>Analysis result</div>", unsafe_allow_html=True)
        _show_frame(st.empty(), result_rgb, channels="RGB", output_format="JPEG")
        severity = st.session_state.get("yolo_severity", "NORMAL")
        if severity == "NORMAL":
            st.success("No fire or smoke detected in this photo.", icon=":material/check_circle:")
        else:
            st.error(f"Hazard detected — severity {severity}.", icon=":material/dangerous:")


def render_video_feed() -> None:
    if _browser_snapshot_mode():
        render_snapshot_camera()
        return

    header_col, btn_col1, btn_col2 = st.columns([6, 1, 1])

    with btn_col1:
        if st.button("Start", icon=":material/play_arrow:", use_container_width=True, disabled=st.session_state.camera_running, key="start_cam_btn"):
            st.session_state.camera_running = True
            st.session_state.last_error = ""
            st.session_state.last_vlm_candidate_frame = None
            st.session_state.latest_vlm_text = ""
            st.session_state.last_frame_time = time.time()
            st.session_state.camera_started_at = time.time()
            st.session_state.last_frame_rgb = None
            st.session_state.last_resolution = ""
            st.session_state.latest_detection = None
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.yolo_detection = None
            st.session_state.yolo_detection_in_progress = False
            st.session_state.visual_smoke_streak = 0
            st.session_state.visual_fire_streak = 0
            st.session_state.visual_prev_frame = None
            st.session_state.last_visual_result = None
            st.session_state.last_pipeline_ts = 0.0
            st.session_state.frames_received = 0
            st.session_state.last_cloud_hazard_type = ""
            st.session_state.alarm_active = False
            st.session_state.active_hazard_toast = None
            st.session_state._frame_processing_busy = False
            st.session_state._frame_processing_busy_since = 0.0

            if st.session_state.get("yolo_detector") is None:
                st.session_state.yolo_detector = YOLOFireSmokeDetector()

            st.rerun()

    with btn_col2:
        if st.button("Stop", icon=":material/stop:", use_container_width=True, disabled=not st.session_state.camera_running, key="stop_cam_btn"):
            st.session_state.camera_running = False
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.alarm_active = False
            st.session_state.active_hazard_toast = None
            st.session_state._frame_processing_busy = False
            st.session_state._frame_processing_busy_since = 0.0
            st.session_state.get("_cam_holder", {}).clear()
            st.session_state.last_visual_result = None
            st.session_state.visual_prev_frame = None
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
# DATA LOGS (includes Export Reports)
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