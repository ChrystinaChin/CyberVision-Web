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