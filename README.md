# CyberVision: Adaptive Context Optimization at the Edge

---

## 📌 Overview

Uncontrolled fire and smoke incidents pose severe risks to human life, natural ecosystems, and infrastructure. Traditional point-detector physical sensors struggle in high-ceiling or outdoor environments due to delayed response times and lack of spatial awareness. While computer vision systems using object detection models like YOLO offer real-time detection, standalone deployments on resource-constrained edge devices face two major hurdles:

1. **High False-Positive Rates:** Flame-like objects (e.g., skin tones, red clothing, warm lighting) trigger frequent false alarms.
2. **Resource Exhaustion on Edge Hardware:** Heavy multi-modal pipelines running alongside Vision-Language Models (VLMs) cause high CPU/RAM load, frame dropping, and system instability.

**CyberVision** addresses these challenges through an adaptive, multi-stage edge AI surveillance architecture. It combines real-time deep learning detection, deterministic visual fallback color analysis, VLM semantic verification with YARA rule parsing, dynamic resource governance, and offline-first cloud synchronization.

---

## Live Detection Samples

| Active Combustion | Wildfire & Smoke Plumes | Early Smoke Detection |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/fa41c485-10b3-4571-a52f-77a7dac4fc3c" width="230" alt="Active Combustion" /> | <img src="https://github.com/user-attachments/assets/018955ef-efee-4d4f-b987-eaea51a4c1c6" width="230" alt="Wildfire & Smoke Plumes" /> | <img src="https://github.com/user-attachments/assets/cc02d218-29c4-4fcf-a462-0882d7d2f5e2" width="230" alt="Early Smoke Detection" /> |

---

## Key Features & Technical Architecture

### 1. Primary Bounding Box Detection (YOLO)
- Fine-tuned **YOLO** model targeting two primary classes: `fire` and `smoke`.
- Configured with a low initial confidence threshold ($conf_{thresh} = 0.15$) to prioritize early warning sensitivity.
- Calculates a confidence-aware hazard severity score (`NORMAL`, `MEDIUM`, `HIGH`, `CRITICAL`).

### 2. Deterministic Visual Fallback Analysis
- Serves as an ultra-low-latency baseline operating across **RGB, HSV, and YCrCb** color spaces when deep learning models are loading or restricted by system resources.
- **False Positive & Skin Suppression:** Evaluates candidate regions in YCrCb color space to filter human skin tones.
- **Flame & Smoke Segmentation:** Combines HSV bounds, RGB dominance rules, and morphological opening/dilation operations to segment spatial fire ($R_{fire}$) and smoke ($R_{smoke}$) area ratios.

### 3. Vision-Language Verification & YARA Rule Engine
- Dispatches candidate frames asynchronously to the **Moondream Vision-Language Model (VLM)** for context validation.
- Validates generated VLM text outputs using compiled **YARA rule matchers** to eliminate text parsing ambiguities and hallucinations on edge devices.
- Uses a score max-pooling function across YOLO, YARA, and Visual Fallback pipelines to determine final threat severity.

### 4. Dynamic Resource Governor
- Non-blocking monitor that continually tracks host CPU ($U_{CPU}$) and RAM ($U_{RAM}$) utilization via `psutil`.
- **Adaptive Throttling Policy:**
  - **Normal Load ($< 70\%$):** $640 \times 640$ resolution, 1024 token budget, 0.10s interval.
  - **Moderate Load ($70\% - 85\%$):** $640 \times 640$ resolution, 256 token budget, 0.22s interval.
  - **High Strain ($\ge 85\%$):** Dynamically scales input down to $320 \times 320$, limits token budget to 128, and increases interval to 0.75s to prevent Out-Of-Memory (OOM) crashes and system failure.

### 5. Offline-First Storage & Cloud Synchronization
- Confirmed incident logs are buffered instantly into a local **SQLite database** (`cybervision_buffer.db`).
- A daemon thread monitors network state via socket DNS probes.
- When connectivity is restored, buffered events, JPEG keyframes, and JSON metadata automatically sync to **Google Cloud Firestore** and **Google Cloud Storage**.

---

## System Pipeline Flow

```text
+--------------------------------------------------------+
|                   INPUT CAMERA FEED                    |
|             (3x3 Grid Matrix / Single Stream)          |
+---------------------------+----------------------------+
                            |
                            v
+--------------------------------------------------------+
|                   RESOURCE GOVERNOR                    |
|       Monitors CPU/RAM Usage -> Adjusts Res/Tokens     |
+---------------------------+----------------------------+
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
+-----------------------+       +------------------------+
|   YOLO FIRE/SMOKE NET |       | VISUAL FALLBACK        |
| (Primary Bounding Box)|       | (HSV/YCrCb Color Seg)  |
+----------+------------+       +------------+-----------+
           |                                 |
           +----------------+----------------+
                            |
                            v
+--------------------------------------------------------+
|               CANDIDATE HAZARD EVALUATION              |
|        Triggers Moondream VLM & YARA Parsing           |
+---------------------------+----------------------------+
                            |
                            v
+--------------------------------------------------------+
|             MOONDREAM VLM + YARA RULE ENGINE           |
|            Semantic Text Parsing & Max-Pooling         |
+---------------------------+----------------------------+
                            |
                            v
+--------------------------------------------------------+
|             ALERT & FORENSIC SYNC ENGINE               |
|      Local SQLite Buffer -> Audio Alert -> GCP Cloud   |
+--------------------------------------------------------+
```

## Prerequisites & Installation



### Core Requirements

- **Python 3.8+**

- **OpenCV (`opencv-python`)**

- **PyTorch**

- **YOLO (Ultralytics)**

- **Moondream VLM**

- **YARA (`yara-python`)**

- **psutil**

- **Google Cloud SDK (`google-cloud-firestore`, `google-cloud-storage`)**



### Setup

```bash

# Clone the repository

git clone [https://github.com/ChrystinaChin/Computer-Vision.git](https://github.com/ChrystinaChin/Computer-Vision.git)

cd Computer-Vision



# Install dependencies

pip install -r requirements.txt



# Run CyberVision

python main.py 

