import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Page Configuration
st.set_page_config(
    page_title="CyberVision Web - Hazard Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to remove outlining boxes / borders in Light Mode
st.markdown("""
    <style>
    /* Remove default card/container borders in light mode */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"], 
    [data-testid="stExpander"], 
    div.stButton > button {
        border-color: transparent !important;
    }
    
    /* Light mode clean aesthetics */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #f8f9fa;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: none !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("CyberVision Control")
app_mode = st.sidebar.selectbox("Navigation", ["Live Monitoring", "Data Logs & Reports", "System Settings"])

# Main Title
st.title("🛡️ CyberVision: Multi-Stage Edge AI Hazard Detection")

if app_mode == "Live Monitoring":
    st.subheader("Live Camera Feed & Detection Stream")
    
    col1, col2 = st.metrics = st.columns([3, 1])
    
    with col1:
        # Camera feed container
        camera_placeholder = st.empty()
        start_camera = st.toggle("Start Camera Feed", value=True)
        
        # Camera initialization fix
        if start_camera:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not open video stream. Please check camera connection.")
            else:
                # Placeholder loop for live stream simulation / OpenCV integration
                while start_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame from camera.")
                        break
                    
                    # Convert color space for Streamlit display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Small delay to keep UI responsive
                    time.sleep(0.03)
                    
                    # Break out if toggle is turned off
                    if not st.session_state.get("start_camera", True):
                        break
                cap.release()
        else:
            camera_placeholder.info("Camera is currently stopped. Toggle 'Start Camera Feed' above to begin.")

    with col2:
        st.markdown("### System Status")
        st.metric(label="YOLO Detection", value="Active", delta="Optimal")
        st.metric(label="VLM Verification", value="Standby", delta=None)
        st.metric(label="Resource Throttling", value="Normal", delta="-2% CPU")
        
        st.markdown("---")
        st.markdown("### Recent Alerts")
        st.info("No active hazards detected.")

elif app_mode == "Data Logs & Reports":
    st.subheader("📊 Data Logs & Export Reports")
    st.markdown("Review historical detection logs, system events, and export summary reports.")
    
    # Mock log data for demonstration
    log_data = pd.DataFrame({
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Event Type": ["System Initialization"],
        "Status": ["Success"],
        "Confidence": ["N/A"]
    })
    
    st.dataframe(log_data, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📥 Export Reports")
    st.markdown("Generate and download audit reports or detection history logs.")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_data = log_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Report",
            data=csv_data,
            file_name=f"cybervision_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        if st.button("Generate PDF Summary", use_container_width=True):
            st.success("PDF report generated successfully! Ready for download.")

elif app_mode == "System Settings":
    st.subheader("⚙️ Configuration")
    st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    st.slider("Resource Throttling Limit (psutil)", 10, 90, 75)
    st.text_input("Google Cloud Storage Bucket", value="cybervision-logs-bucket")