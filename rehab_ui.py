import streamlit as st
import requests
import os
import tempfile

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Rehab Protocol Dashboard", layout="wide")

st.title("🏃 Ai-Powered Rehabilitation Dashboard")
st.markdown("Phase 5: End-to-End Integration (Upload -> Process -> Rehab)")

st.sidebar.header("Configuration")
athlete_id_input = st.sidebar.text_input("Athlete ID", value="ATH-001")

video_file = st.sidebar.file_uploader("Upload Raw Video for Processing", type=["mp4", "avi", "mov"])

if st.sidebar.button("Run Full Pipeline & Fetch Protocol"):
    if not athlete_id_input:
        st.sidebar.error("Please enter an Athlete ID.")
    elif not video_file:
        st.sidebar.error("Please upload a video file for processing.")
    else:
        with st.spinner("Processing Biomechanics Pipeline (This may take a minute)..."):
            try:
                # 1. Process Video
                files = {"video_file": (video_file.name, video_file.getvalue(), video_file.type)}
                data = {"athlete_id": athlete_id_input}
                process_resp = requests.post(f"{API_BASE_URL}/process_video", files=files, data=data)
                
                if process_resp.status_code == 200:
                    media_path = process_resp.json().get("media_path")
                    
                    # 2. Fetch Rehab Plan
                    rehab_resp = requests.get(f"{API_BASE_URL}/get_rehab_plan/{athlete_id_input}")
                    if rehab_resp.status_code == 200:
                        rehab_data = rehab_resp.json()
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("🚩 Flagged Anomaly")
                            
                            # Load media from backend
                            media_resp = requests.get(f"{API_BASE_URL}/media", params={"path": media_path})
                            if media_resp.status_code == 200:
                                st.video(media_resp.content)
                            else:
                                st.warning("Could not load visualization video from backend.")
                                
                            anomaly_details = rehab_data.get("latest_anomaly", {})
                            st.metric("Anomaly Type", anomaly_details.get("anomaly_type", "N/A"))
                            st.metric("Max Deviation Angle", f"{anomaly_details.get('max_deviation_angle', 0.0)}°")
                            st.metric("Flagged Frames", anomaly_details.get("frame_count", 0))
                            
                        with col2:
                            st.subheader("📋 Personalized Rehab Protocol")
                            protocol = rehab_data.get("rehab_protocol", {})
                            
                            st.markdown(f"**Trigger:** {protocol.get('trigger')}")
                            focus_areas = ", ".join(protocol.get("focus_areas", []))
                            st.markdown(f"**Focus Areas:** {focus_areas}")
                            
                            st.markdown("### Daily Exercises")
                            exercises = protocol.get("exercises", [])
                            
                            if exercises and not str(exercises[0].get('name')).startswith("Mocked Exercise") and not str(exercises[0].get('name')).startswith("API Error"):
                                for idx, ex in enumerate(exercises):
                                    with st.expander(f"Exercise {idx+1}: {ex.get('name')} (Muscle: {ex.get('muscle')})"):
                                        st.markdown(f"**Difficulty:** {ex.get('difficulty')}")
                                        st.markdown(f"**Instructions:** {ex.get('instructions')}")
                            elif exercises:
                                 for idx, ex in enumerate(exercises):
                                    with st.expander(f"Exercise {idx+1}: {ex.get('name')}"):
                                        st.markdown(f"**Instructions:** {ex.get('instructions')}")
                            else:
                                st.info("No exercises returned from API.")
                    else:
                        st.error(f"Failed to fetch rehab plan. Status: {rehab_resp.status_code}")
                else:
                    st.error(f"Pipeline processing failed: {process_resp.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the FastAPI backend. Make sure it is running on http://127.0.0.1:8000")
