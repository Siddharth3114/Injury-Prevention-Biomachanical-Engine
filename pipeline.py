import os
import numpy as np
from extract_pose import extract_pose
from kinematics import detect_valgus, get_kinematics
from dashboard import render_dashboard

def run_full_pipeline(video_path: str):
    print(f"--- Starting Full Pipeline for {video_path} ---")
    
    # 1. Pose Extraction
    npy_path = video_path + ".npy"
    extract_pose(video_path, npy_path)
    
    if not os.path.exists(npy_path):
        raise Exception("Pose extraction failed to produce a .npy file.")
        
    # 2. Kinematics
    pose_data = np.load(npy_path)
    anomalies = detect_valgus(pose_data, threshold_degrees=10.0)
    
    anomaly_type = "None"
    max_dev = 0.0
    frame_count = len(anomalies)
    
    if frame_count > 0:
        anomaly_type = "Knee Valgus"
        angles = get_kinematics(pose_data)
        
        # Approximate Max Deviation
        for f in anomalies:
            max_dev = max(max_dev, angles[f]['LeftKnee_3D'], angles[f]['RightKnee_3D'])

    # 3. Dashboard rendering
    out_mp4 = video_path + "_output.mp4"
    render_dashboard(data_path=npy_path, output_mp4=out_mp4)
    
    final_out_media = out_mp4
    if not os.path.exists(out_mp4):
        gif_out = out_mp4.replace('.mp4', '.gif')
        if os.path.exists(gif_out):
            final_out_media = gif_out
        else:
            raise Exception("Animation generation failed.")
            
    print("--- Pipeline Completed Successfully ---")
    return {
        "anomaly_type": anomaly_type,
        "max_deviation_angle": float(round(max_dev, 2)),
        "frame_count": frame_count
    }, final_out_media
