import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse
import os

def extract_pose(video_path, output_path, model_path='pose_landmarker_heavy.task'):
    print(f"Processing video: {video_path}")
    
    # Initialize the Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frames_data = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # fallback
        
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int(frame_count * 1000 / fps)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # 33 landmarks, 4 channels (x, y, z, visibility)
            landmarks_data = np.zeros((33, 4))
            
            if pose_landmarker_result.pose_landmarks:
                # We only take the first person detected (ST-GCN M=1)
                pose_landmarks = pose_landmarker_result.pose_landmarks[0]
                for i, landmark in enumerate(pose_landmarks):
                    landmarks_data[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    
            frames_data.append(landmarks_data)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Total frames processed: {frame_count}")
    
    data = np.array(frames_data)
    
    if data.size == 0:
        print("No frames were processed.")
        return
        
    # ST-GCN shape: (C, T, V, M)
    # Current shape: (T, V, C) -> Transpose to (C, T, V)
    data = np.transpose(data, (2, 0, 1))
    
    # Add M dimension -> (C, T, V, 1)
    data = np.expand_dims(data, axis=3)
    
    print(f"Data shape for export: {data.shape}")
    
    np.save(output_path, data)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D pose landmarks from video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video file")
    parser.add_argument("--output", type=str, default="pose_data.npy", help="Path to output .npy file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Input video '{args.video}' not found.")
    else:
        extract_pose(args.video, args.output)
