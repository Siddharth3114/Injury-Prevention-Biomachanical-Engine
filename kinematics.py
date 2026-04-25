import numpy as np

def calculate_3d_angle(a, b, c):
    """
    Calculate 3D angle at vertex B between A and C using dot product formula.
    theta = arccos((BA dot BC) / (|BA| * |BC|))
    a, b, c: arrays of shape (3,) representing (X, Y, Z) coordinates.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    numerator = np.dot(ba, bc)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    
    if denominator < 1e-6:
        return 0.0
        
    cos_theta = numerator / denominator
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

def get_kinematics(pose_data):
    """
    Calculate generic 3D angles for the knee across all frames.
    pose_data: (C, T, V, M) tensor
    Returns a list of dictionaries containing Left and Right Knee 3D angles.
    """
    C, T, V, M = pose_data.shape
    angles = []
    
    for t in range(T):
        # 3D Coordinates: take only X, Y, Z (indices 0, 1, 2)
        l_hip = pose_data[:3, t, 23, 0]
        l_knee = pose_data[:3, t, 25, 0]
        l_ankle = pose_data[:3, t, 27, 0]
        l_angle = calculate_3d_angle(l_hip, l_knee, l_ankle)
        
        r_hip = pose_data[:3, t, 24, 0]
        r_knee = pose_data[:3, t, 26, 0]
        r_ankle = pose_data[:3, t, 28, 0]
        r_angle = calculate_3d_angle(r_hip, r_knee, r_ankle)
        
        angles.append({'frame': t, 'LeftKnee_3D': l_angle, 'RightKnee_3D': r_angle})
        
    return angles

def detect_valgus(pose_data, threshold_degrees=10.0):
    """
    Detects Knee Valgus (inward collapse).
    Project to frontal plane (X, Y). Checks if knee shifts inward past the
    line from Hip to Ankle by > threshold_degrees during downward motion.
    
    Returns a list of anomaly frame indices.
    """
    C, T, V, M = pose_data.shape
    anomalies = []
    
    for t in range(1, T):
        # Downward motion check: Is the person moving downwards?
        # Check if Hip Y coordinate is increasing (MediaPipe: origin top-left, Y increases descending)
        prev_hip_y = (pose_data[1, t-1, 23, 0] + pose_data[1, t-1, 24, 0]) / 2.0
        curr_hip_y = (pose_data[1, t, 23, 0] + pose_data[1, t, 24, 0]) / 2.0
        
        # We consider any Y increase as downward motion. We can add a velocity threshold if needed.
        is_downward = (curr_hip_y - prev_hip_y) > 0.001 
        
        if is_downward:
            # --- Frontal Plane Projection ---
            # Extract taking only indices 0 and 1 (X and Y coordinates)
            l_hip = pose_data[:2, t, 23, 0]
            l_knee = pose_data[:2, t, 25, 0]
            l_ankle = pose_data[:2, t, 27, 0]
            
            r_hip = pose_data[:2, t, 24, 0]
            r_knee = pose_data[:2, t, 26, 0]
            r_ankle = pose_data[:2, t, 28, 0]
            
            # Vector from Hip to Ankle and Hip to Knee
            vec_l_ha = l_ankle - l_hip
            vec_l_hk = l_knee - l_hip
            
            vec_r_ha = r_ankle - r_hip
            vec_r_hk = r_knee - r_hip
            
            # Calculate angle between HA and HK in frontal plane
            def angle_between(v1, v2):
                den = np.linalg.norm(v1) * np.linalg.norm(v2)
                if den < 1e-6: return 0.0
                cos_a = np.dot(v1, v2) / den
                return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                
            angle_l = angle_between(vec_l_ha, vec_l_hk)
            angle_r = angle_between(vec_r_ha, vec_r_hk)
            
            # --- Inward Shift Check ---
            # Calculate straight-line knee X on the Hip-Ankle line at the same Y as the actual knee
            if abs(l_ankle[1] - l_hip[1]) > 1e-6:
                t_l = (l_knee[1] - l_hip[1]) / (l_ankle[1] - l_hip[1])
                straight_l_knee_x = l_hip[0] + t_l * (l_ankle[0] - l_hip[0])
            else:
                straight_l_knee_x = l_hip[0]
                
            if abs(r_ankle[1] - r_hip[1]) > 1e-6:
                t_r = (r_knee[1] - r_hip[1]) / (r_ankle[1] - r_hip[1])
                straight_r_knee_x = r_hip[0] + t_r * (r_ankle[0] - r_hip[0])
            else:
                straight_r_knee_x = r_hip[0]
                
            # Midline of the body
            midline_x = (l_hip[0] + r_hip[0]) / 2.0
            
            # If the knee is closer to the midline than the straight line is, it collapsed inward
            dist_l_to_midline = abs(l_knee[0] - midline_x)
            dist_straight_l_to_midline = abs(straight_l_knee_x - midline_x)
            is_l_inward = dist_l_to_midline < dist_straight_l_to_midline
            
            dist_r_to_midline = abs(r_knee[0] - midline_x)
            dist_straight_r_to_midline = abs(straight_r_knee_x - midline_x)
            is_r_inward = dist_r_to_midline < dist_straight_r_to_midline
            
            # Flag if shifted inward beyond safe threshold
            valgus_l = is_l_inward and (angle_l > threshold_degrees)
            valgus_r = is_r_inward and (angle_r > threshold_degrees)
            
            if valgus_l or valgus_r:
                anomalies.append(t)
                
    return sorted(list(set(anomalies)))

if __name__ == "__main__":
    import os
    if os.path.exists('pose_data.npy'):
        data = np.load('pose_data.npy')
        print(f"Loaded data shape: {data.shape}")
        
        anomalies = detect_valgus(data, threshold_degrees=10.0)
        print(f"Frames with valgus anomalies (threshold=10 deg): {anomalies}")
        
        angles = get_kinematics(data)
        print(f"First 5 frames kinematics:")
        for ang in angles[:5]:
            print(f"Frame {ang['frame']}: L {ang['LeftKnee_3D']:.1f} deg | R {ang['RightKnee_3D']:.1f} deg")
    else:
        print("pose_data.npy not found for testing.")
