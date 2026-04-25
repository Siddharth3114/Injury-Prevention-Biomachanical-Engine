import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from kinematics import detect_valgus

# MediaPipe Pose bone connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), 
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

def render_dashboard(data_path="pose_data.npy", output_mp4="dashboard_output.mp4"):
    if not os.path.exists(data_path):
        print(f"Error: dataset {data_path} not found.")
        return
        
    pose_data = np.load(data_path)
    C, T, V, M = pose_data.shape
    print(f"Loaded pose data with shape: {pose_data.shape}")
    
    anomalies = set(detect_valgus(pose_data, threshold_degrees=10.0))
    print(f"Detected {len(anomalies)} anomaly frames.")
    
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('#1e1e1e')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1e1e1e')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=15, azim=20)
    
    # Extract all frames and keypoints for mapping
    xs = pose_data[0, :, :, 0]
    ys = pose_data[1, :, :, 0]
    zs = pose_data[2, :, :, 0]
    
    # MediaPipe coordinate translation for proper 3D rendering: 
    # Mapped so person stands upright (Y inversion)
    plot_x = xs
    plot_y = zs
    plot_z = -ys
    
    # Remove axis ticks for an aesthetic dashboard
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Establish bounding box dynamically based on the dataset's entire range
    min_x, max_x = np.min(plot_x), np.max(plot_x)
    min_y, max_y = np.min(plot_y), np.max(plot_y)
    min_z, max_z = np.min(plot_z), np.max(plot_z)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
    
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)
    
    # Construct 3D Lines
    lines = []
    for _ in POSE_CONNECTIONS:
        line, = ax.plot([], [], [], lw=2, color='#aaaaaa')
        lines.append(line)
        
    title = ax.set_title("Biomechanical Engine - Phase 3\nInitializing...", color='white', pad=20)
    
    def init():
        return lines + [title]
        
    def update(frame):
        curr_x = plot_x[frame]
        curr_y = plot_y[frame]
        curr_z = plot_z[frame]
        
        is_anomaly = frame in anomalies
        
        for i, (u, v) in enumerate(POSE_CONNECTIONS):
            # Coordinates
            lx = [curr_x[u], curr_x[v]]
            ly = [curr_y[u], curr_y[v]]
            lz = [curr_z[u], curr_z[v]]
            
            lines[i].set_data(lx, ly)
            lines[i].set_3d_properties(lz)
            
            # Highlight target structure logic (Knee Joints & Bones)
            is_knee_conn = ((u, v) in [(23,25), (25,27), (24,26), (26,28)] or 
                           (v, u) in [(23,25), (25,27), (24,26), (26,28)])
            
            if is_knee_conn:
                if is_anomaly:
                    lines[i].set_color('#ff3333') # Red logic
                    lines[i].set_linewidth(4.0)
                else:
                    lines[i].set_color('#33ff33') # Standard green
                    lines[i].set_linewidth(3.0)
            else:
                lines[i].set_color('#888888')
                lines[i].set_linewidth(1.5)
                
        if is_anomaly:
            title.set_text(f"Frame {frame}/{T} - KNEE VALGUS DETECTED!")
            title.set_color('#ff3333')
        else:
            title.set_text(f"Frame {frame}/{T} - Normal Kinematics")
            title.set_color('white')
            
        return lines + [title]

    print(f"Rendering animation across {T} frames...")
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=33)
    
    try:
        ani.save(output_mp4, writer='ffmpeg', fps=30)
        print(f"Successfully saved dashboard visualization to {output_mp4}")
    except Exception as e:
        print(f"FFMpeg save failed: {e}")
        print("Falling back to GIF output...")
        gif_out = output_mp4.replace('.mp4', '.gif')
        ani.save(gif_out, writer='pillow', fps=30)
        print(f"Saved fallback GIF to {gif_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Kinematic 3D Dashboard")
    parser.add_argument("--data", type=str, default="pose_data.npy", help="Path to Phase 1/2 .npy output data")
    parser.add_argument("--out", type=str, default="dashboard_output.mp4", help="Filename of the rendered output video")
    args = parser.parse_args()
    
    render_dashboard(args.data, args.out)
