# Biomechanical Analysis Engine

An end-to-end AI-powered system designed to analyze athlete biomechanics from normal video files. It leverages computer vision for spatial pose extraction, geometric methods for calculating 3D joint angles to detect kinematic anomalies (like knee valgus), and generative AI to recommend personalized rehabilitation protocols.

## Key Features

- **3D Pose Extraction**: Utilizes MediaPipe to extract 3D human pose landmarks directly from standard 2D `.mp4` video files.
- **Kinematic Anomaly Detection**: Analyzes spatial data to calculate physiological joint angles (e.g., 3D knee joints) and detects biomechanical anomalies, such as knee valgus (inward knee collapse) during exercise.
- **Data Visualization**: Renders an interactive 3D dashboard highlighting the skeleton and dynamically changing colors (red bounds) during flagged anomaly frames over time.
- **Rehabilitation Planning (GenAI)**: Synthesizes anomalous data and creates personalized rehab and recovery plans utilizing Google's Gemini LLM. Includes SQLite storage of session histories for progressive tracking.
- **Interactive UI**: A full-stack Streamlit application allowing users to upload videos, process them immediately, view the annotated animation, and read their AI-generated recovery plans.

## Project Structure

- `extract_pose.py`: MediaPipe integration for grabbing (x,y,z) anatomical landmarks frame-by-frame and exporting to ST-GCN compliant `.npy` datasets.
- `kinematics.py`: Mathematics module calculating 3D joint angles to identify conditions such as knee valgus based on predefined physiological angle thresholds.
- `dashboard.py`: 3D Matplotlib animation script to render the skeletal wireframe and save `.mp4`/`.gif` visualizations highlighting detected irregularities.
- `pipeline.py`: Orchestrator script connecting video input -> pose extraction -> anomaly detection -> dashboard rendering.
- `rehab_generator.py` & `api.py`: Connects identified kinematics data to the Gemini Generative AI backend to create structured rehab protocols.
- `rehab_ui.py`: The user-facing Streamlit application.
- `database.py`: Handles SQLite schemas and transactions to log processing sessions.


## Output Structure

Raw processed kinematics are temporarily cached as NumPy array arrays (`.npy`), compliant with shapes `(C, T, V, M)` natively suitable if extending this application with PyTorch ST-GCN integrations.
