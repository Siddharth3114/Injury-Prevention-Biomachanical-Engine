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

## Setup Instructions

### 1. Requirements

Ensure you have Python 3.8+ installed. You also need a Gemini API Key to use the generative rehab features. 

### 2. Virtual Environment Setup

```bash
# Initialize Virtual Environment
python -m venv venv

# Activate Virtual Environment (Windows)
.\venv\Scripts\activate

# Activate Virtual Environment (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

You will need the following primary packages:
```bash
pip install opencv-python mediapipe numpy matplotlib streamlit google-generativeai fastapi uvicorn
```

### 4. Environment Variables

Create a file named `.env` in your root directory and add your secret keys. Check the provided `.gitignore` to ensure it is not tracked.
```env
GEMINI_API_KEY="your_api_key_here"
```

## Running the Application

The simplest way to use the unified engine is starting the Streamlit dashboard:

```bash
streamlit run rehab_ui.py
```

This will deploy a local web application. You can browse and upload a `.mp4` video file, run the complete biomechanical diagnostic, view the 3D replay (highlighting frames of failure like knee valgus), and review your customized rehabilitation regime.

## Output Structure

Raw processed kinematics are temporarily cached as NumPy array arrays (`.npy`), compliant with shapes `(C, T, V, M)` natively suitable if extending this application with PyTorch ST-GCN integrations.
