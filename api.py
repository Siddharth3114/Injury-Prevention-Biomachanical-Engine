from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import shutil
import os
import pipeline
from sqlalchemy.orm import Session
import database
import rehab_generator

app = FastAPI(title="Biomechanical Engine API")

# Pydantic schema for incoming requests
class AnomalyPayload(BaseModel):
    athlete_id: str
    anomaly_type: str
    max_deviation_angle: float
    frame_count: int

@app.post("/log_anomaly")
def log_anomaly(payload: AnomalyPayload, db: Session = Depends(database.get_db)):
    """
    Ingests anomaly data from Phase 3 and saves it to the SQLite database.
    """
    new_log = database.AnomalyLog(
        athlete_id=payload.athlete_id,
        anomaly_type=payload.anomaly_type,
        max_deviation_angle=payload.max_deviation_angle,
        frame_count=payload.frame_count
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return {"message": "Anomaly logged successfully", "data": new_log}

@app.get("/get_rehab_plan/{athlete_id}")
def get_rehab_plan(athlete_id: str, db: Session = Depends(database.get_db)):
    """
    Fetches the specified athlete's latest anomaly log, 
    triggers the rehab generator, and returns the protocol.
    """
    # Fetch the latest anomaly for this athlete
    latest_log = db.query(database.AnomalyLog).filter(
        database.AnomalyLog.athlete_id == athlete_id
    ).order_by(database.AnomalyLog.date.desc()).first()
    
    if not latest_log:
        raise HTTPException(status_code=404, detail="No anomalies found for this athlete.")
        
    # Convert DB model to dict
    anomaly_data = {
        "id": latest_log.id,
        "athlete_id": latest_log.athlete_id,
        "date": latest_log.date,
        "anomaly_type": latest_log.anomaly_type,
        "max_deviation_angle": latest_log.max_deviation_angle,
        "frame_count": latest_log.frame_count
    }
    
    # Generate the rehab protocol using the API/LLM mock
    protocol = rehab_generator.generate_rehab_protocol(anomaly_data)
    
    return {
        "athlete_id": athlete_id,
        "latest_anomaly": anomaly_data,
        "rehab_protocol": protocol
    }

@app.post("/process_video")
def process_video(athlete_id: str = Form(...), video_file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    """
    End-to-End Pipeline: Accepts video, runs extraction/kinematics/dashboard generation, logs to DB.
    """
    try:
        # Save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            shutil.copyfileobj(video_file.file, temp_file)
            temp_path = temp_file.name

        # Run pipeline
        log_data, out_media_path = pipeline.run_full_pipeline(temp_path)

        # Log into database automatically
        new_log = database.AnomalyLog(
            athlete_id=athlete_id,
            anomaly_type=log_data["anomaly_type"],
            max_deviation_angle=log_data["max_deviation_angle"],
            frame_count=log_data["frame_count"]
        )
        db.add(new_log)
        db.commit()
        
        return {"message": "Pipeline execution successful", "media_path": out_media_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/media")
def get_media(path: str):
    """Serve the generated MP4 or GIF visualization"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
