import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_rehab_exercises_from_api(muscle: str, limit: int = 3):
    """
    Fetches exercises from API Ninjas Exercises API.
    """
    api_key = os.environ.get("API_NINJAS_KEY")
    if not api_key:
        return [{"name": "Mocked Exercise (No API Key Provided)", "muscle": muscle, "instructions": "Please provide an API_NINJAS_KEY environment variable to get real exercises."}]
        
    api_url = "https://api.api-ninjas.com/v1/exercises"
    params = {
        "muscle": muscle,
        "difficulty": "beginner"
    }
    headers = {"X-Api-Key": api_key}
    
    try:
        response = requests.get(api_url, params=params, headers=headers)
        if response.status_code == requests.codes.ok:
            exercises = response.json()
            return exercises[:limit]
        else:
            return [{"name": f"API Error: {response.status_code}", "instructions": response.text}]
    except Exception as e:
        return [{"name": "Request Error", "instructions": str(e)}]


def generate_rehab_protocol(anomaly_log: dict):
    """
    Logic to convert an anomaly log into a tailored rehabilitation program.
    """
    anomaly_type = anomaly_log.get("anomaly_type", "")
    
    protocol = {
        "athlete_id": anomaly_log.get("athlete_id"),
        "date_flagged": str(anomaly_log.get("date", "Unknown")),
        "trigger": anomaly_type,
        "focus_areas": [],
        "exercises": []
    }
    
    if anomaly_type.lower() == "knee valgus":
        protocol["focus_areas"] = ["Neuromuscular Control", "Hip Abductor Strengthening", "Glute Activation"]
        # Fetch exercises using the API Ninjas API (abductors / glutes)
        abductor_exercises = get_rehab_exercises_from_api("abductors", limit=2)
        glute_exercises = get_rehab_exercises_from_api("glutes", limit=1)
        
        protocol["exercises"].extend(abductor_exercises)
        protocol["exercises"].extend(glute_exercises)
    else:
        protocol["focus_areas"] = ["General Stability"]
        protocol["exercises"].extend(get_rehab_exercises_from_api("legs", limit=3))
        
    return protocol
