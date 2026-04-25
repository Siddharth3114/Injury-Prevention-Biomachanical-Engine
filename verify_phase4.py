import requests

API_BASE_URL = "http://127.0.0.1:8000"

payload = {
    "athlete_id": "ATH-001",
    "anomaly_type": "Knee Valgus",
    "max_deviation_angle": 15.5,
    "frame_count": 12
}

try:
    print("Testing POST /log_anomaly...")
    response = requests.post(f"{API_BASE_URL}/log_anomaly", json=payload)
    print("Response:", response.status_code, response.json())
    
    print("\nTesting GET /get_rehab_plan/ATH-001...")
    response = requests.get(f"{API_BASE_URL}/get_rehab_plan/ATH-001")
    print("Response:", response.status_code, response.json())
except Exception as e:
    print(e)
