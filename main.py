import os
import json
import threading
import time
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# ============================
# 🔑 Firebase Initialization
# ============================
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)

cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# ============================
# 📦 Load Model + Artifacts
# ============================
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# ============================
# 🚀 FastAPI App
# ============================
app = FastAPI(title="Irrigation Backend", version="1.0")

# ============================
# 📊 Data Model
# ============================
class SensorData(BaseModel):
    humidity: float
    temperature: float
    soilMoisture: float

# ============================
# 🤖 Prediction Function
# ============================
def predict_irrigation(data: SensorData):
    try:
        now = datetime.now()
        full_input = {
            'soil_moisture_percent': data.soilMoisture,
            'temperature_celsius': data.temperature,
            'humidity_percent': data.humidity,
            'rainfall_mm_prediction_next_1h': 0.5,
            'hour': now.hour,
            'day_of_year': now.timetuple().tm_yday,
            'month': now.month,
            'district': 'Coimbatore',
            'zone': 'Western Zone',
            'season': 'southwest_monsoon'
        }

        # Encode categorical features
        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        # Extra engineered features
        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

        # Build feature vector
        feature_vector = np.array([[
            full_input['soil_moisture_percent'],
            full_input['temperature_celsius'],
            full_input['humidity_percent'],
            full_input['rainfall_mm_prediction_next_1h'],
            full_input['hour'],
            full_input['day_of_year'],
            full_input['month'],
            district_enc,
            zone_enc,
            season_enc,
            heat_stress,
            drought_stress,
            soil_temp_interaction,
            humidity_rain_interaction
        ]])

        # Scale & predict
        scaled_input = scaler.transform(feature_vector)
        irrigation_class = int(model.predict(scaled_input)[0])

        # Save result to Firebase
        timestamp = datetime.now().isoformat()
        db.reference('sensorData/prediction_class').set(irrigation_class)
        db.reference('sensorData/last_prediction_time').set(timestamp)
        
        print(f"✅ Prediction updated: Class {irrigation_class} at {timestamp}")

        return {"irrigation_class": irrigation_class, "timestamp": timestamp}
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}

# ============================
# 🌐 API Routes
# ============================

# Root route for Render health check
@app.get("/")
def root():
    return {"message": "🌱 Irrigation backend is running"}

# Manual prediction endpoint
@app.post("/predict")
def predict_route(data: SensorData):
    return predict_irrigation(data)

# Health check
@app.get("/health")
def health_check():
    try:
        test_ref = db.reference("sensorData/raw")
        current_data = test_ref.get()
        
        return {
            "status": "healthy",
            "firebase_connected": True,
            "current_sensor_data": current_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Manual trigger from Firebase
@app.post("/trigger-prediction")
def trigger_prediction():
    try:
        ref = db.reference("sensorData/raw")
        current_data = ref.get()
        
        if current_data:
            data = SensorData(
                humidity=float(current_data.get("humidity", 0.0)),
                temperature=float(current_data.get("temperature", 0.0)),
                soilMoisture=float(current_data.get("soilMoisture", 0.0))
            )
            result = predict_irrigation(data)
            return {"status": "success", "result": result, "input_data": current_data}
        else:
            return {"status": "error", "message": "No sensor data found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================
# 🔄 Background Firebase Monitor
# ============================
def monitor_firebase_sensor_data():
    last_values = None
    consecutive_errors = 0
    max_errors = 5

    print("🔄 Starting Firebase monitoring...")

    while True:
        try:
            ref = db.reference("sensorData")
            current = ref.get()
            print(f"📊 Current sensor data: {current}")
            
            if current is not None:
                if last_values is None or current != last_values or not last_values:
                    print("🔔 Detected change in sensor data!")
                    print(f"   Previous: {last_values}")
                    print(f"   Current:  {current}")
                    
                    required_fields = ['humidity', 'temperature', 'soilMoisture']
                    if all(field in current for field in required_fields):
                        try:
                            data = SensorData(
                                humidity=float(current.get("humidity", 0.0)),
                                temperature=float(current.get("temperature", 0.0)),
                                soilMoisture=float(current.get("soilMoisture", 0.0))
                            )
                            result = predict_irrigation(data)
                            print(f"✅ Prediction result: {result}")
                            last_values = current.copy()
                            consecutive_errors = 0
                        except (ValueError, TypeError) as e:
                            print(f"❌ Data validation error: {e}")
                            print(f"   Raw data: {current}")
                    else:
                        missing_fields = [f for f in required_fields if f not in current]
                        print(f"❌ Missing required fields: {missing_fields}")
                        print(f"   Available fields: {list(current.keys())}")
                else:
                    print("📊 No change detected in sensor data")
            else:
                print("⚠️  No sensor data found in Firebase")
                
        except Exception as e:
            consecutive_errors += 1
            print(f"❌ Error while monitoring sensor data (attempt {consecutive_errors}): {e}")
            if consecutive_errors >= max_errors:
                print(f"💥 Too many consecutive errors ({max_errors}). Stopping monitor.")
                break

        time.sleep(5)

# Start monitoring on startup
@app.on_event("startup")
def start_firebase_monitor():
    print("🚀 Starting Firebase monitoring...")
    threading.Thread(target=monitor_firebase_sensor_data, daemon=True).start()
