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

# ✅ Load Firebase credentials from env
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)

cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# ✅ Load model
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# ✅ FastAPI app
app = FastAPI()

# ✅ Data model
class SensorData(BaseModel):
    humidity: float
    temperature: float
    soilMoisture: float

# ✅ Prediction function (reused in both API and thread)
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

        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

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

        scaled_input = scaler.transform(feature_vector)
        irrigation_class = int(model.predict(scaled_input)[0])

        # Update Firebase with timestamp
        timestamp = datetime.now().isoformat()
        db.reference('sensorData/prediction_class').set(irrigation_class)
        db.reference('sensorData/last_prediction_time').set(timestamp)
        
        print(f"✅ Prediction updated: Class {irrigation_class} at {timestamp}")

        return {"irrigation_class": irrigation_class, "timestamp": timestamp}
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}

# ✅ API route
@app.post("/predict")
def predict_route(data: SensorData):
    return predict_irrigation(data)

# ✅ Improved background thread with better change detection
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
            
            # More robust change detection
            if current is not None:
                # Check if data actually changed (handle None vs empty dict)
                data_changed = (
                    last_values is None or 
                    current != last_values or
                    not last_values  # Handle empty dict case
                )
                
                if data_changed:
                    print("🔔 Detected change in sensor data!")
                    print(f"   Previous: {last_values}")
                    print(f"   Current:  {current}")
                    
                    # Validate data before processing
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
                            
                            # Update last_values after successful processing
                            last_values = current.copy()
                            consecutive_errors = 0  # Reset error counter
                            
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

# Alternative: Using Firebase listeners (more efficient)
def setup_firebase_listener():
    """
    Alternative approach using Firebase real-time listeners
    This is more efficient than polling every 5 seconds
    """
    def sensor_data_listener(event):
        try:
            print(f"🔔 Firebase listener triggered: {event.event_type}")
            print(f"📊 New data: {event.data}")
            
            if event.data and event.event_type in ['put', 'patch']:
                required_fields = ['humidity', 'temperature', 'soilMoisture']
                if all(field in event.data for field in required_fields):
                    data = SensorData(
                        humidity=float(event.data.get("humidity", 0.0)),
                        temperature=float(event.data.get("temperature", 0.0)),
                        soilMoisture=float(event.data.get("soilMoisture", 0.0))
                    )
                    result = predict_irrigation(data)
                    print(f"✅ Listener prediction result: {result}")
                else:
                    print(f"❌ Listener: Missing required fields in data: {event.data}")
                    
        except Exception as e:
            print(f"❌ Firebase listener error: {e}")

    # Set up the listener
    ref = db.reference("sensorData/raw")
    ref.listen(sensor_data_listener)
    print("🎧 Firebase real-time listener set up")

# ✅ Start background monitoring (choose one method)
@app.on_event("startup")
def start_firebase_monitor():
    print("🚀 Starting Firebase monitoring...")
    
    # Option 1: Use polling method (your current approach, improved)
    threading.Thread(target=monitor_firebase_sensor_data, daemon=True).start()
    
    # Option 2: Use real-time listeners (more efficient, uncomment to use)
    # setup_firebase_listener()

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Test Firebase connection
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

# ✅ Manual trigger endpoint for testing
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
