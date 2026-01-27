import threading
import time
import json
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from shapely.geometry import Point
from src.models import ZoneClassifier, TrajectoryForecaster
from src.alert_system import AlertSystem
from src.data_generator import generate_trajectory, distance_from_line
from src.config import *

app = Flask(__name__)

# Global State
simulation_state = {
    "lat": REF_LAT,
    "lon": REF_LON,
    "zone": "SAFE",
    "forecast_msg": "Initializing...",
    "prediction": [],  # List of [lat, lon] for predicted path
    "alert_level": "none", # none, caution, danger
    "step": 0
}

class SimulationThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        
    def run(self):
        print("Starting Simulation Thread...")
        # Load Models inside thread (or global, but simpler here for now)
        try:
            zone_model = ZoneClassifier()
            zone_model.load()
            lstm_model = TrajectoryForecaster()
            lstm_model.load()
            alert_sys = AlertSystem()
        except Exception as e:
            print(f"Error loading models in sim thread: {e}")
            return

        while self.running:
             # Generate a new random trip
            trajectory_data = generate_trajectory(trip_id=int(time.time()), n_points=60)
            path_buffer = []

            for i, step in enumerate(trajectory_data):
                if not self.running: break
                
                lat = step['lat']
                lon = step['lon']
                
                # 1. Zone Classification
                predicted_zone = zone_model.predict(lat, lon)
                
                # 2. Forecasting
                path_buffer.append([lat, lon])
                forecast_msg = "Gathering data..."
                prediction_path = [] # For visualization
                
                if len(path_buffer) > LSTM_LOOKBACK:
                    recent_path = path_buffer[-LSTM_LOOKBACK:]
                    next_pos = lstm_model.predict_next(recent_path)
                    
                    # We could predict multiple steps recursively for a line
                    # For now just one step prediction for "Time to Intercept" check
                    prediction_path = [next_pos.tolist()] 
                    
                    p_point = Point(next_pos[1], next_pos[0])
                    line_start = (IMBL_LON_START, IMBL_LAT_START)
                    line_end = (IMBL_LON_END, IMBL_LAT_END)
                    dist_pred = distance_from_line(p_point, line_start, line_end)
                    
                    if dist_pred < DANGER_DIST_KM:
                        if predicted_zone == 'DANGER':
                             forecast_msg = "CRITICAL: ALREADY IN DANGER ZONE!"
                        else:
                             forecast_msg = "PREDICTION: Entering DANGER in ~10m!"
                             alert_sys.trigger_alert('caution')
                    else:
                        # If distance > 2km, it could be Safe (Indian side) OR Deep in Sri Lankan side.
                        # Rely on the Zone Classifier to tell us if we are currently in Danger.
                        if predicted_zone == 'DANGER':
                            forecast_msg = "Forecast: Deep in Danger Zone"
                        else:
                            forecast_msg = "Forecast: Safe"
                
                # 3. Alerts & Update State
                alert_level = "none"
                if predicted_zone == 'DANGER':
                    alert_sys.trigger_alert('danger')
                    alert_level = "danger"
                elif predicted_zone == 'CAUTION':
                    alert_sys.trigger_alert('caution')
                    alert_level = "caution"
                
                # Update Global State
                global simulation_state
                simulation_state = {
                    "lat": lat,
                    "lon": lon,
                    "zone": predicted_zone,
                    "forecast_msg": forecast_msg,
                    "prediction": prediction_path,
                    "alert_level": alert_level,
                    "step": i
                }
                
                time.sleep(1) # Speed of simulation
            
            print("Trip complete. Restarting new trip...")
            time.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(simulation_state)

@app.route('/api/config')
def get_config():
    # Send boundary data to frontend for drawing
    return jsonify({
        "imbl_start": [IMBL_LAT_START, IMBL_LON_START],
        "imbl_end": [IMBL_LAT_END, IMBL_LON_END],
        "ref_point": [REF_LAT, REF_LON]
    })

if __name__ == '__main__':
    sim_thread = SimulationThread()
    sim_thread.start()
    app.run(debug=True, use_reloader=False) # use_reloader=False to prevent double threads
