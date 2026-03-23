import threading
import time
import json
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from shapely.geometry import Point
from src.models import ZoneClassifier, TrajectoryForecaster
from src.alert_system import AlertSystem
from src.data_generator import generate_trajectory
from src.geometry import distance_from_polyline, is_sri_lankan_side
from src.validation import get_validation_metrics
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
            # Uses current time as random seed/ID
            trip_id = int(time.time())
            trajectory_data = generate_trajectory(trip_id=trip_id, n_points=300, force_crossing=True)
            
            # Check if this trajectory ever triggers an alert before visualizing it
            has_alert = any(step['zone'] in ['DANGER', 'CAUTION'] for step in trajectory_data)
            if not has_alert:
                continue # Skip this path since it doesn't trigger any alerts
                
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
                    recent_path = path_buffer[-(LSTM_LOOKBACK + 1):]
                    
                    # Predict multiple steps recursively to form a trajectory line
                    pred_path = []
                    curr_seq = recent_path.copy()
                    
                    # Project 15 steps into the future
                    for _ in range(15):
                        nxt = lstm_model.predict_next(curr_seq[-(LSTM_LOOKBACK + 1):])
                        nxt_list = nxt.tolist() if hasattr(nxt, 'tolist') else nxt
                        pred_path.append(nxt_list)
                        curr_seq.append(nxt_list)
                        
                    prediction_path = pred_path
                    
                    # Check if any future point hits Danger or crosses
                    in_sl_side = False
                    dist_pred = float('inf')
                    for pt in pred_path:
                        d, _ = distance_from_polyline(pt, IMBL_POINTS)
                        if is_sri_lankan_side(pt, IMBL_POINTS):
                            in_sl_side = True
                            dist_pred = min(dist_pred, d)
                            break
                        dist_pred = min(dist_pred, d)
                        
                    if in_sl_side or dist_pred < DANGER_DIST_KM:
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
                
                # Check actual side for "Crossed" status
                current_is_sl = is_sri_lankan_side([lat, lon], IMBL_POINTS)
                
                if current_is_sl:
                    alert_sys.trigger_alert('crossed')
                    alert_level = "crossed"
                    predicted_zone = "CROSSED" # Override zone for display
                elif predicted_zone == 'DANGER':
                    # Close to border but not crossed
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

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    return jsonify(simulation_state)

@app.route('/api/validation')
def get_validation():
    try:
        metrics = get_validation_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config')
def get_config():
    # Send boundary data to frontend for drawing
    return jsonify({
        "imbl_points": IMBL_POINTS,
        "ref_point": [REF_LAT, REF_LON]
    })

if __name__ == '__main__':
    sim_thread = SimulationThread()
    sim_thread.start()
    app.run(debug=True, use_reloader=False) # use_reloader=False to prevent double threads
