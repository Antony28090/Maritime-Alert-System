import threading
import time
import random
from flask import Flask, render_template, jsonify
from src.models import ZoneClassifier, TrajectoryForecaster
from src.alert_system import AlertSystem
from src.data_generator import generate_trajectory
from src.geometry import distance_from_polyline, is_sri_lankan_side
from src.validation import get_validation_metrics
from src.config import *

app = Flask(__name__)

state_lock = threading.Lock()

# Global State
simulation_state = {
    "lat": REF_LAT,
    "lon": REF_LON,
    "zone": "SAFE",
    "forecast_msg": "Initializing...",
    "prediction": [],  # List of [lat, lon] for predicted path
    "alert_level": "none", # none, caution, danger
    "step": 0,
    "actual_history": [],
    "pred_1step_history": [],
    "actual_zone_history": [],
    "pred_zone_history": []
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
            
            # Mix force crossing trips with occasional normal safe trips
            is_forced = (int(time.time()) % 3 != 0) 
            trajectory_data = generate_trajectory(trip_id=trip_id, n_points=300, force_crossing=is_forced)
            
            # Check if this trajectory ever triggers an alert before visualizing it, EXCEPT if it's a normal safe trip
            has_alert = any(step['zone'] in ['DANGER', 'CAUTION'] for step in trajectory_data)
            if is_forced and not has_alert:
                continue # Skip this path since it doesn't trigger any alerts but was forced to
                
            path_buffer = []
            actual_history = []
            pred_1step_history = []
            actual_zone_history = []
            pred_zone_history = []
            steps_after_cross = 0

            for i, step in enumerate(trajectory_data):
                if not self.running: break
                
                lat = step['lat']
                lon = step['lon']
                
                # 1. Zone Classification
                predicted_zone = zone_model.predict(lat, lon)

                if DEMO_NOISE and random.random() < 0.025:
                    z = ["SAFE", "CAUTION", "DANGER"]
                    if predicted_zone in z: z.remove(predicted_zone)
                    predicted_zone = random.choice(z)
                
                # Actual logic to know if it really is in danger
                actual_dist, _ = distance_from_polyline([lat, lon], IMBL_POINTS)
                actual_sl = is_sri_lankan_side([lat, lon], IMBL_POINTS)
                if actual_sl or actual_dist < DANGER_DIST_KM:
                    actual_zone = "DANGER"
                elif actual_dist < CAUTION_DIST_KM:
                    actual_zone = "CAUTION"
                else:
                    actual_zone = "SAFE"
                    
                actual_zone_history.append(actual_zone)
                pred_zone_history.append(predicted_zone)
                
                # 2. Forecasting
                path_buffer.append([lat, lon])
                forecast_msg = "Gathering data..."
                prediction_path = [] # For visualization
                
                if len(path_buffer) > LSTM_LOOKBACK:
                    recent_path = path_buffer[-(LSTM_LOOKBACK + 1):]
                    
                    # Store data for live dashboard (1 step prediction vs actual)
                    nxt_1_step = lstm_model.predict_next(recent_path[:-1])
                    nxt_1_step_list = nxt_1_step.tolist() if hasattr(nxt_1_step, 'tolist') else nxt_1_step
                    
                    if DEMO_NOISE and random.random() < 0.025:
                        sgn_lat = 1 if random.random() > 0.5 else -1
                        sgn_lon = 1 if random.random() > 0.5 else -1
                        nxt_1_step_list[0] += sgn_lat * random.uniform(0.006, 0.010)
                        nxt_1_step_list[1] += sgn_lon * random.uniform(0.006, 0.010)
                        
                    actual_history.append([lat, lon])
                    pred_1step_history.append(nxt_1_step_list)
                    
                    # Predict multiple steps recursively to form a trajectory line (for the map forecast line)
                    pred_path = []
                    curr_seq = recent_path.copy()
                    
                    # Project FORECAST_HORIZON steps into the future (same H as src/evaluate.py)
                    for _ in range(FORECAST_HORIZON):
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
                with state_lock:
                    simulation_state = {
                        "lat": lat,
                        "lon": lon,
                        "zone": predicted_zone,
                        "forecast_msg": forecast_msg,
                        "prediction": prediction_path,
                        "alert_level": alert_level,
                        "step": i,
                        "actual_history": actual_history.copy(),
                        "pred_1step_history": pred_1step_history.copy(),
                        "actual_zone_history": actual_zone_history.copy(),
                        "pred_zone_history": pred_zone_history.copy()
                    }

                time.sleep(1) # Speed of simulation
                # If crossed the boundary, let the vessel travel a bit further before ending this trip
                if current_is_sl:
                    steps_after_cross += 1
                    if steps_after_cross >= 8:
                        print("Vessel crossed IMBL — starting a new trip.")
                        break

            if self.running:
                print("Trip complete — starting a new trip.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    with state_lock:
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
