import time
import pandas as pd
import numpy as np
from src.models import ZoneClassifier, TrajectoryForecaster
from src.alert_system import AlertSystem
from src.data_generator import generate_trajectory, distance_from_line
from src.config import *

def main():
    print("Initializing Maritime Alert System...")
    
    # Load Models
    print("Loading models...")
    try:
        zone_model = ZoneClassifier()
        zone_model.load()
        
        lstm_model = TrajectoryForecaster()
        lstm_model.load()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run src/train.py first.")
        return

    # Initialize Alert System
    alert_sys = AlertSystem()
    
    # Simulate a live trip
    print("\nStarting Real-time Simulation...")
    # Generate a trip that crosses bounds
    # using trip_id=999 as seed
    trajectory_data = generate_trajectory(trip_id=999, n_points=30) 
    
    # Buffer to store recent points for LSTM
    path_buffer = [] 
    
    for i, step in enumerate(trajectory_data):
        lat = step['lat']
        lon = step['lon']
        
        # 1. Zone Classification
        predicted_zone = zone_model.predict(lat, lon)
        
        # 2. Trajectory Forecasting (needs history)
        path_buffer.append([lat, lon])
        forecast_msg = "Gathering data..."
        
        if len(path_buffer) > LSTM_LOOKBACK:
            # Keep only last N points
            recent_path = path_buffer[-LSTM_LOOKBACK:]
            
            # Predict future
            next_pos = lstm_model.predict_next(recent_path)
            
            # Estimate time to intercept (very rough approx)
            # Calculate distance of predicted point to IMBL
            p_point = Point(next_pos[1], next_pos[0]) # lon, lat
            line_start = (IMBL_LON_START, IMBL_LAT_START)
            line_end = (IMBL_LON_END, IMBL_LAT_END)
            dist_pred = distance_from_line(p_point, line_start, line_end)
            
            if dist_pred < DANGER_DIST_KM:
                forecast_msg = "PREDICTION: Will enter DANGER zone in ~10 mins!"
                if predicted_zone != 'DANGER':
                    alert_sys.trigger_alert('caution') # Pre-emptive caution
            else:
                forecast_msg = "Forecast: Safe trajectory"

        # 3. Alerting Logic
        status_color = "WHITE"
        if predicted_zone == 'DANGER':
            alert_sys.trigger_alert('danger')
            status_color = "RED"
        elif predicted_zone == 'CAUTION':
             # Only alert caution if we haven't just alerted danger
             # The alert system handles cooldowns
             alert_sys.trigger_alert('caution')
             status_color = "YELLOW"
        else:
            status_color = "GREEN"

        # Display Status
        print(f"Step {i+1}: Loc({lat:.4f}, {lon:.4f}) | Zone: {predicted_zone} | {forecast_msg}")
        
        time.sleep(1) # Simulate time passing

    print("\nSimulation Complete.")

if __name__ == "__main__":
    from shapely.geometry import Point
    main()
