import numpy as np
import pandas as pd
import random
from shapely.geometry import Point, LineString
from src.config import *

def distance_from_line(point, line_start, line_end):
    """Calculates perpendicular distance from a point to the IMBL line segment."""
    line = LineString([line_start, line_end])
    return line.distance(point) * 111.32  # Approx conversion deg to km

def get_zone(dist_km):
    if dist_km < DANGER_DIST_KM:
        return "DANGER"
    elif dist_km < CAUTION_DIST_KM:
        return "CAUTION"
    else:
        return "SAFE"

def generate_trajectory(trip_id, n_points=50):
    """Generates a vessel path moving towards the boundary."""
    # Start somewhere in safe zone
    start_lat = REF_LAT + random.uniform(-0.1, 0.1)
    start_lon = REF_LON + random.uniform(-0.1, 0.1)
    
    # Target somewhere near or across the boundary
    # Interpolate towards IMBL center
    target_lat = (IMBL_LAT_START + IMBL_LAT_END) / 2 + random.uniform(-0.1, 0.1)
    target_lon = (IMBL_LON_START + IMBL_LON_END) / 2 + random.uniform(-0.1, 0.1)
    
    lats = np.linspace(start_lat, target_lat, n_points)
    lons = np.linspace(start_lon, target_lon, n_points)
    
    # Add noise
    lats += np.random.normal(0, 0.002, n_points)
    lons += np.random.normal(0, 0.002, n_points)
    
    data = []
    line_start = (IMBL_LON_START, IMBL_LAT_START) # Shapely (x, y) = (lon, lat)
    line_end = (IMBL_LON_END, IMBL_LAT_END)
    
    for i in range(n_points):
        p = Point(lons[i], lats[i])
        # Simple distance check (ignoring sign for now, assuming we are on Indian side)
        # For simplicity, we just take distance to the line segment
        dist = distance_from_line(p, line_start, line_end)
        
        zone = get_zone(dist)
        
        data.append({
            'trip_id': trip_id,
            'timestamp': i, # Relative timestamp
            'lat': lats[i],
            'lon': lons[i],
            'distance_to_imbl': dist,
            'zone': zone
        })
        
    return data

def main():
    print("Generating synthetic data...")
    all_data = []
    for trip in range(100): # Generate 100 trips
        all_data.extend(generate_trajectory(trip_id=trip))
        
    df = pd.DataFrame(all_data)
    df.to_csv('data/vessel_data.csv', index=False)
    print(f"Data saved to data/vessel_data.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()
