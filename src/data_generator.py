import numpy as np
import pandas as pd
import random
from shapely.geometry import Point, LineString
from src.config import *

from src.geometry import distance_from_polyline, is_sri_lankan_side


# Global to store real data points for seeding
REAL_DATA_POINTS = []

def load_real_seed_data():
    """Loads processed CSVs to build a list of valid start locations."""
    global REAL_DATA_POINTS
    if REAL_DATA_POINTS: return # Already loaded
    
    import glob
    import os
    print("Loading real data for seeding...")
    files = glob.glob("data/Processed/*.csv")
    points = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Sample a fraction to avoid huge memory usage if necessary, 
            # but usually fine to keep all for density estimation.
            # We just need [lat, lon] pairs.
            if 'lat' in df.columns:
                points.extend(df[['lat', 'lon']].values.tolist())
            elif 'cell_ll_lat' in df.columns:
                points.extend(df[['cell_ll_lat', 'cell_ll_lon']].values.tolist())
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if points:
        REAL_DATA_POINTS = points
        print(f"Loaded {len(REAL_DATA_POINTS)} real seed points.")
    else:
        print("Warning: No real data found. Using default locations.")

def get_zone(dist_km):
    if dist_km < DANGER_DIST_KM:
        return "DANGER"
    elif dist_km < CAUTION_DIST_KM:
        return "CAUTION"
    else:
        return "SAFE"

def generate_trajectory(trip_id, n_points=100, force_crossing=False):
    """Generates a realistic vessel path with drift, steering noise, and sensor noise."""
    # 1. Define Start and Target
    
    # RESEARCH METHODOLOGY: Real-Seeded *Target*
    # We want ships to go where they actually fish.
    if not REAL_DATA_POINTS:
        load_real_seed_data()

    # EXACTLY Start from a Coastal Port (Tamil Nadu)
    loc_name = random.choice(list(START_LOCATIONS.keys()))
    base_loc = START_LOCATIONS[loc_name]
    
    start_lat = base_loc["lat"]
    start_lon = base_loc["lon"]

    if REAL_DATA_POINTS and random.random() < 0.9: # 90% chance to target a real fishing spot
        # Pick a random point from real history as the DESTINATION
        seed_pt = random.choice(REAL_DATA_POINTS)
        imbl_target_lat = seed_pt[0]
        imbl_target_lon = seed_pt[1]
        
        # Calculate target vector
        # If the point is too close to start, pick another or push it out
        if abs(imbl_target_lat - start_lat) < 0.05 and abs(imbl_target_lon - start_lon) < 0.05:
             # Fallback to random IMBL point if too close
             segment = random.choice(IMBL_POINTS)
             imbl_target_lat, imbl_target_lon = segment[0], segment[1]
    else:
        # Fallback: Target random IMBL segment
        segment = random.choice(IMBL_POINTS)
        imbl_target_lat = segment[0]
        imbl_target_lon = segment[1]
    
    # Target: Pick a random segment from IMBL_POINTS and target it
    # This ensures vessels move towards the boundary
    seg_idx = random.randint(0, len(IMBL_POINTS) - 2)
    p1 = IMBL_POINTS[seg_idx]
    p2 = IMBL_POINTS[seg_idx+1]
    
    # Random point on this segment
    t = random.random()
    imbl_target_lat = p1[0] + t * (p2[0] - p1[0])
    imbl_target_lon = p1[1] + t * (p2[1] - p1[1])
    
    if force_crossing:
        # Calculate perpendicular vector to push target BEYOND the line
        dx = p2[1] - p1[1]
        dy = p2[0] - p1[0]
        
        # Perpendicular vector (pointing generally East/South)
        perp_lon = dy 
        perp_lat = -dx
        
        # Normalize
        length = np.sqrt(perp_lon**2 + perp_lat**2)
        if length > 0:
            perp_lon /= length
            perp_lat /= length
            
        # Push 3-5 km beyond line
        push_dist = random.uniform(0.03, 0.05)
        
        # Ensure we are pushing towards East (positive Lon change usually for SL)
        if perp_lon < 0:
            perp_lon = -perp_lon
            perp_lat = -perp_lat
            
        target_lat = imbl_target_lat + (perp_lat * push_dist)
        target_lon = imbl_target_lon + (perp_lon * push_dist)
        
    else:
        # Target is slightly variated from that point
        target_lat = imbl_target_lat + random.uniform(-0.05, 0.05)
        target_lon = imbl_target_lon + random.uniform(-0.05, 0.05)
        
    # [NEW] Navigation Checkpoints (Avoid Land)
    # Point Calimere (Vedaranyam) is approx 10.29N, 79.85E.
    # If starting North of 10.3 and ending South of 10.3, must pass East of 79.9
    waypoints = []
    if start_lat > 10.3 and target_lat < 10.3:
        # Route via East of Point Calimere
        waypoints.append([10.30, 80.0]) # Waypoint 1: Clear the cape
        
    waypoints.append([target_lat, target_lon]) # Final target
    
    current_waypoint_idx = 0
    sub_target_lat, sub_target_lon = waypoints[0]

    
    # 2. Physics / Movement Parameters
    current_lat, current_lon = start_lat, start_lon
    
    # Calculate initial bearing to target
    d_lat = target_lat - start_lat
    d_lon = target_lon - start_lon
    target_heading = np.arctan2(d_lat, d_lon) # Radians
    
    current_heading = target_heading
    
    # Speed setup
    base_speed = 0.008 
    if force_crossing:
        base_speed = 0.012 # Move faster for testing
    
    # Environmental Drift (Currents/Wind) - Constant for the trip
    drift_lat = np.random.normal(0, 0.0002) 
    drift_lon = np.random.normal(0, 0.0002)
    
    data = []
    
    for i in range(n_points):
        # A. Update Heading (Steering)
        # Check if we reached current waypoint (within ~1km)
        dist_to_wp = np.sqrt((sub_target_lat - current_lat)**2 + (sub_target_lon - current_lon)**2)
        if dist_to_wp < 0.01: # approx 1km
            current_waypoint_idx += 1
            if current_waypoint_idx < len(waypoints):
                sub_target_lat, sub_target_lon = waypoints[current_waypoint_idx]
            else:
                # Reached final target, just drift or circle? 
                # Keep aiming at final target
                pass

        # Calculate bearing to CURRENT waypoint
        d_lat = sub_target_lat - current_lat
        d_lon = sub_target_lon - current_lon
        desired_heading = np.arctan2(d_lat, d_lon)
        
        # Heading assumes a "weighted average" between current and desired (inertia) 
        # plus some random steering noise
        # 0.8 * old + 0.2 * new is smooth turn
        
        # Handle angle wrapping? For small area/angles, simple mix is okay usually, 
        # but technically should handle wraparound PI/-PI. 
        # Given the localized area, angles won't jump wildly across pi boundary usually.
        
        # If forcing crossing, steer tighter to target
        steer_factor = 0.2 if not force_crossing else 0.5
        
        current_heading = (1.0 - steer_factor) * current_heading + steer_factor * desired_heading
        current_heading += np.random.normal(0, 0.05) # Random steering wiggle (radians)
        
        # B. Update Speed
        # Speed varies slightly (waves, throttle)
        step_speed = base_speed * np.random.normal(1.0, 0.1)
        
        # C. Update Position (Physics)
        # Move in direction of heading
        delta_lat = step_speed * np.sin(current_heading)
        delta_lon = step_speed * np.cos(current_heading)
        
        # Add drift
        current_lat += delta_lat + drift_lat
        current_lon += delta_lon + drift_lon
        
        # D. Add Sensor Noise (GPS Error)
        # GPS error is usually few meters, so much smaller than movement
        # 0.00005 deg ~= 5 meters
        recorded_lat = current_lat + np.random.normal(0, 0.00005)
        recorded_lon = current_lon + np.random.normal(0, 0.00005)
        
        # E. Calculate Zone
        # Pass [lat, lon] to distance_from_polyline
        dist, _ = distance_from_polyline([recorded_lat, recorded_lon], IMBL_POINTS)
        is_sl_side = is_sri_lankan_side([recorded_lat, recorded_lon], IMBL_POINTS)
        
        if is_sl_side:
            zone = "DANGER" # Always danger if on the wrong side
        else:
            zone = get_zone(dist)
        
        data.append({
            'trip_id': trip_id,
            'timestamp': i,
            'lat': recorded_lat,
            'lon': recorded_lon,
            'distance_to_imbl': dist,
            'zone': zone
        })
        
    return data

def main():
    print("Generating synthetic data...")
    all_data = []
    
    # Generate 500 trips total
    # 50% Normal, 50% Forced Crossing (to learn the aggressive behavior)
    n_trips = 500
    
    for trip in range(n_trips):
        # Alternate between normal and forced
        is_forced = (trip % 2 == 0) 
        all_data.extend(generate_trajectory(trip_id=trip, force_crossing=is_forced))
        
    df = pd.DataFrame(all_data)
    df.to_csv('data/vessel_data.csv', index=False)
    print(f"Data saved to data/vessel_data.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()
