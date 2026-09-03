import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.model_selection import train_test_split
from src.config import *
from src.geometry import distance_from_polyline, is_sri_lankan_side
from src.models import ZoneClassifier, TrajectoryForecaster


# Bounding box for classifier training samples — a rectangle covering the Palk
# Bay + Gulf of Mannar region so the classifier sees points at every distance
# from the boundary, not only the SAFE-heavy trajectories.
CLASSIFIER_LAT_MIN, CLASSIFIER_LAT_MAX = 8.0, 10.5
CLASSIFIER_LON_MIN, CLASSIFIER_LON_MAX = 78.5, 80.5
CLASSIFIER_N_SAMPLES = 30_000

def build_classifier_training_set(random_state=42):
    """
    Uniformly sample lat/lon in the Palk Bay / Gulf of Mannar region and
    label with the deterministic zone rule. This gives the KNN classifier
    a class-balanced view of the region — the labels come from the same
    geometry used at inference, so this is legitimate supervised learning
    of a known target function, not label leakage from the test set.
    """
    rng = np.random.default_rng(random_state)
    lats = rng.uniform(CLASSIFIER_LAT_MIN, CLASSIFIER_LAT_MAX, CLASSIFIER_N_SAMPLES)
    lons = rng.uniform(CLASSIFIER_LON_MIN, CLASSIFIER_LON_MAX, CLASSIFIER_N_SAMPLES)
    zones = []
    for la, lo in zip(lats, lons):
        d, _ = distance_from_polyline([la, lo], IMBL_POINTS)
        if is_sri_lankan_side([la, lo], IMBL_POINTS) or d < DANGER_DIST_KM:
            zones.append("DANGER")
        elif d < CAUTION_DIST_KM:
            zones.append("CAUTION")
        else:
            zones.append("SAFE")
    return pd.DataFrame({"lat": lats, "lon": lons, "zone": zones})

def get_zone_label(lat, lon):
    """
    Determines the zone label based on distance from IMBL.
    Reuse logic from data_generator.py/config.py
    """
    point = [lat, lon]
    dist, _ = distance_from_polyline(point, IMBL_POINTS)
    is_sl = is_sri_lankan_side(point, IMBL_POINTS)
    
    if is_sl:
        return "DANGER"
    elif dist < DANGER_DIST_KM:
        return "DANGER"
    elif dist < CAUTION_DIST_KM:
        return "CAUTION"
    else:
        return "SAFE"

def load_and_preprocess_data():
    print("Loading processed data...")
    processed_dir = os.path.join("data", "Processed")
    all_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {processed_dir}")
        
    df_list = []
    
    # We need to treat each file (daily) or each MMSI as a trajectory.
    # The files contain 'mmsi', 'date', 'hours', 'cell_ll_lat', 'cell_ll_lon', etc.
    # We should reconstruct trajectories based on MMSI and Time.
    # The raw data is daily aggregates, so it might not be a perfect time series "trajectory" 
    # in the sense of high-frequency updates, but we can treat the sequence of daily positions 
    # for a vessel as a trajectory.
    
    # However, the user wants to train the *existing* model which expects trajectories.
    # If the data is just daily points, the "trajectory" might be coarse.
    # Let's assume we can sort by date for each MMSI.
    
    for filename in all_files:
        df = pd.read_csv(filename)
        # Rename columns to match model expectations if needed
        # Expected: 'lat', 'lon', 'mmsi', 'trip_id' (we can use mmsi as trip_id)
        if 'cell_ll_lat' in df.columns:
            df = df.rename(columns={'cell_ll_lat': 'lat', 'cell_ll_lon': 'lon'})
            
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Generate Ground Truth Zones
    print("Generating ground truth zones...")
    # This might be slow for huge data. 
    # Vectorizing would be better but simple apply is easier to implement quickly.
    # Using a small sample for testing if needed, but let's try full.
    full_df['zone'] = full_df.apply(lambda row: get_zone_label(row['lat'], row['lon']), axis=1)
    
    # Create a 'trip_id' equivalent. 
    # Since MMSI tracks a vessel, and we have multiple days, we can treat (MMSI) as a trip.
    # But a vessel might operate for months. 
    # Ideally we'd split by long gaps, but for now, MMSI = Trip.
    full_df['trip_id'] = full_df['mmsi']
    
    # Sort by mmsi and date to ensure sequence
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values(by=['mmsi', 'date'])
    
    print(f"Total rows: {len(full_df)}")
    print(f"Zone distribution:\n{full_df['zone'].value_counts()}")
    
    return full_df

def main():
    try:
        # --- 1. Train Zone Classifier (balanced uniform sample) ---
        print("\n--- Training Zone Classifier (uniform grid sample of region) ---")
        clf_data = build_classifier_training_set()
        print("Class distribution:")
        print(clf_data['zone'].value_counts())

        zone_clf = ZoneClassifier(model_type='knn')
        zone_clf.train(clf_data)
        zone_clf.save()
        print("Zone Classifier saved.")
        
        # --- 2. Train Trajectory Forecaster (SYNTHETIC DATA) ---
        print("\n--- Training Trajectory Forecaster (Synthetic Real-Seeded Data) ---")
        
        # Import generator here to avoid circular imports if any, and ensure it uses updated logic
        from src.data_generator import generate_trajectory, load_real_seed_data
        
        # Ensure seeds are loaded
        load_real_seed_data()
        
        print("Generating synthetic trajectories...")
        synthetic_data = []
        # Generate enough data for training
        N_TRIPS = 200 # More trips for better generalization
        for i in range(N_TRIPS):
            # 50% Force Crossing to learn danger boundaries
            trajectory = generate_trajectory(trip_id=10000+i, n_points=50, force_crossing=(i%2==0))
            synthetic_data.extend(trajectory)
            
        syn_df = pd.DataFrame(synthetic_data)
        print(f"Generated {len(syn_df)} synthetic points.")
        
        # Train LSTM on Synthetic Data
        lstm_forecaster = TrajectoryForecaster(lookback=LSTM_LOOKBACK)
        
        # Split Synthetic Data
        mmsis = syn_df['trip_id'].unique()
        train_mmsis, test_mmsis = train_test_split(mmsis, test_size=0.2, random_state=42)
        
        # Persist the split so src/evaluate.py can score one-step forecast error
        # strictly on held-out trips (no in-sample numbers in the paper).
        syn_df['split'] = np.where(syn_df['trip_id'].isin(train_mmsis), 'train', 'test')

        train_data = syn_df[syn_df['trip_id'].isin(train_mmsis)]
        test_data = syn_df[syn_df['trip_id'].isin(test_mmsis)]
        
        print(f"LSTM Training Data: {len(train_data)} rows")
        
        lstm_forecaster.train(train_data)
        lstm_forecaster.save()
        print("LSTM Model saved.")
        
        # Evaluation
        print("Evaluating LSTM on Test Set...")
        sequences_X = []
        sequences_y = []
        
        test_coords = test_data[['lat', 'lon']].values
        test_coords_scaled = lstm_forecaster.scaler.transform(test_coords)
        
        test_data_scaled = pd.DataFrame(test_coords_scaled, columns=['lat', 'lon'])
        test_data_scaled['trip_id'] = test_data['trip_id'].values 
        
        for trip_id, group in test_data_scaled.groupby('trip_id'):
            trip_vals = group[['lat', 'lon']].values
            if len(trip_vals) > LSTM_LOOKBACK:
                X, y = lstm_forecaster.create_sequences(trip_vals, LSTM_LOOKBACK)
                sequences_X.extend(X)
                sequences_y.extend(y)
                
        if sequences_X:
            X_test = np.array(sequences_X)
            y_test = np.array(sequences_y)
            loss = lstm_forecaster.model.evaluate(X_test, y_test, verbose=0)
            print(f"LSTM Test Loss (MSE): {loss}")
            print(f"RMSE: {np.sqrt(loss)}")
            
            # Save synthetic data for validation script to pick up if needed?
            # Validation script loads 'data/vessel_data.csv'. 
            # We should overwrite it with this high-quality synthetic data so validation uses it.
            syn_df.to_csv('data/vessel_data.csv', index=False)
            print("Saved synthetic dataset to data/vessel_data.csv for validation.")
            
        else:
            print("Not enough data in test set.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
