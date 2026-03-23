
# Constants for Maritime Boundary (IMBL)
# Using a simplified straight line segment for the boundary near Rameswaram/Dhanushkodi for simulation.
# Real coordinates would be more complex polygons.

# Start Locations (Tamil Nadu Fishing Ports)
START_LOCATIONS = {
    "Rameshwaram Fishing Port": {"lat": 9.2876, "lon": 79.3129},
    "Pamban Port": {"lat": 9.2789, "lon": 79.2223},
    "Mandapam Fishing Harbour": {"lat": 9.2762, "lon": 79.1205},
    "Thondi": {"lat": 9.7423, "lon": 79.0197},
    "Mallipattinam": {"lat": 10.2796, "lon": 79.3150},
    "Jagathapattinam": {"lat": 9.9576, "lon": 79.1834},
    "Nagapattinam Port": {"lat": 10.7656, "lon": 79.8424},
    "Kodiakkarai (Point Calimere)": {"lat": 10.2975, "lon": 79.8516},
    "Thoothukudi Fishing Harbour": {"lat": 8.7952, "lon": 78.1610}
}

# Reference Point (Center of activity for map centering)
REF_LAT = 9.8 
REF_LON = 79.5

# Official IMBL Points (Palk Strait 1974 & Gulf of Mannar 1976)
# Format: [Lat, Lon]
IMBL_POINTS = [
    # Palk Strait (1974 Agreement)
    [10.0833, 80.0500], # Pos 1: 10° 05' N, 80° 03' E
    [9.9500, 79.5833],  # Pos 2: 09° 57' N, 79° 35' E
    [9.6692, 79.3767],  # Pos 3: 09° 40.15' N, 79° 22.60' E
    [9.3633, 79.5117],  # Pos 4: 09° 21.80' N, 79° 30.70' E
    [9.2167, 79.5333],  # Pos 5: 09° 13' N, 79° 32' E
    [9.1000, 79.5333],  # Pos 6: 09° 06' N, 79° 32' E (Junction)
    
    # Gulf of Mannar (1976 Agreement)
    [9.1000, 79.5333],  # Pos 1m: Same as Pos 6
    [9.0000, 79.5217],  # Pos 2m: 09°00'.0 N, 79°31'.3 E
    [8.8967, 79.4883],  # Pos 3m: 08°53'.8 N, 79°29'.3 E
    [8.6667, 79.3200],  # Pos 4m: 08°40'.0 N, 79°19'.4 E (Approx from source)
    [8.6233, 79.2167],  # Pos 5m: 08°37'.2 N, 79°13'.0 E (Approx)
    [8.5200, 79.0783],  # Pos 6m: 08°31'.2 N, 79°04'.7 E
    [8.3700, 78.9233],  # Pos 7m: 08°22'.2 N, 78°55'.4 E
    [8.2033, 78.8950]   # Pos 8m: 08°12'.2 N, 78°53'.7 E
    # Truncated further south for scope of Tamil Nadu coastal activity usually in this band
]

# Zone Thresholds (Distance in km from IMBL)
DANGER_DIST_KM = 2.0   # < 2 km from border
CAUTION_DIST_KM = 5.0  # < 5 km from border (but > 2km)
# Safe is > 5 km

# Model Config
LSTM_LOOKBACK = 5  # Number of past steps to look at for prediction
PREDICT_STEPS = 6  # Predict next 6 steps (e.g., next 1 hour if step=10min)
