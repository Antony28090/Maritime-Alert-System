
# Constants for Maritime Boundary (IMBL)
# Using a simplified straight line segment for the boundary near Rameswaram/Dhanushkodi for simulation.
# Real coordinates would be more complex polygons.

# Reference Point (e.g., Shore)
REF_LAT = 9.2872
REF_LON = 79.3130

# Approximate IMBL Line (Caution: for simulation primarily)
IMBL_LAT_START = 10.0
IMBL_LON_START = 80.0
IMBL_LAT_END = 9.0
IMBL_LON_END = 79.5

# Zone Thresholds (Distance in km from IMBL)
DANGER_DIST_KM = 2.0   # < 2 km from border
CAUTION_DIST_KM = 5.0  # < 5 km from border (but > 2km)
# Safe is > 5 km

# Model Config
LSTM_LOOKBACK = 5  # Number of past steps to look at for prediction
PREDICT_STEPS = 6  # Predict next 6 steps (e.g., next 1 hour if step=10min)
