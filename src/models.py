import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model, save_model
import joblib
import os
from src.config import *

class ZoneClassifier:
    def __init__(self, model_type='knn'):
        self.model_type = model_type
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train(self, data):
        # Features: distance_to_imbl (could also use lat/lon but distance is more direct)
        # Using Lat/Lon is better for "geofencing" logic if boundary is complex.
        # But for this simulation, distance or Lat/Lon works. Let's use Lat/Lon for "realism"
        X = data[['lat', 'lon']]
        y = data['zone']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        print(f"Zone Classifier ({self.model_type}) Accuracy: {self.model.score(X_test_scaled, y_test)}")

    def predict(self, lat, lon):
        X_new = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)[0]
    
    def save(self, path='models/zone_model.pkl'):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        
    def load(self, path='models/zone_model.pkl'):
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.scaler = loaded['scaler']

class TrajectoryForecaster:
    def __init__(self, lookback=LSTM_LOOKBACK):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, lookback):
        X, y = [], []
        # data should be a numpy array of [lat, lon]
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback]) # Predict next step
        return np.array(X), np.array(y)

    def train(self, df):
        # We need to train per trajectory, or just concat all valid sequences.
        # Group by trip_id
        sequences_X = []
        sequences_y = []
        
        # Fit scaler on all data first
        coords = df[['lat', 'lon']].values
        self.scaler.fit(coords)
        coords_scaled = self.scaler.transform(coords)
        df_scaled = pd.DataFrame(coords_scaled, columns=['lat', 'lon'])
        df_scaled['trip_id'] = df['trip_id']
        
        for trip_id, group in df_scaled.groupby('trip_id'):
            trip_data = group[['lat', 'lon']].values
            if len(trip_data) > self.lookback:
                X, y = self.create_sequences(trip_data, self.lookback)
                sequences_X.extend(X)
                sequences_y.extend(y)
                
        X_train = np.array(sequences_X)
        y_train = np.array(sequences_y)
        
        # Build LSTM
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.lookback, 2)))
        self.model.add(Dense(2)) # Lat, Lon
        self.model.compile(optimizer='adam', loss='mse')
        
        print("Training LSTM...")
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
    def predict_next(self, recent_path):
        """
        recent_path: list of last 'lookback' (lat, lon) tuples/lists
        """
        if len(recent_path) != self.lookback:
            raise ValueError(f"Need exactly {self.lookback} points")
            
        input_seq = np.array(recent_path)
        input_scaled = self.scaler.transform(input_seq).reshape(1, self.lookback, 2)
        pred_scaled = self.model.predict(input_scaled, verbose=0)
        pred = self.scaler.inverse_transform(pred_scaled)
        return pred[0] # [lat, lon]
        
    def save(self, path='models/lstm_model.keras'):
        self.model.save(path)
        joblib.dump(self.scaler, 'models/lstm_scaler.pkl')
        
    def load(self, path='models/lstm_model.keras'):
        self.model = load_model(path)
        self.scaler = joblib.load('models/lstm_scaler.pkl')
