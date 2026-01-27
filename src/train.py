import pandas as pd
import os
from src.models import ZoneClassifier, TrajectoryForecaster

def main():
    if not os.path.exists('data/vessel_data.csv'):
        print("Data not found. Running generator first...")
        import src.data_generator
        src.data_generator.main()
        
    print("Loading data...")
    df = pd.read_csv('data/vessel_data.csv')
    
    # 1. Train Zone Classifier
    print("\n--- Training Zone Classifier ---")
    zone_clf = ZoneClassifier()
    zone_clf.train(df)
    zone_clf.save()
    print("Zone Classifier saved.")
    
    # 2. Train LSTM
    print("\n--- Training LSTM Forecaster ---")
    lstm = TrajectoryForecaster()
    lstm.train(df)
    lstm.save()
    print("LSTM Forecaster saved.")

if __name__ == "__main__":
    main()
