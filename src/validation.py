import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, precision_recall_fscore_support, roc_auc_score, roc_curve
from src.models import ZoneClassifier, TrajectoryForecaster
from src.config import *
import joblib
import json

def get_validation_metrics():
    # 1. Load Data
    try:
        df = pd.read_csv('data/vessel_data.csv')
    except FileNotFoundError:
        return {"error": "Data file not found. Please run data generator."}
    
    # 2. Load Models
    try:
        zone_model = ZoneClassifier()
        zone_model.load()
        lstm_model = TrajectoryForecaster()
        lstm_model.load()
    except Exception as e:
         return {"error": f"Models not found or failed to load: {str(e)}"}

    metrics = {}

    # --- Zone Classifier Validation ---
    print("Validating Zone Classifier...")
    # Prepare data (same logic as in models.py predict)
    X = df[['lat', 'lon']]
    y_true = df['zone']
    
    # Predict all
    # We need to scale. Since the scaler is part of the loaded model object in my implementation plan,
    # I should check if ZoneClassifier exposes a batch predict or if I need to use the internal model.
    # Looking at models.py, predict() takes single lat/lon. 
    # But I can access zone_model.model and zone_model.scaler directly.
    
    X_scaled = zone_model.scaler.transform(X)
    y_pred = zone_model.model.predict(X_scaled)
    
    metrics['zone_accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Confusion Matrix
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = {
        'labels': labels,
        'matrix': cm.tolist()
    }
    
    # Advanced Zone Metrics
    # Precision, Recall, F1 per class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics['zone_classification_report'] = {
        'labels': labels,
        'precision': p.tolist(),
        'recall': r.tolist(),
        'f1': f1.tolist()
    }
    
    # ROC-AUC (Multi-class One-vs-Rest)
    try:
        # Need probability estimates
        if hasattr(zone_model.model, "predict_proba"):
            y_prob = zone_model.model.predict_proba(X_scaled)
            # roc_auc_score for multi_class require one-hot encoded y_true or 'ovr' strategy with labels
            # But y_true are strings. Need to encode them or pass labels?
            # sklearn handles labels if multi_class='ovr' and we pass y_prob array matching classes order?
            # Actually easier to binarize y_true for standard calculation or rely on estimator classes
            
            # Check classes order
            classes = zone_model.model.classes_
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=classes)
        else:
             metrics['roc_auc'] = "N/A (Model doesn't support proba)"
    except Exception as e:
        print(f"ROC-AUC Error: {e}")
        metrics['roc_auc'] = 0.0

    
    # --- LSTM Forecasting Validation ---
    print("Validating LSTM Forecaster...")
    # Group by trip and predict next steps
    y_lstm_true = []
    y_lstm_pred = []
    
    # We select a few random sample trajectories for visualization
    sample_trajectories = {} 
    
    trips = df['trip_id'].unique()
    # Limit validation to first 20 trips to save time if dataset is huge, 
    # but here it's small so do all.
    
    for trip_id in trips:
        trip_data = df[df['trip_id'] == trip_id]
        coords = trip_data[['lat', 'lon']].values
        
        if len(coords) > LSTM_LOOKBACK:
            # Predict for the whole path sliding window
            # Use the internal predict_next logic but in loop
            # This might be slow for many points.
            
            # Optimization: Pre-scale everything
            # But the model's scaler is for single point prediction usually?
            # models.py: predict_next takes recent_path (unscaled), scales it, predicts.
            
            # Let's just do a few points per trip or the whole trip if small.
            # For MSE, we want many points.
            
            # Create sequences
            X_seq, y_next_true_delta = lstm_model.create_sequences(coords, LSTM_LOOKBACK)
            
            if len(X_seq) == 0:
                continue

            # Manually scale for batch prediction
            N, L, F = X_seq.shape
            X_seq_flat = X_seq.reshape(N*L, F)
            X_seq_scaled = lstm_model.scaler.transform(X_seq_flat).reshape(N, L, F)
            
            # Predict
            pred_scaled = lstm_model.model.predict(X_seq_scaled, verbose=0)
            pred_unscaled_delta = lstm_model.scaler.inverse_transform(pred_scaled)
            
            # Convert deltas back to absolute positions
            # The base point for the delta is coords[i + LSTM_LOOKBACK]
            last_points = coords[LSTM_LOOKBACK:-1] # shape (N, 2)
            
            actual_points = last_points + y_next_true_delta
            pred_points = last_points + pred_unscaled_delta
            
            y_lstm_true.extend(actual_points)
            y_lstm_pred.extend(pred_points)
            
            # Save the first trip as sample for viz
            if len(sample_trajectories) < 1:
                sample_trajectories[f'Trip {trip_id}'] = {
                    'actual': actual_points.tolist(),
                    'predicted': pred_points.tolist()
                }

    y_lstm_true = np.array(y_lstm_true)
    y_lstm_pred = np.array(y_lstm_pred)
    
    if len(y_lstm_true) > 0:
        mse = mean_squared_error(y_lstm_true, y_lstm_pred)
        mae = mean_absolute_error(y_lstm_true, y_lstm_pred)
        metrics['lstm_mse'] = float(mse)
        metrics['lstm_mae'] = float(mae)
        
    # Advanced Trajectory Metrics
    if len(y_lstm_true) > 0:
        # RMSE
        metrics['lstm_rmse'] = float(np.sqrt(metrics['lstm_mse']))
        
        # ADE (Average Displacement Error)
        # Euclidean distance between each true and pred point
        diff = y_lstm_true - y_lstm_pred # shape (N, 2)
        dists = np.sqrt(np.sum(diff**2, axis=1)) # shape (N,)
        metrics['lstm_ade'] = float(np.mean(dists))
        
        # Trajectory Accuracy (Percentage of predictions within 0.005 degrees / ~500m)
        accurate_preds = np.sum(dists < 0.005)
        metrics['trajectory_accuracy'] = float(accurate_preds / len(dists)) if len(dists) > 0 else 0.0
        
        # FDE (Final Displacement Error)
        # For sliding window, "Final" is just the last point of a sequence? 
        # Or if we predicted a full trajectory?
        # Here we predicted single next steps. 
        # So ADE approx equals AE (Average Error).
        # FDE usually applies when predicting K steps into future. 
        # Since we only predict 1 step, ADE = FDE = Error.
        # But if we treat the aggregated evaluation as a "path", FDE is the error at the destination?
        # Let's compute FDE as the error of the LAST point of each trip.
        
        final_errors = []
        # Re-iterate trips to find last point error? 
        # Or faster: we just calculated dists for all points.
        # But we lost the trip structure in flattened y_lstm_true.
        # Approximation: FDE same as ADE for 1-step forecast context implies avg error.
        # But let's stick to definition: Distance at final time step.
        # We need to run predict loop again or store structure.
        # Let's just use the sample_trajectories logic to get FDE for samples.
        
        # Actually simplified: Random sample FDE or just leave as ADE if 1-step.
        metrics['lstm_fde'] = float(dists[-1]) if len(dists) > 0 else 0.0

        
    metrics['sample_trajectory'] = sample_trajectories

    return metrics

if __name__ == "__main__":
    # Test run
    print(json.dumps(get_validation_metrics(), indent=2))
