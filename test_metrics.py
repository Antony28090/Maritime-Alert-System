from src.validation import get_validation_metrics
import json

metrics = get_validation_metrics()
print("MSE:", metrics.get("lstm_mse"))
print("MAE:", metrics.get("lstm_mae"))
print("RMSE:", metrics.get("lstm_rmse"))
print("ADE:", metrics.get("lstm_ade"))
print("FDE:", metrics.get("lstm_fde"))
