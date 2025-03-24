import pandas as pd
from models.trend_analyzer import analyze_trends
from models.anomaly_detector import detect_anomalies

def reconcile_data(historical_df, real_time_df, composite_keys, value_keys, date_time_field):
    composite_keys = [col.strip() for col in composite_keys.split(",")]
    value_keys = [col.strip() for col in value_keys.split(",")]

    # Convert date field to datetime
    historical_df[date_time_field] = pd.to_datetime(historical_df[date_time_field])
    real_time_df[date_time_field] = pd.to_datetime(real_time_df[date_time_field])

    # Identify Trends & Baselines
    baselines = analyze_trends(historical_df, composite_keys, value_keys, date_time_field)

    # Compare & Detect Anomalies (passing date_time_field)
    anomalies = detect_anomalies(real_time_df, historical_df, composite_keys, value_keys, date_time_field)

    return anomalies
