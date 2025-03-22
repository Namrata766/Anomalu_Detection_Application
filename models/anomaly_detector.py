import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(real_time_df, baselines, composite_keys, value_keys):
    anomaly_results = []

    for _, row in real_time_df.iterrows():
        composite_values = tuple(row[key] for key in composite_keys)
        anomaly_status = "Match"
        anomaly_reason = ""

        for value_key in value_keys:
            expected_value = baselines.get((composite_values, value_key), None)
            if expected_value:
                actual_value = row[value_key]
                deviation = abs(actual_value - expected_value) / expected_value

                if deviation > 0.3:  # Threshold for significant deviation
                    anomaly_status = "Mismatch"
                    model = IsolationForest(contamination=0.05)
                    model.fit(real_time_df[[value_key]])
                    prediction = model.predict([[actual_value]])

                    if prediction[0] == -1:
                        anomaly_status = "Anomaly"
                        anomaly_reason = f"Deviation of {deviation * 100:.2f}% from expected baseline."

        row_data = row.to_dict()
        row_data.update({"Status": anomaly_status, "Anomaly Reason": anomaly_reason})
        anomaly_results.append(row_data)

    return pd.DataFrame(anomaly_results)
