import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(real_time_df, historical_df, composite_keys, value_keys, date_key):
    anomaly_results = []
    baselines = compute_baselines(historical_df, composite_keys, value_keys, date_key)

    for _, row in real_time_df.iterrows():
        composite_values = tuple(row[key] for key in composite_keys)
        anomaly_status = "Match"
        anomaly_reason = ""

        for value_key in value_keys:
            expected_value = baselines.get((composite_values, value_key), None)
            actual_value = row[value_key]

            if expected_value is not None:
                deviation = abs(actual_value - expected_value) / expected_value if expected_value != 0 else 0

                # Fetch historical data for the same composite key
                historical_values = historical_df.loc[
                    historical_df.set_index(composite_keys).index == composite_values, value_key
                ]

                if not historical_values.empty:
                    model = IsolationForest(contamination=0.1, random_state=42)
                    model.fit(historical_values.values.reshape(-1, 1))
                    prediction = model.predict([[actual_value]])

                    if prediction[0] == -1 or deviation > 0.10:  # Reduce threshold for more anomalies
                        anomaly_status = "Anomaly"
                        anomaly_reason = f"Deviation of {deviation * 100:.2f}% from baseline."

                # 🚨 Hard threshold for extreme deviations (>50%)
                if deviation > 0.50:
                    anomaly_status = "Critical Anomaly"
                    anomaly_reason = f"Extreme deviation of {deviation * 100:.2f}% from expected baseline!"

        row_data = row.to_dict()
        row_data.update({"Status": anomaly_status, "Anomaly Reason": anomaly_reason})
        anomaly_results.append(row_data)

    return pd.DataFrame(anomaly_results)


def compute_baselines(historical_df, composite_keys, value_keys, date_key):
    baselines = {}
    grouped = historical_df.groupby(composite_keys)

    for composite_values, group in grouped:
        for value_key in value_keys:
            trend_baseline = group.sort_values(by=date_key)[value_key].median()  # Median ensures stable trends
            baselines[(composite_values, value_key)] = trend_baseline

    return baselines
