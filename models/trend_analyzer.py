import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def analyze_trends(df, composite_keys, value_keys, date_time_field):
    trend_data = {}

    for key in composite_keys:
        for value_key in value_keys:
            group_df = df.groupby(key).agg({value_key: 'mean'}).reset_index()
            if len(group_df) > 10:  # Ensure sufficient data points
                try:
                    decomposition = seasonal_decompose(group_df[value_key], model='additive', period=7)
                    trend_data[(key, value_key)] = decomposition.trend.dropna().values[-1]  # Last trend value
                except:
                    trend_data[(key, value_key)] = np.mean(group_df[value_key])  # Fallback average

    return trend_data
