import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.reconciliation import reconcile_data
import os

# App Title
st.title("📊 Data Reconciliation & Anomaly Detection")

# Upload Historical & Real-time Data
st.sidebar.header("Upload Data")
historical_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])
real_time_file = st.sidebar.file_uploader("Upload Real-time Data (CSV)", type=["csv"])

# User Inputs
composite_keys = st.sidebar.text_input("Enter Composite Keys (comma-separated)")
value_keys = st.sidebar.text_input("Enter Value Keys (comma-separated)")
date_time_field = st.sidebar.text_input("Enter Date/Time Field Name")

if st.sidebar.button("Process Data"):
    if historical_file and real_time_file and composite_keys and value_keys and date_time_field:
        # Load Data
        historical_data = load_data(historical_file)
        real_time_data = load_data(real_time_file)

        # Process Data
        result_df = reconcile_data(historical_data, real_time_data, composite_keys, value_keys, date_time_field)

        # Save & Display
        output_path = "output/reconciled_data.csv"
        result_df.to_csv(output_path, index=False)

        st.success("✅ Processing Complete!")
        st.download_button(label="Download Reconciled Data", data=result_df.to_csv(index=False),
                           file_name="reconciled_data.csv")
        st.dataframe(result_df)
    else:
        st.error("⚠️ Please upload both CSV files and provide necessary inputs.")
