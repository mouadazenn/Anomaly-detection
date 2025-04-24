# -*- coding: utf-8 -*-
"""
Streamlit App - Adapted to cleaned ammonia sensor CSV with European-style decimals and column validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os

# ==== CONFIGURATION ==== 
time_steps = 30
columns_to_use = [
    'cur_1530', 'cur_1310',
    'intpl_ntc_1530', 'intpl_ntc_1310',
    'intpl_rawPd1', 'intpl_rawPd2'
]

# ==== TRAINING ON CLEAN DATA ==== 
if not os.path.exists("lstm_model.keras"):
    file_path = "train_data.csv"
    df = pd.read_csv(file_path, delimiter=';')

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass

    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    train_data = df[columns_to_use].dropna().reset_index(drop=True)

    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(train_data)
    joblib.dump(scaler, "scaler.save")

    def create_sequences(data, time_steps):
        return np.array([data[i:i + time_steps] for i in range(len(data) - time_steps)])

    X_train = create_sequences(normalized_train, time_steps)

    input_dim = X_train.shape[2]
    input_layer = Input(shape=(time_steps, input_dim))
    encoded = LSTM(64, activation='relu')(input_layer)
    decoded = RepeatVector(time_steps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    output_layer = TimeDistributed(Dense(input_dim))(decoded)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)
    model.save("lstm_model.keras")

# ==== DETECTION ON NEW DATA ==== 
st.title("📡 LSTM Autoencoder Anomaly Detection (CSV Upload)")

uploaded_file = st.file_uploader("Upload new sensor CSV file:", type=['csv'])
if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file, delimiter=';')

    for col in df_new.columns:
        if df_new[col].dtype == 'object':
            try:
                df_new[col] = df_new[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass

    df_new['timeStamp'] = pd.to_datetime(df_new['timeStamp'],unit = 's')


    # Rename uploaded columns to match training schema
    rename_mapping = {
        'cur_1530': 'cur_1530',
        'cur_1310': 'cur_1310',
        'rec_ntc_1530': 'intpl_ntc_1530',
        'rec_ntc_1310': 'intpl_ntc_1310',
        'rawPd1': 'intpl_rawPd1',
        'rawPd2': 'intpl_rawPd2'
    }
    df_new.rename(columns=rename_mapping, inplace=True)

    # Ensure all required columns exist
    missing_columns = [col for col in columns_to_use if col not in df_new.columns]
    if missing_columns:
        st.error(f"Missing columns in uploaded file: {missing_columns}")
    else:
        sensor_data = df_new[columns_to_use].dropna().reset_index(drop=True)
        scaler = joblib.load("scaler.save")
        sensor_data = sensor_data[scaler.feature_names_in_.tolist()]
        normalized_data = scaler.transform(sensor_data)

        def create_sequences(data, time_steps):
            return np.array([data[i:i + time_steps] for i in range(len(data) - time_steps)])

        X = create_sequences(normalized_data, time_steps)
        model = load_model("lstm_model.keras")
        X_pred = model.predict(X)

        reconstruction_error = np.mean((X_pred - X) ** 2, axis=(1, 2))
        feature_errors = (X_pred - X) ** 2
        feature_errors_mean = feature_errors.mean(axis=1)
        top_feature_indices = feature_errors_mean.argmax(axis=1)
        responsible_features = [columns_to_use[i] for i in top_feature_indices]
        threshold = np.percentile(reconstruction_error, 98)
        anomalies = reconstruction_error > threshold

        clean_timestamps = df_new.loc[df_new[columns_to_use].dropna().index, 'timeStamp'].reset_index(drop=True)
        timestamp_sequence = clean_timestamps.iloc[time_steps:].reset_index(drop=True)

        results_df = pd.DataFrame({
            'reconstruction_error': reconstruction_error,
            'anomaly': anomalies,
            'timeStamp': timestamp_sequence,
            'responsible_feature': responsible_features
        })

        for col in columns_to_use:
            results_df[col] = sensor_data[col].iloc[time_steps:].reset_index(drop=True)

        feature = st.selectbox("Select a feature to view:", columns_to_use)

        st.subheader(f"{feature} Over Time")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(results_df['timeStamp'], results_df[feature], label=feature)
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Value")
        st.pyplot(fig1)

        st.subheader("Reconstruction Error Over Time")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(results_df['timeStamp'], results_df['reconstruction_error'], label='Reconstruction Error')
        anomaly_points = results_df[results_df['anomaly'] == True]
        ax2.scatter(anomaly_points['timeStamp'], anomaly_points['reconstruction_error'], color='red', marker='x', label='Anomaly')
        ax2.axhline(y=threshold, color='orange', linestyle='--', label='Threshold')
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Reconstruction Error")
        ax2.legend()
        st.pyplot(fig2)

        if st.checkbox("Show Anomaly Table"):
            st.dataframe(anomaly_points[['timeStamp', 'reconstruction_error', 'responsible_feature']])

        if st.checkbox("Show Feature-Level Errors"):
            fig3, ax3 = plt.subplots(figsize=(14, 5))
            for i, feat in enumerate(columns_to_use):
                ax3.plot(results_df['timeStamp'], feature_errors_mean[:, i], label=feat)
            ax3.set_title("Feature-wise Reconstruction Errors")
            ax3.set_xlabel("Timestamp")
            ax3.set_ylabel("Error")
            ax3.legend()
            st.pyplot(fig3)
else:
    st.info("Please upload a CSV file containing sensor data.")
