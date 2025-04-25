# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:28:28 2025

@author: Pc
"""

# -*- coding: utf-8 -*-
"""
Vtech lasers & sensors
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import streamlit as st
import joblib
import os
import plotly.express as px

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
    df = pd.read_csv(file_path)

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
st.title("📡 LSTM Autoencoder Anomaly Detection (Multi-CSV Upload)")

uploaded_files = st.file_uploader("Upload one or more sensor CSV files:", type=['csv'], accept_multiple_files=True)
if uploaded_files:
    df_list = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, delimiter=';')
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                except:
                    pass
        df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s', errors='coerce')
        df_list.append(df)

    df_new = pd.concat(df_list, ignore_index=True).sort_values(by='timeStamp').reset_index(drop=True)

    rename_mapping = {
        'cur_1530': 'cur_1530',
        'cur_1310': 'cur_1310',
        'rec_ntc_1530': 'intpl_ntc_1530',
        'rec_ntc_1310': 'intpl_ntc_1310',
        'rawPd1': 'intpl_rawPd1',
        'rawPd2': 'intpl_rawPd2'
    }
    df_new.rename(columns=rename_mapping, inplace=True)

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
        fig1 = px.line(results_df, x='timeStamp', y=feature, title=f"{feature} Over Time", markers=True,
                      color_discrete_sequence=['#1f77b4'])  # Blue
        fig1.update_layout(xaxis_title="Timestamp", yaxis_title="Value", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Reconstruction Error Over Time")
        fig2 = px.line(results_df, x='timeStamp', y='reconstruction_error', title="Reconstruction Error with Anomalies",
                      color_discrete_sequence=['#1f77b4'])  # Red
        fig2.add_scatter(x=results_df['timeStamp'][anomalies], y=results_df['reconstruction_error'][anomalies],
                         mode='markers', name='Anomalies', marker=dict(color='red', symbol='x'))
        fig2.update_layout(xaxis_title="Timestamp", yaxis_title="Reconstruction Error", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        if st.checkbox("Show Anomaly Table"):
            st.dataframe(results_df[results_df['anomaly'] == True][['timeStamp', 'reconstruction_error', 'responsible_feature']])

        if st.checkbox("Show Feature-Level Errors"):
            import plotly.graph_objects as go
            fig3 = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, feat in enumerate(columns_to_use):
                fig3.add_trace(go.Scatter(
                    x=results_df['timeStamp'],
                    y=feature_errors_mean[:, i],
                    mode='lines',
                    name=feat,
                    line=dict(color=colors[i % len(colors)])
                ))
            fig3.update_layout(title="Feature-wise Reconstruction Errors",
                               xaxis_title="Timestamp",
                               yaxis_title="Error",
                               hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)


else:
    st.info("Please upload one or more CSV files to begin.")
