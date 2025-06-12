# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:03:01 2025

@author: Pc
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

st.set_page_config(layout="wide")

# === STREAMLIT UI ===
st.title("📈 Interactive Sensor Signal Analysis")

uploaded_files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

# === Function: Merge uploaded CSVs ===
def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to read {file.name}: {e}")
    if not dfs:
        return None
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# === Signal Processing Functions ===
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_fft(signal, fs):
    N = len(signal)
    fft_values = rfft(signal)
    freqs = rfftfreq(N, d=1/fs)
    return freqs, np.abs(fft_values)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded and merged")

    # Show raw data preview
    with st.expander("🔍 Preview Merged Data"):
        st.write(df.head())

    labels = df.columns.tolist()
    time_col = st.selectbox("Select time column", options=labels)
    pd1_col = st.selectbox("Select Pd1 column", options=labels)
    pd2_col = st.selectbox("Select Pd2 column", options=labels)

    # Time range selection
    time_data = df[time_col]
    try:
        t = pd.to_datetime(time_data)
        t = (t - t.iloc[0]).dt.total_seconds()
    except:
        t = pd.to_numeric(time_data, errors='coerce')

    df['__time__'] = t
    fs = len(t) / (t.iloc[-1] - t.iloc[0])

    start, end = st.slider("Select time range (in seconds)", float(t.min()), float(t.max()), (float(t.min()), float(t.max())))
    mask = (df['__time__'] >= start) & (df['__time__'] <= end)

    pd1 = pd.to_numeric(df.loc[mask, pd1_col], errors='coerce')
    pd2 = pd.to_numeric(df.loc[mask, pd2_col], errors='coerce')
    t_selected = df.loc[mask, '__time__']

    # Apply filter
    lowcut = st.number_input("Low cutoff frequency (Hz)", value=0.8)
    highcut = st.number_input("High cutoff frequency (Hz)", value=16.0)

    filtered_pd1 = bandpass_filter(pd1, lowcut, highcut, fs)
    filtered_pd2 = bandpass_filter(pd2, lowcut, highcut, fs)

    # === PLOTS ===
    st.subheader("📉 Time Domain Signals")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t_selected, y=pd1, mode='lines', name='Original Pd1'))
    fig_time.add_trace(go.Scatter(x=t_selected, y=filtered_pd1, mode='lines', name='Filtered Pd1'))
    fig_time.add_trace(go.Scatter(x=t_selected, y=pd2, mode='lines', name='Original Pd2'))
    fig_time.add_trace(go.Scatter(x=t_selected, y=filtered_pd2, mode='lines', name='Filtered Pd2'))
    fig_time.update_layout(title="Time Domain Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=pd1, y=pd2, mode='markers', name='Pd1 vs Pd2', marker=dict(opacity=0.6)))
    fig_scatter.update_layout(title="Pd1 vs Pd2 Scatter Plot", xaxis_title="Pd1", yaxis_title="Pd2")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📊 Frequency Domain (FFT)")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, mode='lines', name='FFT Pd1'))
    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, mode='lines', name='FFT Pd2'))
    fig_fft.update_layout(title="Frequency Domain Signal", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)
