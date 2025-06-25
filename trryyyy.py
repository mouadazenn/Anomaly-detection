import streamlit as st
from dashboard_assets import *
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, fftconvolve
from math import floor
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧪 Signal Filter Dashboard (Pd1 / Pd2)")

# === Merge CSVs ===
def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter=';')
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to read {file.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

# === Upload UI ===
uploaded_files = st.file_uploader("📁 Upload CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded")

    # === Time parsing ===
    if df['time'].str.contains(" ").any():
        # Format like '2025-03-26 05:36:32.672300577'
        df['datetime'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
    else:
        # If not space-separated, fallback to numeric conversion
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')

    df = df.dropna(subset=['datetime'])
    df['relative_time'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

    # Keep only relevant columns
    df = df[['relative_time', 'rawPd1', 'rawPd2', 'ntc_1530']].dropna()

    # === Array conversion ===
    labels = ['relative_time', 'rawPd1', 'rawPd2', 'ntc_1530']
    array_data = {label: df[label].to_numpy(dtype=np.float64) for label in labels}

    # === Signal Range Selector ===
    st.write("### 🔧 Select Signal Range")
    start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)
    time = array_data['relative_time'][start:end]
    pd1 = array_data['rawPd1'][start:end]
    pd2 = array_data['rawPd2'][start:end]
    temp = array_data['ntc_1530'][start:end]
    fs = compute_sampling_frequency(array_data['relative_time'])

    # === Filter choice ===
    st.write("### 🎛️ Choose a Filter")
    filter_types = ['bandpass_filter', 'bandpass_filter_2', 'bandpass_filter_3', 'bandpass_filter_4', 'bandpass_filter_5']
    filter_type = st.selectbox("Select filter", filter_types)

    try:
        if filter_type == 'bandpass_filter':
            filtered_pd1 = bandpass_filter(pd1, fs=fs)
            filtered_pd2 = bandpass_filter(pd2, fs=fs)
        elif filter_type == 'bandpass_filter_2':
            filtered_pd1 = bandpass_filter_2(pd1, highWindow=30)
            filtered_pd2 = bandpass_filter_2(pd2, highWindow=30)
        elif filter_type == 'bandpass_filter_3':
            filtered_pd1 = bandpass_filter_3(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_3(pd2, fs=fs)
        elif filter_type == 'bandpass_filter_4':
            filtered_pd1 = bandpass_filter_4(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_4(pd2, fs=fs)
        elif filter_type == 'bandpass_filter_5':
            filtered_pd1 = bandpass_filter_5(pd1, fs=fs)
            filtered_pd2 = bandpass_filter_5(pd2, fs=fs)
    except Exception as e:
        st.error(f"⚠️ Filter error: {e}")
        filtered_pd1 = pd1
        filtered_pd2 = pd2

    # === Plots ===
    st.subheader("📉 Time Domain")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time, y=pd1, name="Raw Pd1", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd1, name="Filtered Pd1"))
    fig_time.add_trace(go.Scatter(x=time, y=pd2, name="Raw Pd2", line=dict(dash='dot')))
    fig_time.add_trace(go.Scatter(x=time, y=filtered_pd2, name="Filtered Pd2"))
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=pd1, y=pd2,
        mode='markers',
        name="Raw Pd1 vs Pd2",
        marker=dict(color=temp, colorscale='Viridis'),
        opacity=0.6
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📊 FFT")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, name="FFT Pd1"))
    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, name="FFT Pd2"))
    fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)


