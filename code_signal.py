import streamlit as st
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Sensor Signal Dashboard (Pd1 / Pd2)")

# === Your provided functions ===
def portion_selector(signal_label: str, start: int, end: int):
    assert signal_label in labels, f"{signal_label} not found in labels."
    portion_indices = list(range(start, end))
    return portion_indices, array_data[signal_label][start:end]

def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_fft(signal: np.ndarray, fs: float):
    N = len(signal)
    fft_values = rfft(signal)
    freqs = rfftfreq(N, d=1/fs)
    return freqs, np.abs(fft_values)

def merge_uploaded_csvs(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter=';')
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to read {file.name}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

# === File Upload ===
uploaded_files = st.file_uploader("📁 Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Files loaded and merged")

    # Convert and parse timeStamp
    df['time'] = df['timeStamp'].astype(str)
    if df['time'].str.contains(",").any():
        df[['ts_sec', 'ts_micro']] = df['time'].str.split(",", expand=True)
        df['ts_sec'] = pd.to_numeric(df['ts_sec'], errors='coerce')
        df['ts_micro'] = pd.to_numeric(df['ts_micro'], errors='coerce')
        df['time'] = df['ts_sec'] + df['ts_micro'] * 1e-6
        df.drop(columns=['ts_sec', 'ts_micro'], inplace=True)
    else:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')

    df = df[['time', 'rawPd1', 'rawPd2']].dropna()

    # Prepare arrays for processing
    labels = ['time', 'rawPd1', 'rawPd2']
    array_data = {label: df[label].to_numpy() for label in labels}
    fs = len(array_data['time']) / (array_data['time'][-1] - array_data['time'][0])

    # Range selection
    st.write("### 🔧 Select Portion for Analysis")
    start, end = st.slider("Choose index range", 0, len(df)-1, (0, len(df)-1), step=1)

    idxs_pd1, pd1_portion = portion_selector("rawPd1", start, end)
    idxs_pd2, pd2_portion = portion_selector("rawPd2", start, end)
    time_portion = array_data['time'][start:end]

    if len(pd1_portion) < 28 or len(pd2_portion) < 28:
        st.warning("⚠️ Select a larger portion (at least 28 points).")
        st.stop()

    # Bandpass Filter
    lowcut = 0.8
    highcut = 16.0
    filtered_pd1 = bandpass_filter(pd1_portion, lowcut, highcut, fs)
    filtered_pd2 = bandpass_filter(pd2_portion, lowcut, highcut, fs)

    # === PLOTS ===
    st.subheader("📉 Time Domain Signals")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time_portion, y=pd1_portion, name="Raw Pd1"))
    fig_time.add_trace(go.Scatter(x=time_portion, y=filtered_pd1, name="Filtered Pd1"))
    fig_time.add_trace(go.Scatter(x=time_portion, y=pd2_portion, name="Raw Pd2"))
    fig_time.add_trace(go.Scatter(x=time_portion, y=filtered_pd2, name="Filtered Pd2"))
    fig_time.update_layout(title="Time Domain", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🧪 Pd1 vs Pd2")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=pd1_portion, y=pd2_portion, mode='markers', name='Pd1 vs Pd2'))
    fig_scatter.update_layout(title="Pd1 vs Pd2 Scatter", xaxis_title="Pd1", yaxis_title="Pd2")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📊 FFT of Filtered Signals")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, name="FFT Pd1"))
    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, name="FFT Pd2"))
    fig_fft.update_layout(title="Frequency Domain", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)
