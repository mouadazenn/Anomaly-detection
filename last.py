import streamlit as st
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal import fftconvolve
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

# === Custom Filters ===
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: {low}, {high}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def bandpass_filter_2(signal, highWindow):
    signal = np.copy(signal)
    signal -= np.mean(signal)
    for i in range(0, len(signal), highWindow):
        signal[i:i+highWindow] -= np.mean(signal[i:i+highWindow])
    return signal

def bandpass_filter_3(signal, fs, highCut=16):
    order = floor(fs / highCut)
    signal = np.copy(signal)
    signal -= np.mean(signal)
    for i in range(0, len(signal), order):
        signal[i:i+order] -= np.mean(signal[i:i+order])
    return signal

def bandpass_filter_4(signal, fs, highCut=16):
    order = floor(fs / highCut)
    signal = np.copy(signal)
    delta = np.floor(order / 2).astype(int)
    signal = np.concatenate([signal[delta-1::-1], signal, signal[:-delta-1:-1]])
    signal -= np.mean(signal)
    result = np.zeros(len(signal) - 2 * delta)
    for i in range(len(result)):
        result[i] = np.mean(signal[i:i+2*delta])
    return result

def bandpass_filter_5(signal, fs, highCut=16):
    order = floor(fs / highCut)
    signal = np.copy(signal)
    signal -= np.mean(signal)
    avg_filter = np.hamming(order)
    return fftconvolve(signal, avg_filter, mode='same')

def compute_fft(signal, fs):
    N = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(N, d=1/fs)
    return freqs, np.abs(fft_vals)

# === Upload UI ===
uploaded_files = st.file_uploader("📁 Upload CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = merge_uploaded_csvs(uploaded_files)
    st.success("✅ Data loaded")
 




    # === Time parsing ===
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
    labels = df.columns.tolist()
    array_data = {label: df[label].to_numpy() for label in labels}
    fs = len(df) / (df['time'].iloc[-1] - df['time'].iloc[0])

    # === Signal Range Selector ===
    st.write("### 🔧 Select Signal Range")
    start, end = st.slider("Select sample indices", 0, len(df)-1, (0, len(df)-1), step=1)
    time = array_data['time'][start:end]
    pd1 = array_data['rawPd1'][start:end]
    pd2 = array_data['rawPd2'][start:end]

    # === Filter choice ===
    st.write("### 🎛️ Choose a Filter")
    filter_type = st.selectbox("Select filter type", ['bandpass_filter_2', 'bandpass_filter_3', 'bandpass_filter_4', 'bandpass_filter_5'])

    try:
        if filter_type == 'bandpass_filter_2':
            filtered_pd1 = bandpass_filter_2(pd1, highWindow=50)
            filtered_pd2 = bandpass_filter_2(pd2, highWindow=50)
        elif filter_type == 'bandpass_filter_3':
            filtered_pd1 = bandpass_filter_3(pd1, fs)
            filtered_pd2 = bandpass_filter_3(pd2, fs)
        elif filter_type == 'bandpass_filter_4':
            filtered_pd1 = bandpass_filter_4(pd1, fs)
            filtered_pd2 = bandpass_filter_4(pd2, fs)
        elif filter_type == 'bandpass_filter_5':
            filtered_pd1 = bandpass_filter_5(pd1, fs)
            filtered_pd2 = bandpass_filter_5(pd2, fs)
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
    fig_scatter.add_trace(go.Scatter(x=pd1, y=pd2, mode='markers', name="Raw Pd1 vs Pd2", opacity=0.6))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📊 FFT")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=f1, y=fft1, name="FFT Pd1"))
    fig_fft.add_trace(go.Scatter(x=f2, y=fft2, name="FFT Pd2"))
    fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft, use_container_width=True)


