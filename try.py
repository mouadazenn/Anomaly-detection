import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt

# === STREAMLIT UI ===
st.title("📈 Interactive Sensor Signal Analysis")

uploaded_files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

# === Function: Merge uploaded CSVs ===
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
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# === Signal Processing Functions ===
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    if fs <= 0 or lowcut >= highcut:
        raise ValueError("Invalid filter settings: ensure fs > 0 and lowcut < highcut.")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Normalized cutoff frequencies must be between 0 and 1. Got low={low}, high={high}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_fft(signal, fs):
    N = len(signal)
    fft_values = fft(signal)
    freqs = fftfreq(N, d=1/fs)
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

    time_data = df[time_col].astype(str)

    # === Fix time parsing for comma-separated Unix + microsecond format ===
    if time_data.str.contains(",").any():
        try:
            df[['ts_sec', 'ts_micro']] = time_data.str.split(",", expand=True)
            df['ts_sec'] = pd.to_numeric(df['ts_sec'], errors='coerce')
            df['ts_micro'] = pd.to_numeric(df['ts_micro'], errors='coerce')
            df['__time__'] = df['ts_sec'] + df['ts_micro'] * 1e-6
        except Exception as e:
            st.error(f"❌ Failed to parse time: {e}")
            st.stop()
    else:
        df['__time__'] = pd.to_numeric(time_data, errors='coerce')

    t = df['__time__'].dropna()
    if t.empty:
        st.error("❌ Could not parse time column correctly. Please check formatting.")
        st.stop()

    fs = len(t) / (t.iloc[-1] - t.iloc[0]) if len(t) > 1 else 0
    if fs <= 0:
        st.error("⚠️ Sampling frequency is invalid. Check your time column.")
        st.stop()

    start, end = st.slider("Select time range (in seconds)", float(t.min()), float(t.max()), (float(t.min()), float(t.max())))
    mask = (df['__time__'] >= start) & (df['__time__'] <= end)

    pd1 = pd.to_numeric(df.loc[mask, pd1_col], errors='coerce').dropna()
    pd2 = pd.to_numeric(df.loc[mask, pd2_col], errors='coerce').dropna()
    t_selected = df.loc[mask, '__time__'].iloc[:min(len(pd1), len(pd2))]
    pd1 = pd1.iloc[:len(t_selected)]
    pd2 = pd2.iloc[:len(t_selected)]

    # Apply filter
    lowcut = st.number_input("Low cutoff frequency (Hz)", value=0.8)
    highcut = st.number_input("High cutoff frequency (Hz)", value=16.0)

    try:
        filtered_pd1 = bandpass_filter(pd1, lowcut, highcut, fs)
        filtered_pd2 = bandpass_filter(pd2, lowcut, highcut, fs)
    except ValueError as e:
        st.error(f"⚠️ {e}")
        st.stop()

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
