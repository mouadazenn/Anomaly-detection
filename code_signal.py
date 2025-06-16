import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt


st.title("📈 Pd1 / Pd2 Signal Analysis Dashboard")

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
    df = merge_uploaded_csvs([uploaded_files])
    expected_cols = ['timeStamp', 'ntc_1530', 'rawPd1', 'rawPd2']

    if not all(col in df.columns for col in expected_cols):
        st.error(f"Missing required columns. Expected: {expected_cols}")
        st.stop()

    # Convert time and numeric columns
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    for col in ['temp', 'pd1', 'pd2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    fs = len(df) / (df['time'].iloc[-1] - df['time'].iloc[0])
    st.success(f"✅ Data loaded | Sampling frequency estimated: {fs:.2f} Hz")

    st.write("### 🔍 Signal Preview")
    st.dataframe(df.head())

    # Range selection
    start_idx, end_idx = st.slider("Select index range for analysis",
                                   0, len(df) - 1, (0, len(df) - 1), step=1)

    df_portion = df.iloc[start_idx:end_idx]
    time_portion = df_portion['time'].to_numpy()
    pd1_portion = df_portion['pd1'].to_numpy()
    pd2_portion = df_portion['pd2'].to_numpy()

    if len(pd1_portion) < 28 or len(pd2_portion) < 28:
        st.warning("⚠️ Please select a larger portion (at least 28 points required).")
        st.stop()

    # Filter inputs
    st.write("### ⚙️ Filter Settings")
    col1, col2 = st.columns(2)
    with col1:
        lowcut = st.number_input("Low cutoff frequency (Hz)", value=0.8)
    with col2:
        highcut = st.number_input("High cutoff frequency (Hz)", value=16.0)

    try:
        filtered_pd1 = bandpass_filter(pd1_portion, lowcut, highcut, fs)
        filtered_pd2 = bandpass_filter(pd2_portion, lowcut, highcut, fs)
    except ValueError as e:
        st.error(f"⚠️ Filter error: {e}")
        st.stop()

    # === PLOTS ===
    st.write("### 📉 Time Domain Signal")
    fig, ax = plt.subplots()
    ax.plot(time_portion, pd1_portion, label='Raw Pd1')
    ax.plot(time_portion, filtered_pd1, label='Filtered Pd1')
    ax.plot(time_portion, pd2_portion, label='Raw Pd2')
    ax.plot(time_portion, filtered_pd2, label='Filtered Pd2')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.write("### 🧪 Pd1 vs Pd2")
    fig2, ax2 = plt.subplots()
    ax2.scatter(pd1_portion, pd2_portion, alpha=0.6)
    ax2.set_xlabel("Pd1")
    ax2.set_ylabel("Pd2")
    ax2.set_title("Pd1 vs Pd2 Scatter")
    ax2.grid()
    st.pyplot(fig2)

    st.write("### 📊 Frequency Domain (FFT)")
    f1, fft1 = compute_fft(filtered_pd1, fs)
    f2, fft2 = compute_fft(filtered_pd2, fs)
    fig3, ax3 = plt.subplots()
    ax3.plot(f1, fft1, label='FFT Pd1')
    ax3.plot(f2, fft2, label='FFT Pd2')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid()
    st.pyplot(fig3)
