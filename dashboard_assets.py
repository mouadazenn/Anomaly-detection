# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:35:16 2025

@author: Admin
"""

import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal import fftconvolve
from math import floor
import matplotlib.pyplot as plt

# === Custom Filters ===
def bandpass_filter(signal, fs, lowcut=0.8, highcut=16, order=4):
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

def compute_sampling_frequency(time_array):
    fs = len(time_array) /(time_array[-1]-time_array[0])
    fs *= 1000 # ms to s conversion
    return fs