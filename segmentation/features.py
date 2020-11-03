import numpy as np

from scipy.signal import resample, welch
from scipy import signal
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

import librosa
import librosa.display
import matplotlib.pyplot as plt
from segmentation import *
import matplotlib.pyplot as plt

from statistics import median

ALENGTH = {
    'S1':   150,
    'Sys':  210,
    'S2':   120,
    'Dia':  510,
}

def _mean_std(x):
    return (np.mean(x), np.std(x))

def diff(t):
    return t[1] - t[0]

# Import data
states = custom_loadmat('a0001_states.mat')['assigned_states']
pcg = custom_loadmat('a0001_audio.mat')['audio']
transitions = get_transitions(states)

# Get boundaries
RR_boundary = boundaries(transitions, 'RR')
sys_boundary = boundaries(transitions, 'Sys')
dias_boundary = boundaries(transitions, 'Dia')
S1_boundary = boundaries(transitions, 'S1')
S2_boundary = boundaries(transitions, 'S2')

### Section 1: Time Domain Features
# Extract Interval Length Functions
interval_length_RR   = diff(RR_boundary)
interval_length_sys  = diff(sys_boundary)
interval_length_S1 = diff(S1_boundary)
interval_length_S2 = diff(S2_boundary)
interval_length_dias = diff(dias_boundary)

# Features 1-10
mean_RR, std_RR = _mean_std(interval_length_RR)
mean_sys, std_sys = _mean_std(interval_length_sys)
mean_S1, std_S1 = _mean_std(interval_length_S1)
mean_S2, std_S2 = _mean_std(interval_length_S2)
mean_dias, std_dias = _mean_std(interval_length_dias)

# Features 11-16
mean_ratio_sysRR, std_ratio_sysRR = _mean_std(interval_length_sys / interval_length_RR)
mean_ratio_diaRR,   std_ratio_diaRR  = _mean_std(interval_length_dias / interval_length_RR)
mean_ratio_sysDia,  std_ratio_sysDia = _mean_std(interval_length_sys / interval_length_dias)

# Determines the absolute amplitudes from each interval, and computes
# the mean of each of these as output.
def get_mean_abs(intervals):
    res = []
    for interval in intervals:
        interval_abs_amp = [abs(n) for n in interval]
        res.append(np.mean(interval_abs_amp))
    return np.array(res)

# Extract Amplitude Functions
# get intervals for each S1, Ds, S2, Ss
S1_intervals = get_intervals(pcg, transitions, 'S1', resize=ALENGTH['S1'])
S2_intervals = get_intervals(pcg, transitions, 'S2', resize=ALENGTH['S2'])
sys_intervals = get_intervals(pcg, transitions, 'Sys', resize=ALENGTH['Sys'])
dias_intervals = get_intervals(pcg, transitions, 'Dia', resize=ALENGTH['Dia'])

mean_abs_S1 = get_mean_abs(S1_intervals)
mean_abs_S2 = get_mean_abs(S2_intervals)
mean_abs_sys = get_mean_abs(sys_intervals)
mean_abs_dias = get_mean_abs(dias_intervals)

# Features 17-20 (Abs mean amplitude ratio)
ratio_S1_sys_mean, ratio_S1_sys_sd = _mean_std(mean_abs_sys / mean_abs_S1)
ratio_S2_dias_mean, ratio_S2_dias_mean = _mean_std(mean_abs_dias / mean_abs_S2)

# Features 21-28 (skewness)
skew_S1_mean, skew_S1_std  = _mean_std(np.array([skew(interval) for interval in S1_intervals]))
skew_S2_mean, skew_S2_std = _mean_std(np.array([skew(interval) for interval in S2_intervals]))
skew_sys_mean, skew_sys_std = _mean_std(np.array([skew(interval) for interval in sys_intervals]))
skew_dias_mean, skew_dias_std = _mean_std(np.array([skew(interval) for interval in dias_intervals]))

# Features 29-36 (kurtosis)
kurtosis_S1_mean, kurtosis_S1_std = _mean_std(np.array([kurtosis(interval) for interval in S1_intervals]))
kurtosis_S2_mean, kurtosis_S2_std = _mean_std(np.array([kurtosis(interval) for interval in S2_intervals]))
kurtosis_sys_mean, kurtosis_sys_std = _mean_std(np.array([kurtosis(interval) for interval in sys_intervals]))
kurtosis_dias_mean, kurtosis_dias_std = _mean_std(np.array([kurtosis(interval) for interval in dias_intervals]))

def plot_histogram(signal):
    hist,bin_edges = np.histogram(signal)
    plt.figure(figsize=[10,8])
    plt.bar(bin_edges[:-1], hist, width = 0.005, color='#0504aa',alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram',fontsize=15)
    plt.show()

# Analyze and visualize skewness for validation purposes
# print(np.array([skew(interval) for interval in S1_intervals]))
# plot_histogram(S1_intervals[0])

# Analyze and visualize kurtosis for validation purposes
# print(np.array([kurtosis(interval) for interval in S1_intervals]))
# plot_histogram(S1_intervals[0])
# plot_histogram(S1_intervals[2])


### Section 2: Frequency domain features
def calc_median_power_mean(intervals):
    band_medians = {
        "25_45": [],
        "45_65": [],
        "65_85": [],
        "85_105": [],
        "105_125": [],
        "125_150": [],
        "150_200": [],
        "200_300": [],
        "300_400": []
    }

    for interval in intervals:
        freqs, pows = welch(interval, fs=2000, window='hamming', nfft=2000)
        # [20, 30, 40...]
        # [pow at 20, pow at 30]
        for band in band_medians:
            band_start = int(band.split("_")[0])
            band_end = int(band.split("_")[1])
            band_indices = np.where((freqs < band_end) & (freqs > band_start))
            band_powers = pows[band_indices[0][0]:band_indices[0][-1]+1]
            median_band_power = median(band_powers)
            band_medians[band].append(median_band_power)

    results = {}
    for key in band_medians:
        results[key] = np.mean(band_medians[key])

    return results

median_power_means = []
for interval in [S1_intervals, S2_intervals, sys_intervals, dias_intervals]:
    median_power_means.append(calc_median_power_mean(interval))

# print(calc_median_power_mean(S1_intervals))
# print(calc_median_power_mean(S2_intervals))
# print(calc_median_power_mean(sys_intervals))
# print(calc_median_power_mean(dias_intervals))

# Extract MFCCs
def get_mfcc_means(intervals):
    mfccs = [[] for i in range(13)]
    for interval in intervals:
        mfcc = librosa.feature.mfcc(y=interval, n_mfcc=13, sr=2000)
        for i in range(13):
            mfccs[i].append(mfcc[i])
    results = [np.mean(mfcc) for mfcc in mfccs]
    return results

mfcc_means = []
for cycle in [S1_intervals, S2_intervals, sys_intervals, dias_intervals]:
    mfcc_means.append(get_mfcc_means(cycle))