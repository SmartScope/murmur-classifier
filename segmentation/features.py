import numpy as np

from scipy.signal import resample
from scipy.io import loadmat

import librosa
import librosa.display
import matplotlib.pyplot as plt
from segmentation import *

def _mean_std(x):
    return (np.mean(x), np.std(x))

def diff(t):
    return t[1] - t[0]

# Import data
states = custom_loadmat('a0001_states.mat')['assigned_states']
pcg = custom_loadmat('a0001_audio.mat')['audio']
transitions = get_transitions(states)
intervals = get_intervals(pcg, transitions)

# Extract Interval Length Functions
interval_length_RR   = diff(boundaries(transitions, 'RR'))
interval_length_sys  = diff(boundaries(transitions, 'Sys'))
interval_length_S1 = diff(boundaries(transitions, 'S1'))
interval_length_S2 = diff(boundaries(transitions, 'S2'))
interval_length_dias = diff(boundaries(transitions, 'Dia'))

mean_RR, std_RR = _mean_std(interval_length_RR)
mean_sys, std_sys = _mean_std(interval_length_sys)
mean_S1, std_S1 = _mean_std(interval_length_S1)
mean_S2, std_S2 = _mean_std(interval_length_S2)
mean_dias, std_dias = _mean_std(interval_length_dias)

mean_ratio_sysRR, std_ratio_sysRR = _mean_std(interval_length_sys / interval_length_RR)
mean_ratio_diaRR,   std_ratio_diaRR  = _mean_std(interval_length_dias / interval_length_RR)
mean_ratio_sysDia,  std_ratio_sysDia = _mean_std(interval_length_sys / interval_length_dias)

# Extract Amplitude Functions




# Extract MFCCs
signal = np.concatenate(intervals)
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=1000)
# print(mfccs.shape)
# librosa.display.specshow(mfccs, x_axis="time", sr=1000)
# plt.colorbar(format="%+2.f")
# plt.show()

mean_mel = np.mean(mfccs, axis=0)
std_mel  = np.std(mfccs, axis=0)
# print(mean_mel, std_mel)

# Mean and std dev of each of the segments
# Ratio of systolic to RR intervals