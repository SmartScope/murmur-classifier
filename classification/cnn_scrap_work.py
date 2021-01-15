import scipy.signal
from segmentation_util import custom_loadmat, get_transitions
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from cnn import CNNParameters

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_data():
    file_names = ["./challenge_data/b" + str(i).zfill(4) for i in range(1, 300)]

    X = []
    for filename in file_names:
        pcg = custom_loadmat(f"{filename}_audio.mat")['audio']
        # bands = frequency_decomposition(pcg)
        mfcc = librosa.feature.mfcc(y=pcg, sr=2000, n_mfcc=13)
        # X.append(bands)
        X.append(mfcc)
    
    # get labels
    abnormal_records = set()
    with open("./challenge_data/RECORDS-abnormal") as fp:
        for line in fp:
            l = line.rstrip("\n")
            abnormal_records.add(l)

    # 1 means abnormal, 0 means normal
    y = [1 if remove_prefix(fname, "./challenge_data/") in abnormal_records else 0 for fname in file_names]

    return X, y

def frequency_decomposition(pcg, N=60, sr=1000):
    Wn = (45 * 2) / sr
    b1 = scipy.signal.firwin(N + 1, Wn, window='hamming', pass_zero='lowpass')

    Wn = [(45 * 2) / sr, (80 * 2) / sr]
    b2 = scipy.signal.firwin(N + 1, Wn, window='hamming')

    Wn = [(80 * 2) / sr, (200 * 2) / sr]
    b3 = scipy.signal.firwin(N + 1, Wn, window='hamming')

    Wn = (200 * 2) / sr
    b4 = scipy.signal.firwin(N + 1, Wn, window='hamming', pass_zero='highpass')

    return [
        scipy.signal.filtfilt(b1, 1, pcg),
        scipy.signal.filtfilt(b2, 1, pcg),
        scipy.signal.filtfilt(b3, 1, pcg),
        scipy.signal.filtfilt(b4, 1, pcg)
    ]

signal, sr = librosa.load("./challenge_data/b0001.wav")
# pcg_resampled = librosa.resample(pcg, sr, 1000)

# Todo: apply low and high pass filters and spike removal

states = custom_loadmat("./challenge_data/b0001_states.mat")['assigned_states']
transitions = get_transitions(states)

# Filter signal into 4 different frequency bands
pcg = frequency_decomposition(signal)

num_freq_bands = 4
num_samples = 2500
num_cardiac_cycles = len(transitions) - 1

# X = np.zeros(shape=(num_cardiac_cycles, num_samples, num_freq_bands))
X = np.zeros(shape=(num_freq_bands, num_cardiac_cycles, num_samples))
print(X.shape)

# a1 = np.array([
#     [1, 2, 3, 4],
#     [1, 2, 3, 4]
# ])

# a2 = np.array([
#     [1, 2, 3, 4],
#     [1, 2, 3, 4]
# ])

# a3 = np.array([a1, a2])
# print(a3.shape)

for cycle in range(num_cardiac_cycles):
    for freq_band in range(num_freq_bands):
        temp = pcg[freq_band][transitions[cycle]:transitions[cycle + 1]]
        for i in range(len(temp)):
            X[freq_band][cycle][i] = temp[i]

for cycle in num_cardiac_cycles:
    cycles_from_bands = []
    for band in range(num_freq_bands):
        cycles_from_bands.append(X[band][cycle])

