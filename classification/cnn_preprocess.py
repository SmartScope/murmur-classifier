import json
import os
import math
import librosa
import scipy.signal as signal
from segmentation_util import custom_loadmat, get_transitions
import numpy as np

def frequency_decomposition(pcg, N=60, sr=2000):
    """
    Decomposes a PCG signal into 4 distinct frequency bands of:
        1. 0 - 45 Hz
        2. 45 - 80 Hz
        3. 80 - 200 Hz
        4. 200 - 400 Hz
    
    Args:
        pcg (array): input signal to be decomposed
        N (int): length of the filter (filter order + 1)
        sr (int): sampling rate of the input signal
    Returns:
        pcg_decomposed (ndarray): an array of arrays corresponding to each frequency band
    """

    Wn = (45 * 2) / sr
    b1 = signal.firwin(N + 1, Wn, window='hamming', pass_zero='lowpass')

    Wn = [(45 * 2) / sr, (80 * 2) / sr]
    b2 = signal.firwin(N + 1, Wn, window='hamming')

    Wn = [(80 * 2) / sr, (200 * 2) / sr]
    b3 = signal.firwin(N + 1, Wn, window='hamming')

    Wn = (200 * 2) / sr
    b4 = signal.firwin(N + 1, Wn, window='hamming', pass_zero='highpass')

    return [
        signal.filtfilt(b1, 1, pcg),
        signal.filtfilt(b2, 1, pcg),
        signal.filtfilt(b3, 1, pcg),
        signal.filtfilt(b4, 1, pcg)
    ]

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_data(json_path="/Users/manthanshah/Desktop/data.json"):
    """
    Converts audio data into JSON data file for input to CNN.

    Args:
        json_path: the location where to store the output data
    """

    # Dictionary to store mapping, labels, and values
    data = {
        "mapping": ["normal", "abnormal"],
        "labels": [],
        "values": []
    }

    # Get labels
    abnormal_records = set()
    with open("./challenge_data/RECORDS-abnormal") as fp:
        for line in fp:
            l = line.rstrip("\n")
            abnormal_records.add(l)
    
    filenames = ["./challenge_data/b" + str(i).zfill(4) for i in range(1, 491)]
    for filename in filenames:
        pcg, sr = librosa.load(f"{filename}.wav", sr=2000)

        pcg_decomposed = frequency_decomposition(pcg, sr=sr)
        pcg_states = custom_loadmat(f"{filename}_states.mat")['assigned_states']

        transitions = get_transitions(pcg_states)

        num_freq_bands = 4
        num_samples = 2500 # 2.5 s
        num_cardiac_cycles = min(32, len(transitions) - 1) # cap at 32 cycles per signal

        if num_cardiac_cycles < 32:
            continue

        # Data shape: 32 cycles x 1500 samples / cycle x 4 bands
        X = np.zeros(shape=(num_cardiac_cycles, num_samples, num_freq_bands))

        for cardiac_cycle in range(num_cardiac_cycles):
            for freq_band in range(num_freq_bands):
                values = pcg_decomposed[freq_band][transitions[cardiac_cycle]:transitions[cardiac_cycle + 1]]

                if len(values) > 2500:
                    values = values[:len(values)-2500]

                for sample in range(len(X[cardiac_cycle])):
                    if sample < len(values):
                        X[cardiac_cycle][sample][freq_band] = values[sample]

        data["values"].append(X.tolist())
        if remove_prefix(filename, "./challenge_data/") in abnormal_records:
            data["labels"].append(1)
        else:
            data["labels"].append(0)
    
    # save values to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Run this to pre-process any data for input to the CNN
get_data()