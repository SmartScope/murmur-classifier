import json
import os
import math
import librosa
import scipy.signal as signal
from segmentation_util import custom_loadmat

def frequency_decomposition(pcg, N=60, sr=1000):
    Wn = (45 * 2) / sr
    b1 = signal.firwin(N + 1, Wn, window='hamming', pass_zero='lowpass')

    Wn = [(45 * 2) / sr, (80 * 2) / sr]
    b2 = signal.firwin(N + 1, Wn, window='hamming')

    Wn = [(80 * 2) / sr, (200 * 2) / sr]
    b3 = signal.firwin(N + 1, Wn, window='hamming')

    Wn = (200 * 2) / sr
    b4 = signal.firwin(N + 1, Wn, window='hamming', pass_zero='highpass')

    return b1, b2, b3, b4

    return [
        scipy.signal.filtfilt(b1, 1, pcg),
        scipy.signal.filtfilt(b2, 1, pcg),
        scipy.signal.filtfilt(b3, 1, pcg),
        scipy.signal.filtfilt(b4, 1, pcg)
    ]

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_data(json_path="/Users/manthanshah/Desktop/data.json"):
    # Dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": ["normal", "abnormal"],
        "labels": [],
        "mfcc": []
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

        # todo
        # b1, b2, b3, b4 = frequency_decomposition(pcg, sr=sr)

        mfcc = librosa.feature.mfcc(y=pcg, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        
        if len(mfcc) == 32:
            data["mfcc"].append(mfcc.tolist())

            if remove_prefix(filename, "./challenge_data/") in abnormal_records:
                data["labels"].append(1)
            else:
                data["labels"].append(0)
    
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

get_data()