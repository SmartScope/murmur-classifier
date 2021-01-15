import json
import os
import math
import librosa
from segmentation_util import custom_loadmat

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
    
    filenames = ["./challenge_data/b" + str(i).zfill(4) for i in range(1, 300)]
    for filename in filenames:
        pcg, sr = librosa.load(f"{filename}.wav", sr=2000)
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