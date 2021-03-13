import numpy as np

from scipy.signal import welch
from scipy import signal
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

import librosa
import librosa.display
import matplotlib.pyplot as plt
from classification.segmentation_util import *

from statistics import median

class FeaturesProcessor:
    def __init__(self, filename, ALENGTH = {"S1": 150, "Sys": 210, "S2": 120, "Dia":  510}):
        self.filename = filename
        self.ALENGTH = ALENGTH

    def load_data(self):
        states = custom_loadmat(f"{self.filename}_states.mat")['assigned_states']
        pcg = custom_loadmat(f"{self.filename}_audio.mat")['audio']
        transitions = get_transitions(states)
        return pcg, transitions

    def _mean_std(self, x):
        return (np.mean(x), np.std(x))

    def diff(self, t):
        return t[1] - t[0]

    # Determines the absolute amplitudes from each interval, and computes
    # the mean of each of these as output.
    def get_mean_abs(self, intervals):
        res = []
        for interval in intervals:
            interval_abs_amp = [abs(n) for n in interval]
            res.append(np.mean(interval_abs_amp))
        return np.array(res)

    def get_time_domain_features(self):
        pcg, transitions = self.load_data()

        # Get boundaries
        RR_boundary = boundaries(transitions, 'RR')
        sys_boundary = boundaries(transitions, 'Sys')
        dias_boundary = boundaries(transitions, 'Dia')
        S1_boundary = boundaries(transitions, 'S1')
        S2_boundary = boundaries(transitions, 'S2')

        # Extract Interval Length Functions
        interval_length_RR   = self.diff(RR_boundary)
        interval_length_sys  = self.diff(sys_boundary)
        interval_length_S1 = self.diff(S1_boundary)
        interval_length_S2 = self.diff(S2_boundary)
        interval_length_dias = self.diff(dias_boundary)

        # Features 1-10
        mean_RR, std_RR = self._mean_std(interval_length_RR)
        mean_sys, std_sys = self._mean_std(interval_length_sys)
        mean_S1, std_S1 = self._mean_std(interval_length_S1)
        mean_S2, std_S2 = self._mean_std(interval_length_S2)
        mean_dias, std_dias = self._mean_std(interval_length_dias)

        # Features 11-16
        mean_ratio_sysRR, std_ratio_sysRR = self._mean_std(interval_length_sys / interval_length_RR)
        mean_ratio_diaRR, std_ratio_diaRR  = self._mean_std(interval_length_dias / interval_length_RR)
        mean_ratio_sysDia, std_ratio_sysDia = self._mean_std(interval_length_sys / interval_length_dias)

        # Extract Amplitude Functions
        # get intervals for each S1, Ds, S2, Ss
        S1_intervals = get_intervals(pcg, transitions, 'S1', resize=self.ALENGTH['S1'])
        S2_intervals = get_intervals(pcg, transitions, 'S2', resize=self.ALENGTH['S2'])
        sys_intervals = get_intervals(pcg, transitions, 'Sys', resize=self.ALENGTH['Sys'])
        dias_intervals = get_intervals(pcg, transitions, 'Dia', resize=self.ALENGTH['Dia'])

        mean_abs_S1 = self.get_mean_abs(S1_intervals)
        mean_abs_S2 = self.get_mean_abs(S2_intervals)
        mean_abs_sys = self.get_mean_abs(sys_intervals)
        mean_abs_dias = self.get_mean_abs(dias_intervals)

        # Edge case: if any of the arrays contain 0, replace with mean of array
        # to prevent divide by 0 error.
        mean_abs_S1[mean_abs_S1 == 0] = np.mean(mean_abs_S1)
        mean_abs_S2[mean_abs_S2 == 0] = np.mean(mean_abs_S2)
        mean_abs_sys[mean_abs_sys == 0] = np.mean(mean_abs_sys)
        mean_abs_dias[mean_abs_dias == 0] = np.mean(mean_abs_dias)

        mean_ratio_sysS1 = self._mean_std(mean_abs_sys / mean_abs_S1)[0]
        std_ratio_sysS1 = self._mean_std(mean_abs_sys / mean_abs_S1)[1]
        mean_ratio_diasS2 = self._mean_std(mean_abs_dias / mean_abs_S2)[0]
        std_ratio_diasS2 = self._mean_std(mean_abs_dias / mean_abs_S2)[1]

        # Features 21-28 (skewness)
        skew_S1_mean, skew_S1_std  = self._mean_std(np.array([skew(interval) for interval in S1_intervals]))
        skew_S2_mean, skew_S2_std = self._mean_std(np.array([skew(interval) for interval in S2_intervals]))
        skew_sys_mean, skew_sys_std = self._mean_std(np.array([skew(interval) for interval in sys_intervals]))
        skew_dias_mean, skew_dias_std = self._mean_std(np.array([skew(interval) for interval in dias_intervals]))

        # Features 29-36 (kurtosis)
        kurtosis_S1_mean, kurtosis_S1_std = self._mean_std(np.array([kurtosis(interval) for interval in S1_intervals]))
        kurtosis_S2_mean, kurtosis_S2_std = self._mean_std(np.array([kurtosis(interval) for interval in S2_intervals]))
        kurtosis_sys_mean, kurtosis_sys_std = self._mean_std(np.array([kurtosis(interval) for interval in sys_intervals]))
        kurtosis_dias_mean, kurtosis_dias_std = self._mean_std(np.array([kurtosis(interval) for interval in dias_intervals]))

        return [
            mean_RR, 
            std_RR,
            mean_sys, 
            std_sys,
            mean_S1, 
            std_S1,
            mean_S2, 
            std_S2,
            mean_dias,
            std_dias,
            mean_ratio_sysRR, 
            std_ratio_sysRR,
            mean_ratio_diaRR, 
            std_ratio_diaRR,
            mean_ratio_sysDia, 
            std_ratio_sysDia,
            mean_ratio_sysS1,
            std_ratio_sysS1,
            mean_ratio_diasS2,
            std_ratio_diasS2,
            skew_S1_mean,
            skew_S1_std,
            skew_S2_mean, 
            skew_S2_std,
            skew_sys_mean, 
            skew_sys_std,
            skew_dias_mean, 
            skew_dias_std,
            kurtosis_S1_mean, 
            kurtosis_S1_std,
            kurtosis_S2_mean, 
            kurtosis_S2_std,
            kurtosis_sys_mean, 
            kurtosis_sys_std,
            kurtosis_dias_mean, 
            kurtosis_dias_std
        ]

    def calc_median_power_mean(self, intervals):
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
        results_arr = []
        for key in band_medians:
            results[key] = np.mean(band_medians[key])
            results_arr.append(np.mean(band_medians[key]))

        return results, results_arr

    def get_mean_mfccs(self, intervals, num_mfccs=13):
        mfccs = [[] for i in range(num_mfccs)]
        for interval in intervals:
            mfcc = librosa.feature.mfcc(y=interval, n_mfcc=num_mfccs, sr=2000)
            for i in range(num_mfccs):
                mfccs[i].append(mfcc[i])
        results = [np.mean(mfcc) for mfcc in mfccs]
        return results

    def get_frequency_domain_features(self):
        pcg, transitions = self.load_data()

        S1_intervals = get_intervals(pcg, transitions, 'S1', resize=self.ALENGTH['S1'])
        S2_intervals = get_intervals(pcg, transitions, 'S2', resize=self.ALENGTH['S2'])
        sys_intervals = get_intervals(pcg, transitions, 'Sys', resize=self.ALENGTH['Sys'])
        dias_intervals = get_intervals(pcg, transitions, 'Dia', resize=self.ALENGTH['Dia'])

        # Features 0 - 36
        median_power_mean_S1 = self.calc_median_power_mean(S1_intervals)[1]
        median_power_mean_S2 = self.calc_median_power_mean(S2_intervals)[1]
        median_power_mean_sys = self.calc_median_power_mean(sys_intervals)[1]
        median_power_mean_dias = self.calc_median_power_mean(dias_intervals)[1]

        # Features 37 - 88 (52 MFCC features)
        mfcc_S1 = self.get_mean_mfccs(S1_intervals)
        mfcc_S2 = self.get_mean_mfccs(S2_intervals)
        mfcc_sys = self.get_mean_mfccs(sys_intervals)
        mfcc_dias = self.get_mean_mfccs(dias_intervals)

        features = [
            median_power_mean_S1, 
            median_power_mean_S2, 
            median_power_mean_sys, 
            median_power_mean_dias,
            mfcc_S1,
            mfcc_S2,
            mfcc_sys,
            mfcc_dias
        ]

        flat_features_list = [item for sublist in features for item in sublist]
        return flat_features_list
    
    def get_all_features(self):
        time_domain_features = self.get_time_domain_features()
        freqeuncy_domain_features = self.get_frequency_domain_features()
        
        if time_domain_features is None or freqeuncy_domain_features is None:
            return None
        
        features = time_domain_features + freqeuncy_domain_features
        return features