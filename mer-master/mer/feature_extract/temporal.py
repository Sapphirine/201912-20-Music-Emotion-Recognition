import warnings

import numpy as np
import librosa

def get_envelopes(signal, sr):
    onset_samples = np.unique(librosa.onset.onset_detect(signal, sr=sr, backtrack=True, units='samples'))
    all_envelopes = np.split(signal, onset_samples)

    return all_envelopes


def temporal_centroid(envelope, sr):
    """computes the temporal centroid of an onset envelope"""
    D = np.abs(librosa.stft(envelope))
    times = librosa.times_like(D)

    onset_strength = librosa.onset.onset_strength(y=envelope, sr=sr)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            temporal_centroid = sum(onset_strength * times) / sum(onset_strength)
    except RuntimeWarning:
        temporal_centroid = np.nan

    return temporal_centroid


def log_attack_time(envelope, sr, thresh_percent):
    """ Computes the log of the attack phase of the envelope of a signal"""
    D = np.abs(librosa.stft(envelope))
    times = librosa.times_like(D)
    onset_strength = librosa.onset.onset_strength(y=envelope, sr=sr)

    stop_attack_index = np.argmax(onset_strength)
    stop_attack_value = envelope[stop_attack_index]
    thresh = stop_attack_value * thresh_percent / 100

    try:
        start_attack_index = [x > thresh for x in onset_strength].index(True)
    except ValueError:
        return np.nan

    if start_attack_index == stop_attack_index:
        start_attack_index -= 1

    log_attack_time = np.log10(times[stop_attack_index] - times[start_attack_index])

    return log_attack_time


def extract_temporal_features(signal, sr):
    """ Extract all temporal features for a given signal and return as an array"""
    all_envelopes = get_envelopes(signal, sr)

    zero_crossings = np.array([sum(librosa.zero_crossings(x, pad=False)) for x in all_envelopes])
    zero_features = np.array([np.mean(zero_crossings), np.std(zero_crossings)])

    temporal_centroids = np.array([temporal_centroid(x, sr) for x in all_envelopes])
    temporal_centroids = temporal_centroids[~np.isnan(temporal_centroids)]
    temporal_cen_features = np.array([np.mean(temporal_centroids), np.std(temporal_centroids)])

    log_attacks = np.array([log_attack_time(x, sr, 50) for x in all_envelopes])
    log_attacks = log_attacks[~np.isnan(log_attacks)]
    log_attack_features = np.array([np.mean(log_attacks), np.std(log_attacks)])

    all_features = np.concatenate([zero_features, temporal_cen_features, log_attack_features])

    return all_features
