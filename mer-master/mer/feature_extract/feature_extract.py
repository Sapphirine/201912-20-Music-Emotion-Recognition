import numpy as np

from . import harmonic, spectral, temporal, rhythmic


def extract_feature_vector(signal, sr):
    """ Extracts all features from an audio signal and returns them as a numpy array"""
    harmonic_features = harmonic.extract_harmonic_features(signal, sr)
    spectral_features = spectral.extract_spectral_features(signal, sr)
    temporal_features = temporal.extract_temporal_features(signal, sr)
    rhythmic_features = rhythmic.extract_rhythmic_features(signal, sr)

    all_features = np.concatenate([harmonic_features, spectral_features, temporal_features, rhythmic_features])
    # all_features = np.concatenate([harmonic_features, temporal_features, rhythmic_features])

    return all_features


def reduce_feature_vector(feature_vector):
    """ Perform PCA dimensionality reduction on the feature vector

    Depending on implementation this may be done on individual classes of features and not here

    """
    pass
