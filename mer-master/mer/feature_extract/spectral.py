import numpy as np
import librosa
from scipy import stats


def get_sfm(x):
    return stats.mstats.gmean(x) / np.mean(x)


def get_scf(x):
    return np.max(x) / np.mean(x)


def extract_spectral_features(y, sr):
    per_sample_t = 1.0 / sr

    pitch_frame_length, pitch_hop_length = int(50 / 1000 / per_sample_t), int(25 / 1000 / per_sample_t)
    all_frame_fft = librosa.core.stft(y, n_fft=pitch_frame_length, hop_length=pitch_hop_length, \
                                      window='rect')
    all_frame_fft_energy = np.absolute(all_frame_fft) ** 2

    n_fft, num_frame = all_frame_fft_energy.shape[0], all_frame_fft_energy.shape[1]

    # zero-column delete
    k = 0
    while k < num_frame:
        if k >= all_frame_fft_energy.shape[1]:
            break
        if np.sum(all_frame_fft_energy[:, k]) == 0:
            all_frame_fft_energy = np.delete(all_frame_fft_energy, k, 1)
            k -= 1
        k += 1
    num_frame = all_frame_fft_energy.shape[1]

    all_frame_fft_sfm = np.array([[get_sfm(all_frame_fft_energy[n_fft // 17 * i:n_fft // 17 * (i + 1), j]) \
                                   for j in range(num_frame)] for i in range(8)], dtype=float)

    all_frame_fft_scf = np.array([[get_scf(all_frame_fft_energy[n_fft // 17 * i:n_fft // 17 * (i + 1), j]) \
                                   for j in range(num_frame)] for i in range(8)], dtype=float)

    song_sfm_mean = np.mean(all_frame_fft_sfm, axis=1)  # by row
    song_sfm_std = np.std(all_frame_fft_sfm, axis=1)  # by row

    song_scf_mean = np.mean(all_frame_fft_sfm, axis=1)  # by row
    song_scf_std = np.std(all_frame_fft_sfm, axis=1)  # by row

    mfcc = librosa.feature.mfcc(y, sr)
    song_mfcc = mfcc[:13, :]
    song_mfcc_mean = np.mean(song_mfcc, axis=1)
    song_mfcc_std = np.std(song_mfcc, axis=1)

    song_spectral_length = (len(song_mfcc_mean) + len(song_scf_mean) * 2) * 2

    return np.reshape(np.concatenate([song_sfm_mean, song_sfm_std, song_scf_mean, \
                                      song_scf_std, song_mfcc_mean, song_mfcc_std]), newshape=song_spectral_length)
