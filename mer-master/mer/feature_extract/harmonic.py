import numpy as np
import librosa


def append_val(d, k, v):
    if k not in d:
        d[k] = [v]
    else:
        d[k].append(v)
    return d


def salient_pitch(y, sr):
    per_sample_t = 1.0 / sr
    pitch_frame_length, pitch_hop_length = int(50 / 1000 / per_sample_t), int(25 / 1000 / per_sample_t)

    # we hope to introduce an acf - filtered salient pitch
    salient_pitch = []
    num_total_frames = (len(y) - pitch_frame_length) // pitch_hop_length + 1
    for i in range(num_total_frames):
        acf = librosa.core.autocorrelate(y[i * pitch_hop_length: (i * pitch_hop_length + pitch_frame_length)])
        # introduce a filtered acf method
        for j in range(1, len(acf) - 1):
            if acf[j] > acf[j - 1] and acf[j] > acf[j + 1]:
                if j >= 20 and acf[j] == np.max(acf[j - 20:j + 20]):  # reducing the adjacent HF fluctuation
                    salient_pitch.append(sr / j)
                    break
    return np.array(salient_pitch, dtype=float)


def chroma_centroid(y, sr):
    per_sample_t = 1.0 / sr
    chromagram_frame_length, chromagram_hop_length = int(100 / 1000 / per_sample_t), int(12.5 / 1000 / per_sample_t)
    chromagram_center = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=chromagram_frame_length, \
                                                          hop_length=chromagram_hop_length)[0]

    return chromagram_center[1:]


def key_clarity_mode(y, sr):  # key clarity -> max intensity mean , mode : max major - max minor
    major_index, minor_index = np.array([0, 2, 4, 5, 7, 9, 11], dtype=int), np.array([1, 3, 6, 8, 10], dtype=int)
    major_minor_key_clarity = np.zeros(shape=2, dtype=int)

    per_sample_t = 1.0 / sr
    chromagram_frame_length, chromagram_hop_length = int(100 / 1000 / per_sample_t), int(12.5 / 1000 / per_sample_t)
    chromagram = librosa.feature.chroma_stft(y, sr=sr, n_fft=chromagram_frame_length, \
                                             hop_length=chromagram_hop_length, \
                                             win_length=chromagram_frame_length, window='hamming', \
                                             n_chroma=12)
    mode = []
    # zero-column delete
    k, ln_chro = 0, chromagram.shape[1]
    while k < ln_chro:
        if k >= chromagram.shape[1]:
            break
        if np.sum(chromagram[:, k]) == 0:
            chromagram = np.delete(chromagram, k, 1)
            k -= 1
        k += 1
    major_chro = np.array([chromagram[0, :], chromagram[2, :], chromagram[4, :], chromagram[5, :], chromagram[7, :], \
                           chromagram[9, :], chromagram[11, :]], dtype=float)
    minor_chro = np.array([chromagram[1, :], chromagram[3, :], chromagram[6, :], chromagram[8, :], chromagram[10, :]], \
                          dtype=float)
    key_clarity = []

    key_index = np.argmax([np.mean(chromagram[k, :]) for k in range(12)])
    key_clarity.append(np.mean(chromagram[key_index, :]))
    key_clarity.append(np.std(chromagram[key_index, :]))

    major_minor_key_clarity[0] = major_index[np.argmax([np.sum(major_chro[k, :]) for k in range(7)])]
    major_minor_key_clarity[1] = minor_index[np.argmax([np.sum(minor_chro[k, :]) for k in range(5)])]
    dif_sum = []
    for j in range(chromagram.shape[1]):
        dif_sum.append(chromagram[major_minor_key_clarity[0], j] - \
                       chromagram[major_minor_key_clarity[1], j])
    mode.append(np.mean(dif_sum))
    mode.append(np.var(dif_sum))

    return np.array([key_clarity[0], key_clarity[1], mode[0], mode[1]], dtype=float)


def harmonic_change(y, sr):
    phi = np.array([[np.sin(i * 7 * np.pi / 6) for i in range(12)],
                    [np.cos(i * 7 * np.pi / 6) for i in range(12)],
                    [np.sin(i * 3 * np.pi / 2) for i in range(12)],
                    [np.cos(i * 3 * np.pi / 2) for i in range(12)],
                    [0.5 * np.sin(i * 2 * np.pi / 3) for i in range(12)],
                    [0.5 * np.cos(i * 2 * np.pi / 3) for i in range(12)]])
    phi = np.matrix(phi, dtype=float)

    per_sample_t = 1.0 / sr
    chromagram_frame_length, chromagram_hop_length = int(100 / 1000 / per_sample_t), int(12.5 / 1000 / per_sample_t)
    chromagram = librosa.feature.chroma_stft(y, sr=sr, n_fft=chromagram_frame_length,
                                             hop_length=chromagram_hop_length,
                                             win_length=chromagram_frame_length, window='hamming',
                                             n_chroma=12)
    # zero-column delete
    k, ln_chro = 0, chromagram.shape[1]
    while k < ln_chro:
        if k >= chromagram.shape[1]:
            break
        if np.sum(chromagram[:, k]) == 0:
            chromagram = np.delete(chromagram, k, 1)
            k -= 1
        k += 1
    zeta = np.zeros(shape=(6, chromagram.shape[1]))
    for j in range(chromagram.shape[1]):
        zeta[:, j] = np.dot(phi, chromagram[:, j]) / np.abs(np.sum(chromagram[:, j]))
    delta = []
    for j in range(1, chromagram.shape[1] - 1):
        delta.append(np.sum((zeta[:, j - 1] - zeta[:, j + 1]) ** 2))

    return np.array([np.mean(delta), np.std(delta)], dtype=float)


def extract_harmonic_features(signal, sr):

    salient_pitches = salient_pitch(signal, sr)
    salient_pitch_features = np.array([np.mean(salient_pitches), np.std(salient_pitches)])

    chroma_centroids = chroma_centroid(signal, sr)
    chroma_centroid_features = np.array([np.mean(chroma_centroids), np.std(chroma_centroids)])

    key_clarity_features = key_clarity_mode(signal, sr)

    harm_change_features = harmonic_change(signal, sr)

    all_features = np.concatenate([salient_pitch_features, chroma_centroid_features, key_clarity_features,
                                   harm_change_features])

    return all_features
