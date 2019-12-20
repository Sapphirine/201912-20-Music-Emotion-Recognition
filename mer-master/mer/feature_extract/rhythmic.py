import numpy as np

from libs.rp_extract.rp_extract import rp_extract

def extract_rhythmic_features(signal, sr):
    rhythm = rp_extract(signal, sr, extract_rh=True, transform_db=True, transform_phon=True, transform_sone=True,
            fluctuation_strength_weighting=True,
            skip_leadin_fadeout=1,
            step_width=1)
    rhythm_hist = rhythm['rh']
    rhythm_mean = np.array([np.mean(rhythm_hist)])

    all_features = np.concatenate([rhythm_hist, rhythm_mean])

    return all_features
