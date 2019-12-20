- \__init__.py
- setup.py
- README.md
- .gitignore
- requirements.txt
- **libs**

  - \__init__.py
  - **rp_extract**
- **feature_extract**

  - \__init__.py
  - harmonic.py
  - spectral.py
  - temporal.py
  - rhythmic.py
  - feature_extract.py
- learn_kde_audio.py (Functions for mapping factor learning and emotion space mapping)


Steps this package needs to help us accomplish

1. We perform KDE on each of our training songs based on VA data and create mxm matrices of probability distributions for each
2. We extract the audio features of each of our training songs based on their clips.
3. For each new song that comes in
   1. We extract all of its audio features
   2. We perform *mapping factor learning*  based on our dictionary of known training song audio features (will be pulled from database) and **return a mapping factors vector**
   3. Given the mapping factors vector from 3.2, we perform *emotion space mapping* based on our dictionary of known KDEs (will be pulled from database) and **return a new KDE**