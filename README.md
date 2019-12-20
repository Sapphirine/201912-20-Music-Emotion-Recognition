# Big Data Analytics Final Project - Music Emotion Recognition

**Jesse Cahill, Zhongling Jiang, and Zile Wang**



This repo contains the code necessary to reproduce the analysis and recommendation system for our EECSE-6893 final project.



## Project Background

Music is a medium to express emotion. According to literature, music emotion can be quantified continuously as valence and arousal (VA) density distribution on a 2-D space. However, these data are hard to retrieve as they require intense human effort to manually label songs, especially the number of songs become enormous. The goal of this project is to reproduce a model proposed by Chin, Y.-H. et.al (2018) [1], to predict VA density for a new song based on those densities for training songs as well as audio features of both new and training songs. This will help save human labeling effort on new songs in the future. Furthermore, a prototype of content-based music recommender system is built to demonstrate the usability of the algorithm. 



## Repo Structure

Our code was originally formatted into two parts:

1. A python package called `mer` which contains functions for feature extraction, emotion pdf prediction, and new song recommendation.

2. A web app built in Python using Django, which imports `mer` functions to easily perform feature extraction and make recommendations.

For  the sake of submission, we put these two pieces together into one repository.  The python package is in /mer-master and the web app is in /webapp. The `mer` package exists as a standalone repo at https://github.com/j-cahill/mer. To install it, clone or download the repo, navigate to the base directory and install with `pip install .`
