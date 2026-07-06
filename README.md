# Neural Decoding of EEG Signals

Time-resolved decoding of animal vs. non-animal object categories from EEG
signals using multivariate pattern analysis (linear SVM classification) and
permutation testing.

## Overview

For each subject, EEG trials are labeled by object category (animal vs.
non-animal) and a linear SVM is trained/tested at each time point to track
how decodable the category is over the course of the trial. Decoding
accuracy is evaluated against a null distribution built by shuffling labels,
to establish above-chance significance.

There are three variants of the analysis:
- All electrodes, with pseudo-trial averaging
- Frontal electrodes only
- Posterior electrodes only
- All electrodes, using single (raw) trials instead of pseudo-averaging

## Repository structure

- `Scripts/` — the decoding pipelines (`ThingEEG_Decoding_ExamSP26_part_*.m`)
- `Data/` — `findPresentationIndices_LS.m` (maps stimulus categories to EEG
  trial indices), `object_concepts.xlsx` (stimulus category labels),
  `S0_epoched_eventlist.mat` (trial event list)
- `Matrices/` — decoding accuracy results (`DA_*.mat`) and permutation null
  distributions (`DA_sh_*.mat`)
- `Plots/` — decoding accuracy time-course figures

## Data

The raw epoched EEG recording (`S0_epoched.mat`, ~900MB) is too large for
git and is instead published as a [GitHub Release asset](../../releases).
Download it and place it in `Data/` before running the scripts.

## Requirements

MATLAB with the Statistics and Machine Learning Toolbox (`fitcsvm`,
`cvpartition`).
