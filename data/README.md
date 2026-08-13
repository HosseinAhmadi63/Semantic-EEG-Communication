# Dataset cache

The project uses the public BNCI2014-001 dataset, also known as BCI Competition IV Dataset 2a. MOABB downloads the files automatically on the first run and stores them in `data/mne_data/`.

Raw EEG files are not committed to this repository. The experiment maps the original MATLAB trial-level artifact flags to MOABB epochs and excludes flagged trials before fitting or evaluation.
