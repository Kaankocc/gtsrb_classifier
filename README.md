# GTSRB Classifier

This repository contains a basic structure to train a traffic sign classifier on the GTSRB dataset.

## Project Structure

- `.venv/` - Python virtual environment (should be ignored)
- `.gitignore` - Files and directories to exclude from git
- `requirements.txt` - Library dependencies
- `data/GTSRB/` - Original GTSRB data (Final_Training and Train.csv)
- `data/processed/` - Preprocessed numpy arrays, tensors, or TFRecords
- `notebooks/` - Jupyter notebooks for exploration and training
- `src/` - Source code package for dataset, model, training, utilities
- `models/` - Saved model artifacts and checkpoints

## Setup

1. Create and activate a virtual environment, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the GTSRB dataset into `data/GTSRB/` and ensure `Train.csv` is present.

3. Use the notebooks for exploratory analysis and for training reference.

## Notes

- The repository contains skeleton code to help you get started. Add your dataset preprocessing and training code under `src/`.
- Keep model artifacts under `models/` and include good checkpointing in `train.py`.
