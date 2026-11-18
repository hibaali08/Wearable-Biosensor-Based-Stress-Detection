# Wearable Stress Detection

This project evaluates binary stress detection using WESAD.

## Task
- Class 0: Baseline
- Class 1: Stress

## Shared data
The processed feature file is:
`outputs/wesad_features.csv`

The raw WESAD dataset is not included in this repository.
If the processed CSV is not hosted here, obtain it from the team-shared drive
and place it at `outputs/wesad_features.csv`.

## Features
- Chest EDA: mean, standard deviation, minimum, maximum
- Chest temperature: mean, standard deviation, minimum, maximum

## Evaluation
Leave-One-Subject-Out (LOSO) evaluation.
Each fold holds out one subject for testing.

## Setup
Create and activate a Python virtual environment, then install:
`pip install -r requirements.txt`

## Run preprocessing
`python src/prepare_data.py`

Run preprocessing only if you have downloaded and extracted the raw dataset
in the location expected by the script.

## Run Logistic Regression
`python src/run_logreg.py`