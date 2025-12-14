
**Project Overview**

This folder contains code used for the CIS 5300 final project: models, baselines, ensembling scripts, and helpers used to train, evaluate, and generate outputs for a hate-speech detection task.

**Quick Start**
- **Python:** Use Python 3.8+.

**How this README is organized**
- Short description for each file in this folder.
- How to run the main scripts and what inputs/outputs to expect.

**File Description**
- `simple-baseline.py`: Contains the code for the simple baseline implementation from Milestone #2. A more detailed description and how to run it is in `simple-baseline.md`.

- `simple-baseline.md`: Documentation of the simple baseline

- `strong-baseline.py`: A stronger classical baseline (e.g., richer preprocessing, feature engineering, or ensemble of classical models). Run with `python3 strong-baseline.py`.

- `strong-baseline.md`: Notes and evaluation results for the stronger baseline.

- `score.py`: Utility used to compute evaluation metrics (accuracy, F1, precision, recall). Use it to score prediction files against ground truth. Typical usage from command line may look like `python3 score.py truth.npy preds.npy` (adapt arguments depending on the script signature).

- `generate_outputs.py`: Script for converting model outputs into the final submission format (e.g., mapping logits to labels, writing CSV). Run after model predictions are produced: `python3 generate_outputs.py --preds preds.npy --out submission.csv` (check script flags for exact names).

- `BERT_CNN.py` and `BERT_LSTM.py`: Model training and evaluation scripts using a BERT encoder with a task-specific head:
	- `BERT_CNN.py`: BERT plus convolutional layers (text-CNN head). Trains/fine-tunes the model and saves checkpoints and predictions.
	- `BERT_LSTM.py`: BERT plus BiLSTM head. Similar behavior: training, evaluation, and saving predictions.
	Run via `python3 BERT_CNN.py` or `python3 BERT_LSTM.py`. These scripts likely rely on `transformers` and `torch` and expect dataset paths (check the top of each script for CLI args or edit the paths inline).

- `BERT_CNN.md` and `BERT_LSTM.md`: Notebooks or notes describing model architectures, hyperparameters, and experiment results.

- `Soft_Voting.py`, `Hard_Voting.py`, `Max_Voting.py`: Ensemble scripts that combine predictions from multiple models:
	- `Soft_Voting.py`: Averages (or weights) predicted probabilities (logits) across models and selects the highest average probability per sample.
	- `Hard_Voting.py`: Performs majority voting on discrete class predictions.
	- `Max_Voting.py`: Selects class by choosing the model with the highest-confidence prediction per sample (or picks the class from model with max score).
	Typical usage: collect individual model prediction files (e.g., `.npy` for logits or labels) and run `python3 Soft_Voting.py --preds model1.npy model2.npy ... --out soft-preds.npy`.

- `Stacking.py`: Implements a stacking ensemble: uses out-of-fold predictions from base models as features to train a meta-classifier. It expects base-level predictions and ground truth on the dev set to train the stacker. Run as `python3 Stacking.py` (see the script for CLI options).

- `Max_Voting.py`: (same as above — included if present to emphasize variant) Use to select the max-confidence label across models.

- `Milestone #2 (Prepare Data).ipynb`: Notebook used to prepare and explore the dataset for Milestone 2. Contains preprocessing steps used by the baselines.

- `score.py` (duplicate presence): There may be more than one `score.py` across the repo (e.g., top-level and Milestone folders). Use the one in the folder you intend to work with — they are utilities to compute metrics.

**Inputs and Outputs (general)**
- Inputs: CSV files in the `data/` folder (`train_data.csv`, `dev_data.csv`, `test_data.csv`) and any saved model checkpoints in `Milestone*` folders.
- Typical outputs: prediction `.npy` files, `*-results.csv` evaluation summaries, model checkpoints, and final submission CSVs.

**How to reproduce common tasks**
- Run the simple baseline and save predictions:
```bash
cd code
python3 simple-baseline.py
```
- Train/fine-tune a BERT model (example):
```bash
cd code
python3 BERT_CNN.py --train --data ../../data/train_data.csv --dev ../../data/dev_data.csv
```
(Check the script headers for actual argument names.)

- Create ensemble predictions after you have model outputs:
```bash
python3 Soft_Voting.py --preds modelA_logits.npy modelB_logits.npy --out soft_preds.npy
python3 generate_outputs.py --preds soft_preds.npy --out submission.csv
```

**Notes & Next Steps**
- Inspect individual scripts for exact CLI arguments and available flags — many scripts contain argument parsers near the top.
- If you'd like, I can:
	- Add a `requirements.txt` listing precise package versions used in experiments.
	- Add usage examples for each script by reading their argument parsers and documenting flags.