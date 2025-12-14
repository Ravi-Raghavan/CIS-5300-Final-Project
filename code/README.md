# Directory Overview

This directory contains the code for the CIS 5300 Term Project, including baseline models, extensions, and their corresponding markdown documentation.

## How this README is organized
- Short description for each file in this folder.
- How to run the main scripts and what inputs/outputs to expect.

## File Description
- [`simple-baseline.py`](simple-baseline.py): Contains the code for the simple baseline implementation from Milestone #2. A more detailed description and how to run it is in [`simple-baseline.md`](simple-baseline.md).

- [`simple-baseline.md`](simple-baseline.md): Documentation of the simple baseline from Milestone #2. Contains a more detailed description of the simple baseline and instructions on how to run [`simple-baseline.py`](simple-baseline.py)

- [`strong-baseline.py`](strong-baseline.py): Contains the code for the strong baseline implementation from Milestone #2. A more detailed description and how to run it is in [`strong-baseline.md`](strong-baseline.md)

- [`strong-baseline.md`](strong-baseline.md): Documentation of the strong baseline from Milestone #2. Contains a more detailed description of the strong baseline and instructions on how to run [`strong-baseline.py`](strong-baseline.py)

- [`score.py`](score.py): Utility used to compute evaluation metrics (accuracy, F1, precision, recall). Use it to obtain evaluation metrics by comparing prediction files against the ground truth. Typical usage from the command line: `python score.py --true_labels_file test_labels.npy --pred_labels_file prediction_file.npy`. More details will be provided below. 

- [`scoring.md`](scoring.md): Markdown file providing detailed documentation on how to use `score.py`, including explanations of the evaluation metric, example commands, and guidelines for interpreting the results.

- [`generate_outputs.py`](generate_outputs.py): Script for generating final model outputs as a `.npy` file on the test dataset. Run after training the model and saving its weights. Typical usage from the command line: `python generate_outputs.py --ensemble_method="BERT"`. More details will be provided below. 

- [`BERT_CNN.py`](BERT_CNN.py): Scripts used to train BERT + CNN as well as generate evaluation metrics on the Dev and Test Datasets. Please refer to [`BERT_CNN.md`](BERT_CNN.md) for more detailed documentation and how to run this script.

- [`BERT_CNN.md`](BERT_CNN.md): Markdown file providing detailed documentation for the BERT + CNN model from Milestone #3

- [`BERT_LSTM.py`](BERT_LSTM.py): Scripts used to train BERT + LSTM as well as generate evaluation metrics on the Dev and Test Datasets. Please refer to [`BERT_LSTM.md`](BERT_LSTM.md) for more detailed documentation and how to run this script.

- [`BERT_LSTM.md`](BERT_LSTM.md): Markdown file providing detailed documentation for the BERT + LSTM model from Milestone #3

- [`Soft_Voting.py`](Soft_Voting.py): Script used to implement a soft voting ensemble of BERT, BERT + CNN, and BERT + LSTM. Please refer to [`Soft_Voting.md`](Soft_Voting.md) for detailed documentation and instructions on running this script.

- [`Hard_Voting.py`](Hard_Voting.py): Script used to implement a hard voting ensemble of BERT, BERT + CNN, and BERT + LSTM. Please refer to [`Hard_Voting.md`](Hard_Voting.md) for detailed documentation and instructions on running this script.

- [`Stacking.py`](Stacking.py): Script used to implement a stacking ensemble of BERT, BERT + CNN, and BERT + LSTM with a Logistic Regression meta-learner. Please refer to [`Stacking.md`](Stacking.md) for detailed documentation and instructions on running this script.

- [`Max_Voting.py`](Max_Voting.py): Script used to implement a max voting ensemble of BERT, BERT + CNN, and BERT + LSTM. Please refer to [`Max_Voting.md`](Max_Voting.md) for detailed documentation and instructions on running this script.

- [`Milestone #2 (Prepare Data).ipynb`](Milestone%20%232%20(Prepare%20Data).ipynb): Notebook used to prepare and explore the dataset for Milestone 2. It samples 3% of the original dataset and performs exploratory data analysis (EDA), including computing split statistics across training, development, and test sets, and investigating class imbalance.

## Inputs and Outputs (general)
- Inputs: CSV files in the `data/` folder (`train_data.csv`, `dev_data.csv`, `test_data.csv`) and any saved model checkpoints in `Milestone*` folders.
- Typical outputs: prediction `.npy` files, `*-results.csv` evaluation summaries, model checkpoints, and final submission CSVs.

## How to reproduce common tasks
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