# Ensemble Extension 2: Soft Voting Ensemble of BERT, BERT + CNN, BERT + LSTM

## Overview
This extension implements a **Soft Voting ensemble classifier** that combines the outputs of three independently trained models: (1) a fine-tuned BERT baseline, (2) BERT + CNN, and (3) BERT + LSTM. Instead of voting on discrete class labels, each model produces a probability distribution over the two classes (hate vs. non-hate). These probabilities are **averaged across models**, and the final prediction is obtained by selecting the class with the highest averaged probability.

The motivation behind this approach is to leverage complementary inductive biases across architectures while retaining confidence information from each model. The baseline BERT captures rich contextual semantics, the CNN-based model emphasizes local n-gram patterns, and the BiLSTM-based model captures sequential dependencies. By aggregating probabilistic outputs rather than hard labels, soft voting allows more confident models to exert greater influence, often leading to improved stability and performance without requiring additional training.

> **Note to Grader:** The base learners in this ensemble were **not retrained**. As described later in this document, we reuse the pretrained models from previous milestones as fixed base learners, with no additional weight updates performed.

## Evaluation and Metrics
The model is evaluated on the held-out **test set**. Using `sklearn` tooling, this script reports metrics including accuracy, precision, recall, and F1 for both classes. The test-set performance of the Soft Voting Ensemble Model is:

- **F1 (Hate):** 0.723

## Usage Instructions
Please make sure to have the training data (`train_data.csv`), validation data (`dev_data.csv`), and test data (`test_data.csv`) arranged according to the below folder structure

> **Note to Grader:** In addition to the data, you will also need the saved weights from Milestone #2 Fine-Tune of BERT Model, the saved weights from Milestone #3 training of BERT + CNN, and the saved weights from Milestone #3 training of BERT + LSTM. To access the weights, here is a Google Drive Link to a Shared Folder called [Model Weights](https://drive.google.com/drive/folders/1JdV65HwCtDd10AN_qBGzfz82W5dWcFil?usp=drive_link). Within this folder, you will see the following sub-folders

- Milestone2-Baseline-BERT-FinalModel (i.e. Saved Weights from Milestone #2 Fine-Tune of BERT Model)
- Milestone3-BERT-CNN-FinalModel (i.e. Saved Weights from Milestone #3 Training of BERT + CNN)
- Milestone3-BERT-BiLSTM-FinalModel (i.e. Saved Weights from Milestone #3 Training of BERT + LSTM)

Please download the above folders and structure your local repository as follows: 

Required Folder Structure
```text
├── data/
│   ├── train_data.csv
│   ├── dev_data.csv
│   └── test_data.csv
├── Milestone #2/
│   ├── Milestone2-Baseline-BERT-FinalModel # Saved Weights from Milestone #2 Fine-Tune of BERT Model
├── Milestone #3/
│   ├── Milestone3-BERT-CNN-FinalModel # Saved Weights from Milestone #3 Training of BERT + CNN
│   ├── Milestone3-BERT-BiLSTM-FinalModel # Saved Weights from Milestone #3 Training of BERT + LSTM
├── src/
│   ├── Soft_Voting.py # Script for Soft Voting ensemble
```

Running the Script

To execute the Soft Voting Ensemble, execute the following command from within the src folder
```python
python Soft_Voting.py
```

At the end of execution, you will see the following three files in your directory
- Soft-Voting-train-results.csv: List of evaluation metrics on Training Dataset
- Soft-Voting-dev-results.csv: List of evaluation metrics on Validation/Dev Dataset
- Soft-Voting-test-results.csv: List of evaluation metrics on Test Dataset

Output Metrics

The above files contain the below evaluation metrics:
- **accuracy:** Overall proportion of correctly classified posts (both hateful and non-hateful).  
- **pos_precision:** Of the posts predicted as hateful (`1`), the fraction that are actually hateful. Formula: `TP / (TP + FP)`.  
- **pos_recall:** Of all truly hateful posts, the fraction correctly identified as hateful. Formula: `TP / (TP + FN)`.  
- **neg_precision:** Of the posts predicted as non-hateful (`0`), the fraction that are actually non-hateful. Formula: `TN / (TN + FN)`.  
- **neg_recall:** Of all truly non-hateful posts, the fraction correctly identified as non-hateful. Formula: `TN / (TN + FP)`.  
- **pos_f1:** F1 score for the hateful class, harmonic mean of Pos Precision and Pos Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`.  
- **neg_f1:** F1 score for the non-hateful class, harmonic mean of Neg Precision and Neg Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`.
- **f1_macro:** Average of Pos F1 and Neg F1, treating both classes equally regardless of class frequency.  
- **f1_micro:** Global F1 considering total true positives, false positives, and false negatives across all classes.  
- **f1_weighted:** Average of Pos F1 and Neg F1 weighted by the number of instances in each class.

**__Note to Grader:__** While multiple evaluation metrics are saved in the above files, our primary metric for assessing model performance is the F1 score(i.e. pos_f1). The other metrics are provided solely for additional analysis.

Metric Prefixes

All metrics in the saved CSVs are prefixed by the split:

| Split | Prefix   | Example Metrics                                 |
|-------|----------|--------------------------------------------------|
| Train | `train_` | `train_accuracy`, `train_pos_f1`, `train_f1_macro` |
| Dev   | `dev_`   | `dev_neg_precision`, `dev_pos_recall`, `dev_f1_micro` |
| Test  | `test_`  | `test_f1_weighted`, `test_neg_recall`, `test_accuracy` |