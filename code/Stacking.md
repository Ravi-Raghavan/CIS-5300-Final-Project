# Ensemble Extension 4: Stacking Ensemble of BERT, BERT + CNN, BERT + LSTM

## Overview
This extension implements a **Stacking ensemble classifier** that combines three independently trained models: (1) a fine-tuned BERT baseline, (2) BERT + CNN, and (3) BERT + LSTM. Instead of directly aggregating predictions, this approach learns how to optimally combine model outputs using a **logistic regression meta-classifier**.

For each input instance, the three base models produce **logits** for the two classes (hate vs. non-hate). These logits are concatenated to form the input features for the meta-classifier. **Training logits** from the base models are used to train the logistic regression meta-classifier. **Development (dev) logits** are used exclusively for hyperparameter tuning and model selection. Finally, **test logits** are used to report all final evaluation results, ensuring a clean separation between training, tuning, and evaluation.

The motivation behind stacking is to move beyond fixed aggregation rules (e.g., voting) and instead **learn model-specific and class-specific contributions directly from data**. By training the meta-classifier on base-model logits and tuning it on held-out dev data, the ensemble can effectively emphasize complementary strengths of the underlying architectures while avoiding information leakage, leading to more robust and principled final predictions.

> **Note to Grader:** The base learners in this ensemble were **not retrained**. As described later in this document, we reuse the pretrained models from previous milestones as fixed base learners, with no additional weight updates performed.

## Evaluation and Metrics
The model is evaluated on the held-out **test set**. Using `sklearn` tooling, this script reports metrics including accuracy, precision, recall, and F1 for both classes. The test-set performance of the Stacking Ensemble Model is:

- **F1 (Hate):** 0.715

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
│   ├── Stacking.py # Script for Stacking ensemble
```
Running the Script

To execute the Stacking Ensemble, execute the following command from within the src folder
```python
python Stacking.py
```

At the end of execution, you will see the following three files in your directory
- Stacking-train-results.csv: List of evaluation metrics on Training Dataset
- Stacking-dev-results.csv: List of evaluation metrics on Validation/Dev Dataset
- Stacking-test-results.csv: List of evaluation metrics on Test Dataset

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