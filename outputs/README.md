# Output Folder Overview

This section explains the purpose of each file in this directory, to help the grader understand how predictions are generated and evaluated.

## Prediction Files (`.npy`)

All `.npy` files contain NumPy arrays of binary predictions (`0` or `1`) for the **test set**, where:
- `0` = non-hateful post  
- `1` = hateful post  

Each array is aligned index-wise with `test_labels.npy` (i.e. the Ground Truth Test Labels).

- **`BERT-Predictions.npy`**  
  Predictions produced by the fine-tuned BERT classifier from Milestone #2. This serves as our strong baseline.

- **`BERT-CNN-Predictions.npy`**  
  Predictions from a hybrid BERT + CNN model, developed in Milestone #3. 

- **`BERT-LSTM-Predictions.npy`**  
  Predictions from a hybrid BERT + LSTM model, developed in Milestone #3. 

- **`simple-baseline-test-preds.npy`**  
  Predictions from a simple baseline model(majority classifier) developed in Milestone #2.

- **`Hard Vote-Predictions.npy`**  
  Ensemble predictions obtained via **hard voting**, where each base model votes for a class label and the majority label is selected. This was developed for Milestone #4.

- **`Soft Vote-Predictions.npy`**  
  Ensemble predictions obtained via **soft voting**, where predicted probabilities from multiple models are averaged before thresholding. This was developed for Milestone #4.

- **`Max Vote-Predictions.npy`**  
  Ensemble predictions where we select the class with the maximum confidence score across all models. This was developed for Milestone #4.

- **`Stacking-Predictions.npy`**  
  Predictions from a **stacked ensemble**, where a meta-classifier is trained on the outputs of multiple base models. This was developed for Milestone #4.

## Ground Truth Labels

- **`test_labels.npy`**  
  Ground-truth (gold) labels for the test set. This file is used by `score.py` to compute all evaluation metrics.

## Evaluation Script

- **`score.py`**  
  A standalone evaluation script that loads a prediction file and the ground-truth labels, then reports Accuracy, Precision, Recall, and multiple F1 variants. While multiple metrics are reported, **Pos F1 (F1 score for the hateful class)** is the primary metric used for model comparison.

## Documentation

- **`README.md`**  
  This file. It documents the evaluation metric choice, references prior work, explains how to run the evaluation, and describes the contents of the repository.


# Description of Evaluation Measure
This project focuses on leveraging state-of-the-art (SOTA) NLP techniques for hate speech detection using the MetaHate dataset, a unified benchmark for this task. Specifically, given a social media post, the goal is to classify it as hateful or non-hateful. Since this is a binary classification problem, the F1 score will be used as the primary evaluation metric, as it balances Precision and Recall—two critical aspects in hate speech detection. Recall captures the proportion of actual hate speech posts correctly identified by the model. Optimizing for recall ensures harmful content is not overlooked. Precision measures the proportion of posts labeled as hate speech that are truly hateful. Optimizing for precision reduces the risk of mislabeling benign content and unnecessarily restricting users’ expression. For social media platforms represented in the MetaHate dataset, both metrics are essential: missing hate speech allows harmful content to spread, while false positives can unfairly penalize users. Striking the right balance ensures the platform remains safe and inclusive without over-censoring legitimate expression, maintaining trust and engagement among users. In this work, the F1 score is computed specifically for the hate speech class (labeled as 1 in our dataset), as correctly identifying hateful content is our primary concern.

Mathematically, Precision, Recall, and F1 score are defined as follows: 
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score: (2 * Precision * Recall) / (Precision + Recall)

where TP, FP, and FN denote the number of true positives, false positives, and false negatives, respectively. The F1 score provides a single metric that balances Precision and Recall, making it particularly suitable for tasks like hate speech detection where both false negatives and false positives carry significant consequences.

We want to note that accuracy alone is insufficient in this scenario because the dataset is imbalanced, with only about 20% of posts labeled as hate speech. A naive classifier that always predicts “non-hateful” could achieve 80% accuracy without detecting any actual hate speech. By using the F1 score as the primary metric, we ensure that the model’s performance reflects its ability to correctly identify hateful content while minimizing false positives, providing a more meaningful evaluation for this imbalanced dataset.

To conclude, we note that our strong baseline draws direct inspiration from the paper MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection, which fine-tunes a BERT model on the MetaHate dataset. In their experiments, they also use F1 score as an evaluation metric, emphasizing its importance for balancing Precision and Recall in hate speech detection tasks. 

# Hate Speech Evaluation Metrics: Literature Review
In this section, we review prior works that utilize the F1 score as a primary evaluation metric for hate speech detection:
- [MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection](https://arxiv.org/pdf/2401.06526)
- [Advancing Hate Speech Detection with Transformers: Insights from the MetaHate](https://arxiv.org/pdf/2508.04913)

Both studies leverage the MetaHate dataset and report the F1 score to assess the effectiveness of their proposed approaches in identifying hate speech.

# Running score.py
Pass NumPy files (`.npy`) containing the ground truth and predicted labels. Each file contains a NumPy array of integers `0` or `1`, where:

- `0` represents a non-hateful post  
- `1` represents a hateful post  

## Example Command

```bash
python score.py \
    --true_labels_file test_labels.npy  \
    --pred_labels_file simple-baseline-test-preds.npy
```

## Example Output

```
Accuracy: 0.7500
Pos Precision: 0.7500
Pos Recall: 0.7500
Neg Precision: 0.7500
Neg Recall: 0.7500
Pos F1: 0.7500
Neg F1: 0.7500
F1 Macro: 0.7500
F1 Micro: 0.7500
F1 Weighted: 0.7500
```

### Output Interpretation

- **Accuracy:** Overall proportion of correctly classified posts (both hateful and non-hateful).  
- **Pos Precision:** Of the posts predicted as hateful (`1`), the fraction that are actually hateful. Formula: `TP / (TP + FP)`.  
- **Pos Recall:** Of all truly hateful posts, the fraction correctly identified as hateful. Formula: `TP / (TP + FN)`.  
- **Neg Precision:** Of the posts predicted as non-hateful (`0`), the fraction that are actually non-hateful. Formula: `TN / (TN + FN)`.  
- **Neg Recall:** Of all truly non-hateful posts, the fraction correctly identified as non-hateful. Formula: `TN / (TN + FP)`.  
- **Pos F1:** F1 score for the hateful class, harmonic mean of Pos Precision and Pos Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`.  
- **Neg F1:** F1 score for the non-hateful class, harmonic mean of Neg Precision and Neg Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`
- **F1 Macro:** Average of Pos F1 and Neg F1, treating both classes equally regardless of class frequency.  
- **F1 Micro:** Global F1 considering total true positives, false positives, and false negatives across all classes.  
- **F1 Weighted:** Average of Pos F1 and Neg F1 weighted by the number of instances in each class.


**__Note to Grader:__** While the script returns multiple evaluation metrics, our primary metric for assessing model performance is the F1 score(i.e. Pos F1). The other metrics are provided solely for additional analysis.
