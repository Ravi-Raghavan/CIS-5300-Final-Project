# Strong Baseline: Fine-Tuned BERT Model

## Overview
This strong baseline leverages a fine-tuned BERT model (bert-base-uncased) for hate speech detection. Unlike a simple majority-class classifier, this model benefits from the rich contextual language representations learned during BERT’s pre-training and adapts them to the MetaHate dataset via supervised fine-tuning. All social media posts in the MetaHate dataset are tokenized using BERT’s tokenizer with truncation and padding to a maximum sequence length of 512 tokens, enabling batch processing.

## Handling Class Imbalance
To address the class imbalance in the dataset—where hateful posts make up only about 20% of the training data while non-hateful posts constitute the remaining 80%—we employ a class-weighted cross-entropy loss. The weights are determined based on the proportion of each class in the training data, so that the model gives more emphasis to the minority class (hateful posts) during training. This custom loss is implemented by subclassing Hugging Face’s Trainer class, which allows the model to learn effectively from both classes while avoiding being biased toward the majority class.

## Training Setup
- **Epochs:** 3  
- **Batch Size:** 32  
- **Learning Rate:** 5e-5  
- **Evaluation Metric:** F1 score for the hate speech class  

During training, the model is evaluated periodically on the development set to track performance.

## Evaluation and Metrics
After training, the model is evaluated on the held-out **test set**. The evaluation uses a scoring function (i.e. `compute_metrics`) to report detailed metrics including accuracy, precision, recall, and F1 for both classes. The test-set performance of the strong baseline is:

- **F1 (Hate):** 0.712  

These results demonstrate that the fine-tuned BERT model substantially outperforms the majority-class baseline, establishing a strong performance threshold for future models.

## Usage Instructions
Please make sure to have the training data (`train_data.csv`), validation data (`dev_data.csv`), and test data (`test_data.csv`) arranged 
according to the following folder structure

📁 Folder Structure
```text
├── data/
│   ├── train_data.csv
│   ├── dev_data.csv
│   └── test_data.csv
├── src/
│   ├── strong-baseline.py # Script for Strong baseline
```

▶️ Running the Script

To execute the BERT fine-tuning, execute the following command from within the src folder
```python
python strong-baseline.py
```

Once the above command is executed the following operations will occur: 
1. Load the dataset
2. Compute class imbalance
3. Define a weighted cross-entropy loss
4. Tokenize using bert-base-uncased
5. Train for 3 epochs (batch size 32, LR = 5e-5)
6. Save checkpoints every 50 steps
7. Evaluate on train/dev/test
8. Save evaluation metrics in CSV files

At the end of execution, you will see the following three files in your directory
- strong-baseline-train-results.csv: List of evaluation metrics on Training Dataset
- strong-baseline-dev-results.csv: List of evaluation metrics on Validation/Dev Dataset
- strong-baseline-test-results.csv: List of evaluation metrics on Test Dataset

📊 Output Metrics

The above files contain the below evaluation metrics:
- **accuracy:** Overall proportion of correctly classified posts (both hateful and non-hateful).  
- **pos_precision:** Of the posts predicted as hateful (`1`), the fraction that are actually hateful. Formula: `TP / (TP + FP)`.  
- **pos_recall:** Of all truly hateful posts, the fraction correctly identified as hateful. Formula: `TP / (TP + FN)`.  
- **neg_precision:** Of the posts predicted as non-hateful (`0`), the fraction that are actually non-hateful. Formula: `TN / (TN + FN)`.  
- **neg_recall:** Of all truly non-hateful posts, the fraction correctly identified as non-hateful. Formula: `TN / (TN + FP)`.  
- **pos_f1:** F1 score for the hateful class, harmonic mean of Pos Precision and Pos Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`.  
- **neg_f1:** F1 score for the non-hateful class, harmonic mean of Neg Precision and Neg Recall.  
- **f1_macro:** Average of Pos F1 and Neg F1, treating both classes equally regardless of class frequency.  
- **f1_micro:** Global F1 considering total true positives, false positives, and false negatives across all classes.  
- **f1_weighted:** Average of Pos F1 and Neg F1 weighted by the number of instances in each class.

**__Note to Grader:__** While multiple evaluation metrics are saved in the above files, our primary metric for assessing model performance is the F1 score(i.e. pos_f1). The other metrics are provided solely for additional analysis.

🔑 Metric Prefixes

All metrics in the saved CSVs are prefixed by the split:

| Split | Prefix   | Example Metrics                                 |
|-------|----------|--------------------------------------------------|
| Train | `train_` | `train_accuracy`, `train_pos_f1`, `train_f1_macro` |
| Dev   | `dev_`   | `dev_neg_precision`, `dev_pos_recall`, `dev_f1_micro` |
| Test  | `test_`  | `test_f1_weighted`, `test_neg_recall`, `test_accuracy` |


💾 Saving the Model

During training and at the end of training, the checkpointed models, final model, and Hugging Face trainer state are saved to:
```mathematica
Milestone2-Baseline-BERT-FineTuning/
Milestone2-Baseline-BERT-FinalModel/
trainer_state.json
```