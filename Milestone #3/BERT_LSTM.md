# Extension #2: BERT + LSTM

## Overview
For the strong baseline in Milestone 2, we fine-tuned a BERT model on the MetaHate dataset to detect hate speech. BERT provides powerful contextual embeddings and effectively captures relationships between words across a sentence, but it has certain limitations. While BERT incorporates positional encodings to represent the sequential positions of tokens, its self-attention mechanism is inherently permutation-invariant, meaning that, irrespective of token order, it considers all token relationships equivalently, potentially yielding similar outputs. Consequently, changes in word order, such as “They are allies” versus “Allies they are,” may not be fully distinguished, even though the first is neutral or positive, while the second could be interpreted sarcastically or as hostile depending on context. To address these limitations, we proposed two model extensions: BERT + CNN and BERT + LSTM.

The second extension, BERT + LSTM, explicitly models the sequential structure of words, enhancing sensitivity to word order and helping BERT compensate for self-attention’s permutation-invariance. By using an LSTM on top of BERT embeddings, the model can better capture sequential dependencies, including long-range interactions between words and phrases where rearranging words changes meaning. This is especially useful for handling negations, sarcasm, or multi-clause statements often found in hate speech. The BERT + LSTM model achieved an F1 score of 0.724.

## Handling Class Imbalance
To address the class imbalance in the dataset—where hateful posts make up only about 20% of the training data while non-hateful posts constitute the remaining 80%—we employ a class-weighted cross-entropy loss. The weights are determined based on the proportion of each class in the training data, so that the model gives more emphasis to the minority class (hateful posts) during training. This custom loss is implemented by subclassing Hugging Face’s Trainer class, which allows the model to learn effectively from both classes while avoiding being biased toward the majority class.

## Training Setup
- **Epochs:** 3  
- **Batch Size:** 32  
- **Learning Rate:** 5e-5  
- **Evaluation Metric:** F1 score for the hate speech class  

During training, the model is evaluated periodically on the development set to track performance.

## Evaluation and Metrics
After training, the model is evaluated on the held-out **test set**. The evaluation uses a scoring function (i.e. `compute_metrics`) to report detailed metrics including accuracy, precision, recall, and F1 for both classes. The test-set performance of the BERT + LSTM Model is:

- **F1 (Hate):** 0.724


## Usage Instructions
Please make sure to have the training data (`train_data.csv`), validation data (`dev_data.csv`), and test data (`test_data.csv`) arranged 
according to the following folder structure

Folder Structure
```text
├── data/
│   ├── train_data.csv
│   ├── dev_data.csv
│   └── test_data.csv
├── src/
│   ├── BERT_LSTM.py # Script for BERT + LSTM
```

Running the Script

To execute the BERT + LSTM Model, execute the following command from within the src folder
```python
python BERT_LSTM.py
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
- BERT-BiLSTM-train-results.csv: List of evaluation metrics on Training Dataset
- BERT-BiLSTM-dev-results.csv: List of evaluation metrics on Validation/Dev Dataset
- BERT-BiLSTM-test-results.csv: List of evaluation metrics on Test Dataset

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


Saving the Model

During training and at the end of training, the checkpointed models, final model, and Hugging Face trainer state are saved to:
```mathematica
Milestone3-BERT-BiLSTM-FineTuning/
Milestone3-BERT-BiLSTM-FinalModel/
trainer_state.json
```
