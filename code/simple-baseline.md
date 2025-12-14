# Simple Baseline: Majority-Class Classifier

## Description
This baseline model is a simple **majority-class classifier** that ignores all text features and relies only on the distribution of labels in the training set. During training, identifies the single most frequent class in the training set between 0 (non-hate) and 1 (hate). This majority class becomes the model’s single prediction rule. On new unseen data, the model consistently predicts this class for every example. Because it does not analyze text or learn any linguistic patterns, this baseline sets the lowest meaningful performance threshold that more advanced models, such as BERT, are expected to exceed. It provides a naive benchmark for assessing the impact of class imbalance and for determining whether stronger models truly learn features associated with hateful speech, rather than simply leveraging label priors from the training data.


## Sample Inputs and Outputs

Please make sure to have the training data (`train_data.csv`), validation data (`dev_data.csv`), and test data (`test_data.csv`) arranged 
according to the following folder structure

Folder Structure
```text
├── data/
│   ├── train_data.csv
│   ├── dev_data.csv
│   └── test_data.csv
├── src/
│   ├── simple-baseline.py # Script for Simple baseline
```

Running the Script

To execute the Majority Class classifier, execute the following command from within the src folder

```
python simple-baseline.py
```

Outputs:

```
Majority class in training set = 0
Saved prediction files:
  simple-baseline-train-preds.npy
  simple-baseline-dev-preds.npy
  simple-baseline-test-preds.npy
```

The script first loads the training data and determines that the majority class label in the training set is 0. It then applies this majority-class prediction rule to the train, dev, and test splits, producing one prediction per row in each dataset. These predictions, which are formatted as numpy arrays, are saved as .npy files named simple-baseline-train-preds.npy, simple-baseline-dev-preds.npy, and simple-baseline-test-preds.npy. Each file contains an array of 0's (majority class in training set) matching the length of the corresponding dataset. For test set, evaluation, "simple-baseline-test-preds.npy" can be passed into **score.py** with "ground-truth-test.npy" to compute evaluation metrics for the simple baseline. See the next section for how this is done.


## Evaluation and Metrics

To evaluate the performance of the majority-class classifier, run the following:

```
python simple-baseline.py

python score.py \
    --true_labels_file ground-truth-test.npy \
    --pred_labels_file simple-baseline-test-preds.npy
```

This will run the simple-baseline.py script to train the majority-class classifier on the labels in "train_data.csv", generate predictions on the test set from "test_data.csv", and then save those test predictions as a .npy file named "simple-baseline-test-preds.npy". That same file is then passed into **score.py** with "ground-truth-test.npy". 

**score.py** will then use the predicted labels and ground truth labels from the test set to report detailed metrics including accuracy, precision, recall, and F1 for both classes. These are the outputs using the scoring script:

```
Accuracy: 0.7881
Pos Precision: 0.0000
Pos Recall: 0.0000
Neg Precision: 0.7881
Neg Recall: 1.0000
Pos F1: 0.0000
Neg F1: 0.8815
F1 Macro: 0.4408
F1 Micro: 0.7881
F1 Weighted: 0.6948
```

The positive F1 score of 0.0 for this simple baseline arises because it always predicts the majority class, which in our dataset is non-hate (0). Since only ~20% of the posts are actually hate speech (1), the model never predicts any true positives. This shows that the model is not suitable for practical use, as it completely fails to identify any hate speech. While it may achieve 80% accuracy by predicting only non-hate (matching the ~80% majority class), it is unable to detect the presence of actual hate speech.

**__Note to Grader:__** While the script returns multiple evaluation metrics, our primary metric for assessing model performance is the F1 score(i.e. Pos F1). The other metrics are provided solely for additional analysis.
