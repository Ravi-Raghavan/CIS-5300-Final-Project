# Description
This project uses a set of standard evaluation metrics to compare the simple majority-class baseline with the stronger transformer-based classifier, with a focus on imbalanced text classification where hate speech examples are much less frequent. Because accuracy alone can be misleading when one class dominates the dataset, we rely on precision, recall, F1 scores, and aggregated measures to evaluate how well each model identifies hate speech while avoiding unnecessary misclassification of non-hate text.
<br>
Accuracy measures the proportion of correct predictions, but it can be misleading when most samples belong to the non-hate class. To evaluate hate-speech detection directly, positive precision shows how often predicted hate is truly hate, while positive recall shows how many actual hate-speech examples the model successfully finds. Their combination, the positive F1 score, balances these two aspects and is especially important for minority-class evaluation. Likewise, negative precision, negative recall, and negative F1 capture the model’s reliability on non-hate examples and help ensure that improvements in detecting hate speech do not come at the cost of mislabeling harmless content. We also compute macro-F1 to average the F1 score of each class equally, micro-F1 to compute F1 on all aggregated predictions, and weighted-F1 to average class F1 scores in proportion to class frequency.



# Running the scoring script

You can run the scoring script by supplying two CSV files that each contain a single column of labels (0 or 1). The script will load the labels, compute all evaluation metrics, and print them in the format shown below. The following example assumes `true_labels.csv` contains:

`0, 0, 1, 1, 0, 1, 0, 1`

and `pred_labels.csv` contains:

`0, 1, 1, 1, 0, 0, 0, 1`


### Example arguments

```bash
python scoring.py \
    --true_labels ./results/true_labels.csv \
    --pred_labels ./results/pred_labels.csv
```

### Example outputs

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