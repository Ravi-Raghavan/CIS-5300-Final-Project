# Import Libraries and Set Random State
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Set Random State
random_state = 42

# Load data
train_df = pd.read_csv('../data/train_data.csv')
dev_df = pd.read_csv('../data/dev_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

# Define a Majority Class Model
class MajorityClassModel:
    # Initialize
    def __init__(self):
        self.majority_class = None

    # Fit a Majority Class Model to the dataset
    def fit(self, y_train: pd.Series):
        self.majority_class = y_train.mode()[0]
        print(f"Majority class in training set = {self.majority_class}")

    # Predict the majority class for any given samples
    def predict(self, X: pd.DataFrame):
        return [self.majority_class] * len(X)

# Train a Majority Class Model
model = MajorityClassModel()
model.fit(train_df["label"])

# --- Compute Metrics (Accuracy, Precision, Recall, and F1 Score) ---
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pos_precision = precision_score(y_true, y_pred, pos_label = 1, zero_division=0)
    pos_recall = recall_score(y_true, y_pred, pos_label = 1, zero_division=0)
    neg_precision = precision_score(y_true, y_pred, pos_label = 0, zero_division=0)
    neg_recall = recall_score(y_true, y_pred, pos_label = 0, zero_division=0)
    pos_f1 = f1_score(y_true, y_pred, average = "binary", pos_label = 1, zero_division=0)
    neg_f1 = f1_score(y_true, y_pred, average = "binary", pos_label = 0, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, pos_precision, pos_recall, neg_precision, neg_recall, pos_f1, neg_f1, f1_macro, f1_micro, f1_weighted

# Predict on train, dev, and test values ---
y_pred_train = model.predict(train_df)
y_true_train = train_df["label"].values

y_pred_dev = model.predict(dev_df)
y_true_dev = dev_df['label'].values

y_pred_test = model.predict(test_df)
y_true_test = test_df['label'].values

# Obtain Accuracy, Precision, Recall, and F1 Score for each Split (i.e. Train, Dev, and Test)
train_acc, train_pos_precision, train_pos_recall, train_neg_precision, train_neg_recall, train_pos_f1, train_neg_f1, train_f1_macro, train_f1_micro, train_f1_weighted = compute_metrics(y_true_train, y_pred_train)
dev_acc, dev_pos_precision, dev_pos_recall, dev_neg_precision, dev_neg_recall, dev_pos_f1, dev_neg_f1, dev_f1_macro, dev_f1_micro, dev_f1_weighted = compute_metrics(y_true_dev, y_pred_dev)
test_acc, test_pos_precision, test_pos_recall, test_neg_precision, test_neg_recall, test_pos_f1, test_neg_f1, test_f1_macro, test_f1_micro, test_f1_weighted = compute_metrics(y_true_test, y_pred_test)

# Store metrics in Pandas Dataframe
metrics_df = pd.DataFrame({
    "Split": ["Train", "Dev", "Test"],
    "Accuracy": [train_acc, dev_acc, test_acc],
    "Positive Precision": [train_pos_precision, dev_pos_precision, test_pos_precision],
    "Positive Recall": [train_pos_recall, dev_pos_recall, test_pos_recall],
    "Negative Precision": [train_neg_precision, dev_neg_precision, test_neg_precision],
    "Negative Recall": [train_neg_recall, dev_neg_recall, test_neg_recall],
    "Positive F1": [train_pos_f1, dev_pos_f1, test_pos_f1],
    "Negative F1": [train_neg_f1, dev_neg_f1, test_neg_f1],
    "Macro F1": [train_f1_macro, dev_f1_macro, test_f1_macro],
    "Micro F1": [train_f1_micro, dev_f1_micro, test_f1_micro],
    "Weighted F1": [train_f1_weighted, dev_f1_weighted, test_f1_weighted]
})

metrics_df.to_csv("simple-baseline-results.csv", index = False)