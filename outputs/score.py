from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import argparse

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

# Parse arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--true_labels_file', type=str, required=True, help='Path to .npy file containing ground truth labels')
parser.add_argument('--pred_labels_file', type=str, required=True, help='Path to .npy file containing predicted labels')
args = parser.parse_args()

# --- Ground Truth and Predicted Labels  ---
y_true = np.load(args.true_labels_file).flatten()
y_pred = np.load(args.pred_labels_file).flatten()

# --- Compute metrics ---
metrics = compute_metrics(y_true, y_pred)

# --- Print metrics ---
metric_names = ["Accuracy", "Pos Precision", "Pos Recall", "Neg Precision", "Neg Recall", 
                "Pos F1", "Neg F1", "F1 Macro", "F1 Micro", "F1 Weighted"]

for name, value in zip(metric_names, metrics):
        print(f"{name}: {value:.4f}")