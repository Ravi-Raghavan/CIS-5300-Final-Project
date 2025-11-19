# Import Libraries and Set Random State
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Set Random State
random_state = 42

# Load data
train_df = pd.read_csv('.../data/train_data.csv')
dev_df = pd.read_csv('.../data/dev_data.csv')
test_df = pd.read_csv('.../data/test_data.csv')

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




# Predict on train, dev, and test values 
y_pred_train = model.predict(train_df)
y_pred_dev = model.predict(dev_df)
y_pred_test = model.predict(test_df)


# Save predictions
np.save("simple-baseline-train-preds.npy", y_pred_train)
np.save("simple-baseline-dev-preds.npy", y_pred_dev)
np.save("simple-baseline-test-preds.npy", y_pred_test)

print("Saved prediction files:")
print("  simple-baseline-train-preds.npy")
print("  simple-baseline-dev-preds.npy")
print("  simple-baseline-test-preds.npy")

