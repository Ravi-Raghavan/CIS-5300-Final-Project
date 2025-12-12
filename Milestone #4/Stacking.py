# Import system tooling
import sys
import os
from sklearn.linear_model import LogisticRegression
from itertools import product
import tqdm
os.environ["USE_TF"] = "0"

# Get path of Milestone #2 and Milestone #3
Milestone_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Milestone #2"))
Milestone_3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Milestone #3"))
sys.path.append(Milestone_2)
sys.path.append(Milestone_3)

# Pandas, Numpy, Torch
import pandas as pd
import numpy as np
import torch
from scipy.stats import mode
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Set Random State for Reproducability
random_state = 42

# Import Hugging Face Tooling
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset

# Load Safe Tensors
from safetensors.torch import load_file

# Use CPU/MPS if possible
device = None
if "google.colab" in sys.modules:
    # Running in Colab
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    # Not in Colab (e.g., Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print("Using device:", device)

train_df = pd.read_csv('../data/train_data.csv')
dev_df = pd.read_csv('../data/dev_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

# Compute Class Proportions
p0 = (train_df['label'] == 0).mean() # Computes the percentage of our training dataset that has label = 0
p1 = (train_df['label'] == 1).mean() # Computes the percentage of our training dataset that has label = 1
print(f"{p0  * 100}% of our dataset has label = 0 and {p1  * 100}% of our dataset has label = 1")

# Define Custom Loss Criterion to Address Class Imbalance
class_weights = torch.tensor([p1, p0]).float().to(device)
custom_criterion = nn.CrossEntropyLoss(weight = class_weights)
print(f"Class Weights: {class_weights}")

# Fetch BERT Tokenizer from HuggingFace
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Define BERT + CNN Hybrid Model
class BertCNNClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=2, dropout=0.3):
        super().__init__()

        # BERT Encoder
        self.bert = BertModel.from_pretrained(bert_model_name) # Fetch BERT Encoder
        hidden_size = self.bert.config.hidden_size # Dimensionality of the encoder layers and the pooler layer

        # Define Convolutional Layers
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        # Define ReLU and Dropout 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Add Dense Layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_labels)

    # Define Forward Pass
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Fetch sequence output from BERT Encoder
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Fetch output from last hidden state
        x = sequence_output.last_hidden_state # Shape: (batch, seq_len, hidden_size)

        # Transpose x prior to convolutional layers
        x = x.permute(0, 2, 1) # Shape: (batch, hidden_size [Represents Channels], seq_len)

        # CNN + ReLU + Dropout
        x = self.conv1(x) # Shape: (Batch, 256, Output Sequence Length_1)
        x = self.relu(x) # Shape: (Batch, 256, Output Sequence Length_1)
        x = self.dropout(x) # Shape: (Batch, 256, Output Sequence Length_1)

        # CNN + ReLU
        x = self.conv2(x) # Shape: (Batch, 256, Output Sequence Length_2)
        x = self.relu(x) # Shape: (Batch, 256, Output Sequence Length_2)

        # Perform Global Max Pooling by taking the maximum across the sequence dimension for each channel
        x, _ = torch.max(x, dim = 2) # Shape: (Batch, 256)

        # Run through Dense + ReLU + Dropout + Dense
        x = self.fc1(x) # Shape: (Batch, 128)
        x = self.relu(x) # Shape: (Batch, 128)
        x = self.dropout(x) # Shape: (Batch, 128)
        logits = self.fc2(x) # Shape: (Batch, 2)

        # Return model output
        return SequenceClassifierOutput(logits=logits)

class BertLSTMClassifier(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        num_labels=2,
        hidden_dim=256,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    ):
        super().__init__()

        # Load pretrained BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        # LSTM configuration
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Fully connected classifier layers
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # BERT encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Extract sequence embeddings
        x = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Feed into LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)     # (batch, seq_len, hidden_dim * num_directions)

        if self.bidirectional:
            # Take the last hidden state of LSTM for forward and backward directions
            x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            # Take the last hidden state of LSTM
            x = h_n[-1]

        # Classifier layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return SequenceClassifierOutput(logits=logits)

# Define load model
def load_model(model_class, model_path, device="mps"):
    device = torch.device(device) # Define device
    model = None # Define model

    if "Milestone2" in model_path:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels = 2)
    else:
        model = model_class()
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# Define Model paths
BERT_LSTM_path = os.path.join(Milestone_3, "Milestone3-BERT-BiLSTM-FinalModel")
BERT_CNN_path = os.path.join(Milestone_3, "Milestone3-BERT-CNN-FinalModel")
BERT_path = os.path.join(Milestone_2, "Milestone2-Baseline-BERT-FinalModel")

# Define Models and create ensemble
BERT_LSTM = load_model(BertLSTMClassifier, BERT_LSTM_path)
BERT_CNN = load_model(BertCNNClassifier, BERT_CNN_path)
BERT = load_model(None, BERT_path)
ensemble = [BERT_LSTM, BERT_CNN, BERT]

# Create Text Dataset
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Fetch Text and Label
        text = row['text'].to_list()
        label = torch.tensor(row['label'].to_list(), dtype=torch.long)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length = self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }

# Create Datasets and DataLoaders
batch_size = 32
train_dataset = TextDataset(train_df, tokenizer)
dev_dataset  = TextDataset(dev_df, tokenizer)
test_dataset = TextDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)
dev_loader  = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define function to fetch logits from all datasets
def fetch_logits(ensemble, dataloader):
    all_logits = [] # Store All Logits
    all_labels = [] # Store All Labels

    # Set up progress bar
    pbar = tqdm.tqdm(total=len(dataloader), desc="Fetching Logits...", ncols=100)

    with torch.no_grad():
        for batch in dataloader:
            # Get Input IDs, Attention Mask
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Store logits per model
            per_model_logits = []

            # get predictions from each model
            for m in ensemble:
                out = m(input_ids=input_ids, attention_mask=attention_mask) # Get Model Output
                logits = out.logits
                per_model_logits.append(logits.cpu().numpy())

            # stack shape: (batch_size, # of logits * ensemble size) and store in all_logits
            stacked = np.hstack(per_model_logits)
            all_logits.append(stacked)
            all_labels.extend(labels.cpu().numpy().tolist())

            # Update progress bar
            pbar.update(1)

    pbar.close()
    return np.concatenate(all_logits, axis = 0), np.array(all_labels)

logits_cache = {
    "train": "train_logits.npy",
    "dev":   "dev_logits.npy",
    "test":  "test_logits.npy",
}

train_logits_path = logits_cache['train']
dev_logits_path = logits_cache['dev']
test_logits_path = logits_cache['test']

if os.path.exists(train_logits_path):
    train_logits = np.load(train_logits_path)
    train_labels = np.load("train_labels.npy")
else:
    train_logits, train_labels = fetch_logits(ensemble, train_loader)
    np.save(train_logits_path, train_logits)
    np.save("train_labels.npy", train_labels)

if os.path.exists(dev_logits_path):
    dev_logits = np.load(dev_logits_path)
    dev_labels = np.load("dev_labels.npy")
else:
    dev_logits, dev_labels = fetch_logits(ensemble, dev_loader)
    np.save(dev_logits_path, dev_logits)
    np.save("dev_labels.npy", dev_labels)

if os.path.exists(test_logits_path):
    test_logits = np.load(test_logits_path)
    test_labels = np.load("test_labels.npy")
else:
    test_logits, test_labels = fetch_logits(ensemble, test_loader)
    np.save(test_logits_path, test_logits)
    np.save("test_labels.npy", test_labels)

print(f"Compute Shape of train_logits: {train_logits.shape}, Shape of Train Labels: {train_labels.shape}")
print(f"Compute Shape of dev_logits: {dev_logits.shape}, Shape of Dev Labels: {dev_labels.shape}")
print(f"Compute Shape of test_logits: {test_logits.shape}, Shape of Test Labels: {test_labels.shape}")

# Create Logistic Regression Meta Learner
param_grid = {
    "C": [0.01, 0.1, 1, 10],           # regularization strength
    "penalty": ["l1", "l2"],                
    "max_iter": [200],
    "class_weight":[{0: p1, 1: p0}]
}

log_reg = LogisticRegression() # Create Model

# Store best param set
best_params = None
best_dev_f1 = 0
best_model = None

# Create list of param names + all combinations
keys = list(param_grid.keys())
values = list(param_grid.values())

# Iterate through the param grid
for combo in product(*values):
    params = dict(zip(keys, combo))

    if params["penalty"] == "l1":
        solver = "liblinear"
    else:
        solver = "lbfgs"

    model = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        max_iter=params['max_iter'],
        solver=solver,
        class_weight=params["class_weight"],
    )

    # Train model
    model.fit(train_logits, train_labels)

    # Evaluate on Dev Set
    dev_pred = model.predict(dev_logits)
    dev_f1 = f1_score(dev_labels, dev_pred)

    # Print Results
    print(f"Parameter Set: {params}, Dev Results: {dev_f1}")

    # If applicable, store best params
    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_params = {**params}
        best_model = model

print("Best params:", best_params)
print("Best dev F1:", best_dev_f1)

# Compute necessary metrics
def compute_metrics(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred)
    }

# Compute all split metrics
results = {
    "train": compute_metrics(best_model, train_logits, train_labels),
    "dev":   compute_metrics(best_model, dev_logits,   dev_labels),
    "test":  compute_metrics(best_model, test_logits,  test_labels),
}

# Save each split as CSV
for split_name, metrics in results.items():
    df = pd.DataFrame([metrics])
    filename = f"Stacking-{split_name}-results.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
