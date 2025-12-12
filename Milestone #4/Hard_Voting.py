# Import system tooling
import sys
import os
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

# Define Hard Voting Function
def hard_vote_predict(ensemble, dataloader):
    all_preds = [] # Store All Predictions
    all_labels = [] # Store All Labels

    # Set up progress bar
    pbar = tqdm.tqdm(total=len(dataloader), desc="Evaluating...", ncols=100)

    with torch.no_grad():
        for batch in dataloader:
            # Get Input IDs, Attention Mask, and Labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Store predictions per model
            per_model_preds = []

            # get predictions from each model
            for m in ensemble:
                out = m(input_ids=input_ids, attention_mask=attention_mask) # Get Model Output
                logits = out.logits
                preds = torch.argmax(logits, dim=1)
                per_model_preds.append(preds.cpu().numpy())

            # stack shape: (ensemble size, batch_size)
            stacked = np.vstack(per_model_preds)

            # majority vote
            final_preds = mode(stacked, axis = 0).mode.tolist()

            # Store predictions + labels
            all_preds.extend(final_preds)
            all_labels.extend(labels.cpu().numpy().tolist())

            # Update progress bar
            pbar.update(1)

    pbar.close()
    return np.array(all_preds), np.array(all_labels)

# Evaluate and Save Results
def evaluate_and_save(ensemble_method, ensemble, dataloader, split_name):
    y_pred, y_true = hard_vote_predict(ensemble, dataloader)

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

    # --- Print metrics ---
    metrics = [acc, pos_precision, pos_recall, neg_precision, neg_recall, 
               pos_f1, neg_f1, f1_macro, f1_micro, f1_weighted]
    metric_names = ["Accuracy", "Pos Precision", "Pos Recall", "Neg Precision", "Neg Recall", 
                    "Pos F1", "Neg F1", "F1 Macro", "F1 Micro", "F1 Weighted"]
    
    # Create DataFrame
    df = pd.DataFrame([metrics], columns=metric_names)

    # Save CSV file
    filename = f"{ensemble_method}-{split_name}-results.csv"
    df.to_csv(filename, index=False)

# Evaluate!
evaluate_and_save("Hard-Voting", ensemble, test_loader, "test")