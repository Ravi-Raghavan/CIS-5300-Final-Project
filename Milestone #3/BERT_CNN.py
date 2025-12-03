# Import Libraries
import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn

## Ensure TensorFlow is not used
import os
os.environ["USE_TF"] = "0"

## Set Random State for Reproducability
random_state = 42

# Import Hugging Face Tooling
from transformers import BertTokenizer
from transformers import BertModel
from transformers import Trainer, TrainingArguments
import evaluate
from datasets import Dataset

# Use CPU/MPS if possible
device = None
if "google.colab" in sys.modules:
    # Running in Colab
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    # Not in Colab (e.g., Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print("Using device:", device)

# Load data
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
        return {"logits": logits}

model = BertCNNClassifier(bert_model_name, num_labels = 2) # num_labels = 2 since we have 2 classes!

# Create `Hugging Face` Datasets [Train + Dev + Test]
train_hf_dataset = Dataset.from_pandas(train_df)
dev_hf_dataset = Dataset.from_pandas(dev_df)
test_hf_dataset = Dataset.from_pandas(test_df)

# Tokenize Text Data
def tokenize_function(row):
  tokens = tokenizer(row['text'], truncation = True, padding = 'max_length', max_length = tokenizer.model_max_length)
  row['input_ids'] = tokens['input_ids']
  row['attention_mask'] = tokens['attention_mask']
  row['token_type_ids'] = tokens['token_type_ids']
  return row

train_hf_dataset = train_hf_dataset.map(tokenize_function)
dev_hf_dataset = dev_hf_dataset.map(tokenize_function)
test_hf_dataset = test_hf_dataset.map(tokenize_function)

# Define Accuracy, Precision, Recall, and F1 Metrics from Hugging Face
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load('recall')
f1_metric = evaluate.load("f1")

# Define a compute_metrics function
def compute_metrics(eval_pred):
    # Get the model predictions
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Return Metrics
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'], # Accuracy
        "pos_precision": precision_metric.compute(predictions=predictions, references=labels, pos_label = 1, average = 'binary', zero_division = 0)["precision"], # Precision on the Class w/ Label = 1 [Hate Samples]
        "pos_recall": recall_metric.compute(predictions=predictions, references=labels, pos_label = 1, average = 'binary', zero_division = 0)['recall'], # Recall on the Class w/ Label = 1 [Hate Samples]
        "pos_f1": f1_metric.compute(predictions=predictions, references=labels, pos_label = 1, average = 'binary')["f1"], # F1 Score on the Class w/ Label = 1 [Hate Samples]
        "neg_precision": precision_metric.compute(predictions=predictions, references=labels, pos_label = 0, average = 'binary', zero_division = 0)['precision'], # Precision on the Class w/ Label = 0 [Non-Hate Samples]
        "neg_recall": recall_metric.compute(predictions=predictions, references=labels, pos_label = 0, average = 'binary', zero_division = 0)['recall'], # Recall on the Class w/ Label = 0 [Non-Hate Samples]
        "neg_f1": f1_metric.compute(predictions=predictions, references=labels, pos_label = 0, average = 'binary')['f1'], # F1 Score on the Class w/ Label = 0 [Non-Hate Samples]
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average='macro')['f1'], # Macro F1 Score
        "f1_micro": f1_metric.compute(predictions=predictions, references=labels, average='micro')['f1'], # Micro F1 Score
        "f1_weighted": f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1'], # Weighted F1 Score
    }

# Subclass the `Trainer` Class from HuggingFace to use Custom Loss Criterion
# Create a subclassed Trainer that enables us to use the custom loss function defined earlier
class SubTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = custom_criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

# **Initialize the `TrainingArguments` and `Trainer`**
training_args = TrainingArguments(
    output_dir="Milestone3-BERT-CNN-FineTuning",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=3,
    save_strategy="steps",      # save checkpoints every N steps
    save_steps=50,             # save every 50 steps
    eval_strategy="steps",      # evaluate every N steps
    eval_steps=50,             # evaluate every 50 steps
    logging_strategy="steps",
    logging_steps=50,          # log every 50 steps
    report_to="none",
    full_determinism=True
)

trainer = SubTrainer(
    model=model,
    args=training_args,
    train_dataset=train_hf_dataset,
    eval_dataset=dev_hf_dataset,
    compute_metrics=compute_metrics,
)

# **Train the Model: `Fine-Tuning`**
trainer.train() # Always Resume from Last Checkpoint to Save Time
trainer.save_model('Milestone3-BERT-CNN-FinalModel') # Save the Final Model
trainer.save_state() # Save the State of the Trainer (e.g. Losses, etc)

# **Evaluate on Train, Dev, and Test Datasets**
# Split: Train, Dev, or Test
def generate_evaluation_results(split):
    dataset = None
    if split == "train":
        dataset = train_hf_dataset
    elif split == "dev" or split == "validation" or split == "val":
        dataset = dev_hf_dataset
    elif split == "test":
        dataset = test_hf_dataset
    
    results = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split)
    df_results = pd.DataFrame([results])
    df_results.to_csv(f"BERT-CNN-{split}-results.csv", index=False)
    print(f"Saved {split} evaluation metrics to BERT-CNN-{split}-results.csv")

# Generate Evaluation Results on Train, Dev, and Test Splits
generate_evaluation_results("train")
generate_evaluation_results("dev")
generate_evaluation_results("test")