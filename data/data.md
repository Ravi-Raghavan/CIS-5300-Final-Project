# MetaHate Dataset

## Dataset Overview
For the CIS 5300: Natural Language Processing course project on hate speech detection, we use **MetaHate**, a meta-collection of 36 hate speech datasets compiled from social media comments. MetaHate contains a total of **1,226,202 posts** (publicly available: 1,101,165 instances), each labeled for hate speech. Labels are:

- `0` → Non-hate speech  
- `1` → Hate speech  

The dataset is provided in **TSV (tab-separated values) format**, with the following structure:

| Field Name | Type | Description |
|------------|------|-------------|
| text       | string  | Social media post content (unique per row) |
| label      | integer  | 0 = non-hate speech, 1 = hate speech |

**Hugging Face Source:** [MetaHate Dataset](https://huggingface.co/datasets/irlab-udc/metahate)  
**Paper:** [MetaHate Paper](https://arxiv.org/abs/2401.06526)

> **Note to Grader:** We used the publicly available dataset, which includes 1,101,165 social media posts prior to splitting into train/dev/test sets.

## Example Data
A small sample of the dataset is shown below.  
> **Note to Grader:** Hypothetical examples are provided here to avoid graphic or profane content.

| text                                      | label |
|-------------------------------------------|-------|
| I can't believe people still say this online | 0     |
| You are such a horrible person, leave!    | 1     |
| This is just funny, not harmful           | 0     |

## Data Format
- **Original file format:** TSV (`.tsv`)  
- **Columns:** `text`, `label`  
- **Unique posts:** Yes, duplicates have been removed  

> **Note to Grader:** After splitting the data into training, development, and test sets, each split is stored as a separate CSV file.

## Data Splits
The dataset is divided as follows:

- **Training set:** 80% (880,932 posts)  
- **Development set (dev):** 10% (110,116 posts)  
- **Test set:** 10% (110,117 posts)  

We ensure that the **same post does not appear in multiple splits**. This was straightforward since each post in the original dataset is unique.

Each split is stored in its respective CSV file:

```text
data/
├── train.csv
├── dev.csv
└── test.csv
