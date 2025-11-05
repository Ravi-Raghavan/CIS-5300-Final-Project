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

## Example of the Data
A small sample of the dataset is shown below.  
> **Note to Grader:** Hypothetical examples are provided here to avoid graphic or profane content.

| text                                      | label |
|-------------------------------------------|-------|
| I can't believe people still say this online | 0     |
| You are such a horrible person, leave!    | 1     |
| This is just funny, not harmful           | 0     |

> **Note to Grader:** In the actual dataset, the columns are ordered label first then text. I presented the example in the above manner just to make it easier to understand the data!

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
├── training_data_cleaned.csv
├── dev_data_cleaned.csv
└── test_data_cleaned.csv
```

> **Note to Grader:** Due to the large size of the full dataset, submitting the complete train-dev-test splits to Gradescope was not feasible. Instead, I sampled **1% of each dataset** for submission. You will find the following files in the archive:

```text
data_archive.tar.gz/
├── train_sampled.csv
├── dev_sampled.csv
└── test_sampled.csv
```

Here are the number of samples in each of these files
- **Sampled Training set:** (8809 posts)  
- **Sampled Development set (dev):**  (1101 posts)  
- **Sampled Test set:** (1101 posts)  


Here is the Google Drive link to the entire train, dev, and test splits whose files, as described earlier, are named
training_data_cleaned.csv, dev_data_cleaned.csv, and test_data_cleaned.csv

**Google Drive Link:** [Full Dataset Link](https://drive.google.com/drive/folders/1An_FoKLB8422vAvH050zU21qLqOA5JXY?usp=drive_link)