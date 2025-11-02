# English Hate Speech Superset – Data Description

## Overview

The **English Hate Speech Superset** is a dataset of **360,493 social media posts** annotated as hateful (`1`) or not (`0`). It was created by merging and preprocessing publicly available English hate speech datasets as of April 2024. The dataset is intended for training and evaluating hate speech detection models and studying hateful discourse online. It is **not** intended to train models that generate hateful content.  

Each post includes the following metadata:  
- `text`: the content of the post  
- `labels`: binary annotation (`1` = hateful, `0` = not hateful)  
- `source`: origin of the post (e.g., Twitter)  
- `dataset`: original dataset from which the post came  
- `nb_annotators`: number of annotators per post  
- `post_author_country_location`: inferred country of the author (when available)  

## Preprocessing

To prepare the dataset for modeling, the following steps were applied:  
1. Loaded the CSV dataset from Hugging Face.  
2. Dropped rows with missing values in `text` or `labels`.  
3. Removed duplicate text entries to avoid leakage.  
4. Split the data into **training, development, and test sets** using an **80-10-10 stratified split** based on the label to maintain class distribution.

## Dataset Splits

| Split        | Number of Posts |
|--------------|----------------|
| Training     | ~288,394       |
| Development  | ~36,049        |
| Test         | ~36,050        |

## Data Format

Each split is provided as a CSV file with the following columns:  

- `text` (string) – the social media post content  
- `label` (int) – `1` for hateful, `0` for non-hateful  
- `source` (string) – origin platform  
- `dataset` (string) – dataset of origin  
- `nb_annotators` (int) – number of annotators  
- `post_author_country_location` (string, optional) – inferred author location  

### Sample Data

| text                         | label | source  | dataset  | nb_annotators | post_author_country_location |
|-------------------------------|-------|---------|----------|---------------|----------------------------|
| "I hate this group of people" | 1     | Twitter | davidson | 3             | US                         |
| "What a beautiful day!"       | 0     | Twitter | davidson | 2             | UK                         |

## Access & Storage

- Full dataset is available as a gzipped tar archive with folders for `train`, `dev`, and `test` splits.  
- If only a sample is submitted, a link to the full dataset can be provided.  

## Source & Citation

- Hugging Face: [English Hate Speech Superset](https://huggingface.co/datasets/manueltonneau/english_hate_speech_superset)  
- Citation:

```bibtex
@inproceedings{tonneau-etal-2024-languages,
    title = "From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets",
    author = {Tonneau, Manuel  and Liu, Diyi  and Fraiberger, Samuel  and Schroeder, Ralph  and Hale, Scott  and R{\"o}ttger, Paul},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.woah-1.23",
}
