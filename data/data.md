# English Hate Speech Superset

## Dataset Description
The **English Hate Speech Superset** is a dataset of **360,493 social media posts** annotated as hateful (`1`) or not (`0`). It was created by merging and preprocessing publicly available English hate speech datasets as of April 2024.

According to the authors' original paper, they selected datasets that met the following criteria: 
- have proper documentation
- can be accessed publicly
- focus on hate speech, which is defined as follows: "any kind of communication in speech, writing or behavior, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, color, descent, gender or other identity factor" (UN, 2019)

## Dataset Overview
Each post includes the following metadata:  
- `text`: the content of the post  
- `labels`: binary annotation (`1` = hateful, `0` = not hateful)  
- `source`: origin of the post (e.g., Twitter)  
- `dataset`: original dataset from which the post came  
- `nb_annotators`: number of annotators per post  
- `post_author_country_location`: inferred country of the author (when available)  

## Data Preprocessing
To prepare the dataset for modeling, the following steps were applied:  
1. Downloaded the entire CSV dataset from Hugging Face.  
2. Dropped rows with missing values(i.e. NaN) in `text` or `labels`.  
3. Renamed `labels` column to `label`
4. Keep only `text`, `label`, and `source` columns
5. Removed duplicate text entries.  
6. Split the data into **training, development, and test sets** using an **80-10-10 stratified split** based on the label to maintain class distribution.

## Dataset Splits

| Split        | Number of Posts |
|--------------|----------------|
| Training     | 286,950        |
| Development  | 35,869         |
| Test         | 35,869         |

## Data Format
After creating the above splits, the datasets were saved to the following files:
- training_data_cleaned.csv
- dev_data_cleaned.csv
- test_data_cleaned.csv

Each split is provided as a CSV file with the following columns:  
- `text` (string) – the social media post content  
- `label` (int) – `1` for hateful, `0` for non-hateful  
- `source` (string) – origin platform  


### Example of Data
| text                         | label | source  |
|-------------------------------|-------|---------|
| "I hate this group of people" | 1     | Twitter |
| "What a beautiful day!"       | 0     | Twitter |

Above, we see an example of data 

## Dataset Source
- Hugging Face: [English Hate Speech Superset](https://huggingface.co/datasets/manueltonneau/english-hate-speech-superset)