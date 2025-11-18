# Strong Baseline: Fine-Tuned BERT Model

## Overview
This strong baseline uses a fine-tuned **BERT (bert-base-uncased)** model to perform hate speech classification. Unlike the majority-class baseline, this model leverages contextual language representations learned during pre-training and adapts them to our dataset through supervised fine-tuning.  

All text is tokenized using BERT’s tokenizer with **truncation** and **max-length padding (512 tokens)** to ensure consistency across samples.

## Handling Class Imbalance
To account for class imbalance in the training data, we apply a **class-weighted cross-entropy loss**, where class weights are computed from the inverse frequencies of labels in the training set. The custom loss is implemented by subclassing Hugging Face’s `Trainer`.

## Training Setup
- **Epochs:** 3  
- **Batch Size:** 32  
- **Learning Rate:** 5e-5  
- **Evaluation Metric:** F1 score for the hate speech class  

During training, the model is evaluated periodically on the development set to track performance.

## Evaluation and Metrics
After training, the model is evaluated on the held-out **test set**. The evaluation uses a scoring script to report detailed metrics including accuracy, precision, recall, and F1 for both classes. The test-set performance of the strong baseline is:

- **F1 (Hate):** 0.712  

These results demonstrate that the fine-tuned BERT model substantially outperforms the majority-class baseline, establishing a strong performance threshold for future models.

To get these metrics simply run:

```python
generate_evaluation_results("test")
```python

## Generating Sample Predictions
The model can also generate predictions for individual text samples. For example:

Input: "You people should go back to your disgusting countries"

Output: "1" (Hate Speech)
