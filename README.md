# Consumer Complaint Classifier

A text classification project that categorizes consumer financial complaints into one of five categories using both classical deep learning models (SimpleRNN, LSTM, GRU) and a fine-tuned transformer (DistilBERT). Includes an interactive Gradio demo for real-time predictions.

## Categories

- `credit_card`
- `credit_reporting`
- `debt_collection`
- `mortgages_and_loans`
- `retail_banking`



## Project overview

1. **Data cleaning & preprocessing** — lowercasing, URL/punctuation removal, stopword removal, lemmatization
2. **Label encoding** — mapping complaint categories to integer labels
3. **Baseline models** — SimpleRNN, LSTM, and GRU trained on tokenized/padded sequences
4. **Transformer model** — DistilBERT fine-tuned for sequence classification using Hugging Face `transformers`
5. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices for each model
6. **Deployment** — a Gradio web interface for live predictions using the best-performing model

## Results

| Model      | Accuracy | Precision | Recall | F1     |
|------------|----------|-----------|--------|--------|
| DistilBERT | 85.76%   | 85.80%    | 85.76% | 85.55% |
| LSTM       | 82.87%   | 84.15%    | 82.87% | 83.06% |
| GRU        | 82.82%   | 84.11%    | 82.82% | 83.03% |
| SimpleRNN  | 81.92%   | 83.09%    | 81.92% | 82.14% |

DistilBERT achieved the best performance across all metrics and is the model used in the deployed Gradio demo.

## Repository contents

- `Consumer_Complaint_Classification.ipynb` — full notebook: data prep, model training, evaluation, and Gradio demo
- `requirements.txt` — Python dependencies
- `screenshots/` — example predictions from the deployed Gradio app

## Setup

```bash
pip install -r requirements.txt
```

Open the notebook in Jupyter or Google Colab and run cells in order. Note: the notebook expects the dataset and trained model checkpoints to be available in a mounted Google Drive folder (`SAVE_DIR`) — update this path to your own setup, or run the earlier training cells to regenerate everything from scratch.

## Dataset

This project uses the [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) *(replace with your actual source, e.g. Kaggle link)*. The raw dataset is not included in this repository due to size/licensing — download it from the source and place it in the expected path before running the notebook.

## Tech stack

- TensorFlow / Keras (`tf_keras`) for RNN/LSTM/GRU models
- Hugging Face `transformers` for DistilBERT fine-tuning
- scikit-learn for preprocessing and evaluation metrics
- Gradio for the interactive demo
- NLTK for text preprocessing

## Notes

- Training was done in Google Colab with GPU acceleration
- The DistilBERT model achieved the best performance and is used as the deployed model in the Gradio demo
