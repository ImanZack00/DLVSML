import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load train/test (your paths) ---
train_df = pd.read_csv(r"C:\Users\User\PycharmProjects\DLVSML(DL)\data\data\train.csv")
test_df = pd.read_csv(r"C:\Users\User\PycharmProjects\DLVSML(DL)\data\data\test.csv")

# --- Safety: sample subset for quick run ---
train_df = train_df.sample(n=50000, random_state=42)
test_df = test_df.sample(n=10000, random_state=42)

train_texts, train_labels = train_df["text"].astype(str).tolist(), train_df["label"].tolist()
test_texts, test_labels = test_df["text"].astype(str).tolist(), test_df["label"].tolist()

# --- Tokenizer ---
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# --- Model ---
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# --- Optimized Training Arguments ---
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results_test_run"),
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,                    # safe mode: 1 epoch
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# --- Train & Evaluate ---
trainer.train()
results = trainer.evaluate()
print(results)

# --- Save model (separate folder for test run) ---
os.makedirs(os.path.join(BASE_DIR, "models_test_run"), exist_ok=True)
trainer.save_model(os.path.join(BASE_DIR, "models_test_run", "sentiment_model"))
tokenizer.save_pretrained(os.path.join(BASE_DIR, "models_test_run", "sentiment_model"))

# --- Save results to text file ---
os.makedirs(os.path.join(BASE_DIR, "results_test_run"), exist_ok=True)
with open(os.path.join(BASE_DIR, "results_test_run", "training_results.txt"), "w", encoding="utf-8") as f:
    f.write("Training Results (Test Run)\n")
    for k,v in results.items():
        f.write(f"{k}: {v}\n")

print("Safe mode test run complete. Model saved to ./models_test_run/sentiment_model and results written to results_test_run/training_results.txt")