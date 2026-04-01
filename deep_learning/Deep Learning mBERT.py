import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load train/test (your paths) ---
train_df = pd.read_csv(r"C:\Users\User\PycharmProjects\DLVSML(DL)\data\updatedtrain.csv")
test_df = pd.read_csv(r"C:\Users\User\PycharmProjects\DLVSML(DL)\data\updatedtest.csv")

train_texts, train_labels = train_df["text"].astype(str).tolist(), train_df["label"].tolist()
test_texts, test_labels = test_df["text"].astype(str).tolist(), test_df["label"].tolist()

# --- Load tokenizer directly from Hugging Face online ---
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")   # ✅ corrected model name

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

# --- Load model directly from Hugging Face online ---
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# --- Training Arguments (updated for better performance) ---
BATCH_SIZE = 16   # larger batch size if GPU memory allows

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    eval_strategy="epoch",       # ✅ corrected keyword
    save_strategy="epoch",             # save checkpoints each epoch
    save_total_limit=3,                # keep last 3 checkpoints
    load_best_model_at_end=True,       # reload best checkpoint
    metric_for_best_model="f1",        # use F1 to decide best model
    greater_is_better=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10,               # more epochs for higher accuracy
    learning_rate=3e-5,                # tuned LR
    weight_decay=0.01,
    fp16=True,                         # mixed precision for speed
    logging_steps=100,
    warmup_steps=1000,                 # warmup for stability
    lr_scheduler_type="linear"         # linear decay scheduler
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# --- Resume from checkpoint if available ---
checkpoint_dir = os.path.join(BASE_DIR, "results")
if os.path.isdir(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
    print("Resuming training from last checkpoint...")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# --- Evaluate and save results ---
results = trainer.evaluate()
print(results)

# --- Save model ---
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
trainer.save_model(os.path.join(BASE_DIR, "models", "sentiment_model"))
tokenizer.save_pretrained(os.path.join(BASE_DIR, "models", "sentiment_model"))

# --- Save results to text file ---
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
with open(os.path.join(BASE_DIR, "results", "updated_training_results.txt"), "w", encoding="utf-8") as f:
    f.write("Training Results\n")
    for k,v in results.items():
        f.write(f"{k}: {v}\n")

print("✅ Training complete. Best model saved to ./models/sentiment_model and results written to results/updated_training_results.txt")