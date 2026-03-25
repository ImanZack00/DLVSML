import os
import torch
import pandas as pd
import gc
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# --- 1. CONFIG ---
MAX_RAM_GB = 28
BATCH_SIZE = 4
MAX_LENGTH = 128
# Point this to the folder where you put the 5 files
LOCAL_MODEL_PATH = "./bert_model"
def check_memory():
   used_gb = psutil.virtual_memory().used / (1024**3)
   if used_gb > MAX_RAM_GB:
       print(f"!!! SAFETY SHUTDOWN: RAM usage is {used_gb:.2f}GB.")
       exit()
# --- 2. LOAD DATA ---
try:
   df1 = pd.read_csv("annotated_bicodemix_publicsa_v2.csv")
   df1["label"] = df1["majority_sent"].map({"positive":0, "negative":1, "neutral":2})
   df1 = df1.rename(columns={"comment/tweet":"text"})

   df2 = pd.read_csv("MESocSentiment Corpus.csv")
   df2.columns = df2.columns.str.strip()
   df2["label"] = df2["Sentiment (All)"].map({"POSITIVE":0, "NEGATIVE":1, "NEUTRAL":2})
   df2 = df2.rename(columns={"Tweets":"text"})

   df = pd.concat([df1[["text","label"]], df2[["text","label"]]], ignore_index=True)
   df = df.dropna(subset=["label"])
   df["label"] = df["label"].astype(int)
   print(f"Data loaded: {len(df)} rows.")
except Exception as e:
   print(f"Error loading CSV files: {e}")
   exit()

# --- 3. LOAD MODEL OFFLINE ---
print("Loading BERT from local folder (No internet needed)...")

tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_PATH, clean_up_tokenization_spaces=True)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=3)

# --- 4. PREPROCESSING ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
   df["text"].astype(str).tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

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

# --- 5. TRAINING ---
training_args = TrainingArguments(
   output_dir="./results",
   eval_strategy="epoch",
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=BATCH_SIZE,
   gradient_accumulation_steps=4,
   num_train_epochs=3,
   weight_decay=0.01,
   fp16=False,
   logging_steps=50,
   save_total_limit=1
)

def compute_metrics(pred):
   labels = pred.label_ids
   preds = pred.predictions.argmax(-1)
   acc = accuracy_score(labels, preds)
   precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
   return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   compute_metrics=compute_metrics
)

if __name__ == "__main__":
   print("Starting training process offline...")
   trainer.train()
   results = trainer.evaluate()
   print("\n--- Final Results ---")
   print(results)

   # --- 5. SAVE THE MODEL ---
   output_dir = "./sentiment_model_final"

   # Create the directory if it doesn't exist
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)

   # Save the model, configuration, and tokenizer
   trainer.save_model(output_dir)
   tokenizer.save_pretrained(output_dir)

   print(f"Model saved successfully to: {output_dir}")
