import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load test dataset ---
test_df = pd.read_csv(r"C:\Users\User\PycharmProjects\PythonProject2\data\updatedtest.csv")
test_texts, test_labels = test_df["text"].astype(str).tolist(), test_df["label"].tolist()

# --- Reload tokenizer and model from your saved checkpoint ---
model_path = os.path.join(BASE_DIR, "models", "sentiment_model")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# --- Encode test data ---
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

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

test_dataset = SentimentDataset(test_encodings, test_labels)

# --- Use Trainer for prediction ---
trainer = Trainer(model=model)

predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive","Negative","Neutral"],
            yticklabels=["Positive","Negative","Neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BERT")
plt.savefig(os.path.join(BASE_DIR, "results", "bert_confusion_matrix.png"))
plt.close()

# --- Class-wise metrics ---
report = classification_report(y_true, y_pred, target_names=["Positive","Negative","Neutral"])
with open(os.path.join(BASE_DIR, "results", "bert_classwise_report.txt"), "w", encoding="utf-8") as f:
    f.write("Class-wise Metrics for BERT\n")
    f.write(report)

print(report)
print("✅ Confusion matrix saved to results/bert_confusion_matrix.png")
print("✅ Class-wise metrics saved to results/bert_classwise_report.txt")