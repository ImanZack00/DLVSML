import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load train/test ---
train_df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "updatedtrain.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "updatedtest.csv"))

train_texts, y_train = train_df["text"].astype(str).tolist(), train_df["label"].tolist()
test_texts, y_test = test_df["text"].astype(str).tolist(), test_df["label"].tolist()

# --- TF-IDF vectorization ---
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

# --- Logistic Regression classifier ---
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# --- Predictions ---
y_pred = clf.predict(X_test_tfidf)

# --- Metrics ---
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

results = {
    "model": "Logistic Regression (TF-IDF)",
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
print(results)

# --- Save metrics to text file ---
os.makedirs(os.path.join(BASE_DIR, "..", "results"), exist_ok=True)
with open(os.path.join(BASE_DIR, "..", "results", "updated_ml_results.txt"), "w", encoding="utf-8") as f:
    f.write("Machine Learning Results\n")
    f.write(str(results) + "\n\n")
    f.write("Class-wise report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["Positive","Negative","Neutral"]))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive","Negative","Neutral"],
            yticklabels=["Positive","Negative","Neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression (TF-IDF)")

# --- Save confusion matrix as image ---
plt.savefig(os.path.join(BASE_DIR, "..", "results", "ml_confusion_matrix.png"))
plt.close()