import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load train/test ---
train_df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "test.csv"))

train_texts, train_labels = train_df["text"].astype(str).tolist(), train_df["label"].tolist()
test_texts, test_labels = test_df["text"].astype(str).tolist(), test_df["label"].tolist()

# --- TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# --- Helper function ---
def evaluate_model(model, name):
    model.fit(X_train, train_labels)
    preds = model.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='weighted')
    return {
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# --- Train models ---
results = []
results.append(evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression"))
results.append(evaluate_model(MultinomialNB(), "Naive Bayes"))
results.append(evaluate_model(LinearSVC(), "SVM"))

# --- Save results ---
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
with open(os.path.join(BASE_DIR, "results", "ml_results.txt"), "w", encoding="utf-8") as f:
    f.write("Machine Learning Baseline Results\n")
    for r in results:
        f.write(f"\n{r['model']}\n")
        for k,v in r.items():
            if k != "model":
                f.write(f"{k}: {v}\n")

print("Results written to ml_results.txt")
print(len(train_df), len(test_df))