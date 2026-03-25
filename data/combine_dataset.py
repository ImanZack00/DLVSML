import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Load Kaggle JSON ---
def load_json_folder(folder, label_value):
    dfs = []
    files = glob.glob(os.path.join(folder, "*.json"))
    for f in files:
        print(f"Loading {f}")  # debug
        with open(f, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            if isinstance(data, dict) and "root" in data:
                texts = data["root"]
            elif isinstance(data, list):
                texts = data
            else:
                print(f"Skipped {f}: unexpected format")
                continue
            df = pd.DataFrame({"text": texts, "label": label_value})
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text","label"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kaggle_negative = load_json_folder(os.path.join(BASE_DIR, "kaggle", "negative"), 1)
kaggle_positive = load_json_folder(os.path.join(BASE_DIR, "kaggle", "positive"), 0)

# --- Load CSV datasets ---
def load_csv_file(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} not found")
        return pd.DataFrame(columns=["text","label"])

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame(columns=["text","label"])

    if df.empty:
        print(f"Warning: {path} produced no data")
        return pd.DataFrame(columns=["text","label"])

    # --- Map columns depending on dataset ---
    if "sentiment" in df.columns:  # news dataset
        df["label"] = df["sentiment"].map({"Positive":0, "Negative":1, "Neutral":2})
        df = df.rename(columns={"text":"text"})
    elif "Sentiment (MESocSentiment)" in df.columns:  # mesoc dataset
        df["label"] = df["Sentiment (MESocSentiment)"].map({"POSITIVE":0, "NEGATIVE":1, "NEUTRAL":2})
        df = df.rename(columns={"Tweets":"text"})
    elif "majority_sent" in df.columns:  # bicodex dataset
        df["label"] = df["majority_sent"].map({"positive":0, "negative":1, "neutral":2})
        df = df.rename(columns={"comment/tweet":"text"})
    elif "sentiment" in df.columns and "text" in df.columns:  # supervised twitter
        df["label"] = df["sentiment"].map({"Positive":0, "Negative":1, "Neutral":2})
    elif "sentiment" in df.columns and "text" not in df.columns and "id" in df.columns:  # supervised twitter politics
        df = df.rename(columns={"text":"text"})
        df["label"] = df["sentiment"].map({"Positive":0, "Negative":1, "Neutral":2})

    return df[["text","label"]]

mesoc = load_csv_file(os.path.join(BASE_DIR, "mesoc.csv"))
news = load_csv_file(os.path.join(BASE_DIR, "news-sentiment.csv"))
twitter = load_csv_file(os.path.join(BASE_DIR, "supervised-twitter.csv"))
twitterpolitics = load_csv_file(os.path.join(BASE_DIR, "supervised-twitter-politics.csv"))
bicodemix = load_csv_file(os.path.join(BASE_DIR, "annotated_bicodemix_publicsa_v2.csv"))

print("Kaggle negative:", len(kaggle_negative))
print("Kaggle positive:", len(kaggle_positive))
print("Mesoc:", len(mesoc))
print("News:", len(news))
print("Twitter:", len(twitter))
print("Twitter politics:", len(twitterpolitics))
print("Bicodemix:", len(bicodemix))

# --- Combine all datasets ---
df = pd.concat([kaggle_negative, kaggle_positive, mesoc, news, twitter, twitterpolitics, bicodemix], ignore_index=True)
print("Combined before dropna:", len(df))
df = df.dropna(subset=["text","label"])
print("Combined after dropna:", len(df))

# Debug: show unique labels before normalization
print("Unique labels before normalization:", df["label"].unique())

# --- Normalize labels ---
label_map = {
    "Positive": 0, "POSITIVE": 0, "positive": 0,
    "Negative": 1, "NEGATIVE": 1, "negative": 1,
    "Neutral": 2, "NEUTRAL": 2, "neutral": 2
}
df["label"] = df["label"].replace(label_map)

# Debug: show unique labels after normalization
print("Unique labels after normalization:", df["label"].unique())

# Convert to int
df["label"] = df["label"].astype(int)
print(f"Unified dataset size: {len(df)}")

# --- Train/Test Split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "test.csv"), index=False)

# --- Save dataset statistics ---
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

stats = df["label"].value_counts().rename({0:"Positive",1:"Negative",2:"Neutral"})
with open(os.path.join(BASE_DIR, "data", "dataset_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Dataset Summary\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write("Train/Test split sizes:\n")
    f.write(f"Train: {len(train_df)}\n")
    f.write(f"Test: {len(test_df)}\n")
    f.write("Unique labels after normalization:\n")
    f.write(str(df["label"].unique()) + "\n")
    f.write("Label counts:\n")
    for label, count in stats.items():
        f.write(f"{label}: {count}\n")

print("Train/Test split saved. Summary written to dataset_summary.txt")