import pandas as pd
import json
import glob

# --- Load CSV datasets ---
csv_files = glob.glob("data/*.csv")  # adjust path
csv_dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    # normalize column names
    if "sentiment" in df.columns:
        df = df.rename(columns={"sentiment":"label", "text":"text"})
    elif "Sentiment (All)" in df.columns:
        df["label"] = df["Sentiment (All)"].map({"POSITIVE":0, "NEGATIVE":1, "NEUTRAL":2})
        df = df.rename(columns={"Tweets":"text"})
    csv_dfs.append(df[["text","label"]])

# --- Load JSON datasets ---
json_files = glob.glob("data/*.json")
json_dfs = []
for f in json_files:
    with open(f, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        # if it's just a list of strings (negative samples, etc.)
        if isinstance(data, list):
            df = pd.DataFrame({"text": data, "label": 1})  # e.g. 1 = negative
        elif isinstance(data, dict) and "root" in data:
            df = pd.DataFrame({"text": data["root"], "label": 1})
        json_dfs.append(df)

# --- Combine all ---
df = pd.concat(csv_dfs + json_dfs, ignore_index=True)
df = df.dropna(subset=["text"])
print(f"Unified dataset size: {len(df)}")