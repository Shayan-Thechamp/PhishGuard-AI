
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/Phishing_Email.csv")
OUTPUT_FILE = Path("data/processed/phishing_emails_clean.csv")

LABEL_MAPPING = {
    "Safe Email": "legitimate",
    "Phishing Email": "phishing"
}

MIN_TEXT_LENGTH = 10

# 1. Read original dataset
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = ["Email Text", "Email Type"]
if not set(required).issubset(df.columns):
    raise ValueError("Required dataset columns are missing.")

original_count = len(df)

# 2. Keep only required columns
df = df[required].copy()

# 3. Remove missing and empty text
df["Email Text"] = df["Email Text"].fillna("").astype(str).str.strip()

before = len(df)
df = df[df["Email Text"] != ""].copy()
missing_removed = before - len(df)

# 4. Validate labels
df["Email Type"] = df["Email Type"].astype(str).str.strip()

unknown = set(df["Email Type"].unique()) - set(LABEL_MAPPING)

if unknown:
    raise ValueError(f"Unknown labels found: {unknown}")

df["label"] = df["Email Type"].map(LABEL_MAPPING)

# 5. Remove very short text
df["normalized_text"] = (
    df["Email Text"]
    .str.lower()
    .str.split()
    .str.join(" ")
)

before = len(df)
df = df[df["normalized_text"].str.len() >= MIN_TEXT_LENGTH].copy()
short_removed = before - len(df)

# 6. Remove normalized email texts with conflicting labels
label_counts = df.groupby("normalized_text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

conflict_removed = df["normalized_text"].isin(
    conflicting_texts
).sum()

df = df[~df["normalized_text"].isin(conflicting_texts)].copy()

# 7. Remove repeated normalized text with the same label
before = len(df)

df = df.drop_duplicates(
    subset=["normalized_text"],
    keep="first"
).copy()

duplicates_removed = before - len(df)

# 8. Keep only model training columns
df = df[["Email Text", "label"]].copy()

# 9. Final validation
if df.empty:
    raise ValueError("Cleaning removed all records.")

if df["label"].nunique() != 2:
    raise ValueError("Both classification labels must remain.")

if df.isnull().any().any():
    raise ValueError("Missing values remain in cleaned dataset.")

# 10. Save cleaned dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# 11. Print cleaning report
print("\nPHISHGUARD AI - CLEANING REPORT")
print("=" * 45)

print("Original records:", original_count)
print("Missing/empty records removed:", missing_removed)
print("Short text records removed:", short_removed)
print("Conflicting-label records removed:", conflict_removed)
print("Repeated normalized records removed:", duplicates_removed)

print("\nFinal records:", len(df))
print("\nFinal class distribution:")
print(df["label"].value_counts())

print("\nFinal columns:", df.columns.tolist())
print("Saved:", OUTPUT_FILE.resolve())
print("\nDATA CLEANING COMPLETED SUCCESSFULLY.")
