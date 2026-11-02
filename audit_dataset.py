
from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/phishing_emails_clean.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required = {"Email Text", "label"}
if not required.issubset(df.columns):
    raise ValueError("Dataset columns are incorrect.")

# Normalize text for comparison
def normalize(text):
    return " ".join(str(text).lower().split())

df["normalized_text"] = df["Email Text"].map(normalize)

# Create hashes to compare exact normalized email content
df["text_hash"] = df["normalized_text"].map(
    lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
)

print("\nPHISHGUARD AI - DATASET QUALITY AUDIT")
print("=" * 45)

print("\n1. Dataset size:", len(df))

# Check identical normalized email text
duplicate_text_rows = df.duplicated(
    subset=["text_hash"], keep=False
)

print(
    "Rows involved in repeated email text:",
    int(duplicate_text_rows.sum())
)

# Check emails assigned conflicting labels
label_counts = df.groupby("text_hash")["label"].nunique()
conflicting_hashes = label_counts[label_counts > 1].index

print(
    "Unique email texts with conflicting labels:",
    len(conflicting_hashes)
)

# Recreate the same train/test split
X = df["Email Text"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_hashes = set(
    df.loc[X_train.index, "text_hash"]
)

test_hashes = set(
    df.loc[X_test.index, "text_hash"]
)

overlap = train_hashes.intersection(test_hashes)

print("\n2. Train/test leakage check")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Identical normalized email texts in both sets:", len(overlap))

# Check whether the index column was the only removed column
original_rows = 18650
cleaned_rows = len(df)

print("\n3. Record reduction")
print("Original records:", original_rows)
print("Cleaned records:", cleaned_rows)
print("Records removed:", original_rows - cleaned_rows)

# Check class distribution
print("\n4. Class distribution")
print(df["label"].value_counts())

print("\n5. Audit interpretation")

if len(overlap) == 0:
    print("PASS: No exact normalized email-text overlap found.")
else:
    print("WARNING: Exact email-text overlap exists across the split.")

if len(conflicting_hashes) == 0:
    print("PASS: No identical email text has conflicting labels.")
else:
    print("WARNING: Some identical email texts have different labels.")

print("\nAudit completed.")
