
from pathlib import Path
import hashlib
import pandas as pd

path = Path("data/processed/phishing_emails_clean.csv")

df = pd.read_csv(path)

def normalize(text):
    return " ".join(str(text).lower().split())

df["normalized_text"] = df["Email Text"].map(normalize)

# Find normalized email texts with more than one label
label_counts = df.groupby("normalized_text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print("CONFLICTING EMAIL RECORDS")
print("=" * 50)

for number, text in enumerate(conflicting_texts, start=1):
    group = df[df["normalized_text"] == text]

    print(f"\nConflict group: {number}")
    print("Number of records:", len(group))
    print("Labels:", group["label"].value_counts().to_dict())
    print("Email length:", len(text))
    print("Email preview:", text[:150])
    print("-" * 50)

print("\nTotal conflicting normalized email texts:", len(conflicting_texts))
