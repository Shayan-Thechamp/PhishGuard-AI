
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_FILE = Path("data/processed/phishing_emails_clean.csv")
MODEL_FILE = Path("models/phishguard_model.joblib")
REPORT_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# 2. Verify dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Clean dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = {"Email Text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain: {required_columns}"
    )

df = df.dropna(subset=["Email Text", "label"])

X = df["Email Text"].astype(str)
y = df["label"].astype(str)

if set(y.unique()) != {"legitimate", "phishing"}:
    raise ValueError(f"Unexpected labels: {y.unique()}")

if len(df) < 100:
    raise ValueError("Dataset unexpectedly small. Check cleaning.")

print("Dataset loaded successfully.")
print("Total records:", len(df))
print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Split training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Build machine learning pipeline
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("\nTraining model. Please wait...")

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 6. Evaluate on held-out test data
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, pos_label="phishing", zero_division=0
)
recall = recall_score(
    y_test, predictions, pos_label="phishing", zero_division=0
)
f1 = f1_score(
    y_test, predictions, pos_label="phishing", zero_division=0
)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification report:")
report = classification_report(
    y_test,
    predictions,
    labels=["legitimate", "phishing"],
    zero_division=0
)
print(report)

# --------------------------------------------------
# 7. Save model
# --------------------------------------------------

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

# --------------------------------------------------
# 8. Save evaluation report
# --------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["legitimate", "phishing"]
)

metrics = {
    "dataset_records": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "accuracy": float(accuracy),
    "precision_phishing": float(precision),
    "recall_phishing": float(recall),
    "f1_phishing": float(f1),
    "confusion_matrix": cm.tolist()
}

with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\nCONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print("Order: legitimate, phishing")
print(cm)

print("\nMODEL SAVED:", MODEL_FILE.resolve())
print("METRICS SAVED:", (REPORT_DIR / "metrics.json").resolve())
print("REPORT SAVED:", (REPORT_DIR / "classification_report.txt").resolve())

print("\nTRAINING AND EVALUATION COMPLETED.")
