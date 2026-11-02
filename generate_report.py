
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Report directory
REPORT_DIR = Path("reports")

# Check that metrics file exists
metrics_file = REPORT_DIR / "metrics.json"

if not metrics_file.exists():
    raise FileNotFoundError("metrics.json not found. Train the model first.")

# Load saved evaluation results
with open(metrics_file, "r", encoding="utf-8") as file:
    metrics = json.load(file)

# Convert list into NumPy array
cm = np.array(metrics["confusion_matrix"])

# Verify confusion matrix dimensions
if cm.shape != (2, 2):
    raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

# Generate visualization
fig, ax = plt.subplots(figsize=(7, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("PhishGuard AI - Confusion Matrix")

fig.tight_layout()

# Save image
output = REPORT_DIR / "confusion_matrix.png"
fig.savefig(output, dpi=300, bbox_inches="tight")

plt.close(fig)

print("\nCONFUSION MATRIX GENERATED SUCCESSFULLY")
print("Saved at:", output.resolve())

print("\nEVALUATION SUMMARY")
print("------------------")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Phishing precision: {metrics['precision_phishing']:.4f}")
print(f"Phishing recall: {metrics['recall_phishing']:.4f}")
print(f"Phishing F1-score: {metrics['f1_phishing']:.4f}")

print("\nConfusion matrix:")
print(cm)
