import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# Load fingerprints
X = pd.read_csv("data/morgan_fingerprints.csv")

# Load cleaned dataset
df = pd.read_csv("data/cleaned_aromatase.csv")

# Keep only molecules used for fingerprints
from rdkit import Chem
df["Mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()].reset_index(drop=True)

y = df["pIC50"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load("models/aromatase_rf.pkl")

# Predictions
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Experimental pIC50")
plt.ylabel("Predicted pIC50")
plt.title("Experimental vs Predicted")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.savefig("figures/predicted_vs_actual.png", dpi=300)
plt.show()