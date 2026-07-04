import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
# Load fingerprints
X = pd.read_csv("data/morgan_fingerprints.csv")

df = pd.read_csv("data/cleaned_aromatase.csv")

# Keep only valid molecules (same filtering as fingerprint generation)
df["Mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()].reset_index(drop=True)

y = df["pIC50"]

print("X shape:", X.shape)
print("y shape:", y.shape)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# Save model
joblib.dump(model, "models/aromatase_rf.pkl")
print("Model saved successfully!")