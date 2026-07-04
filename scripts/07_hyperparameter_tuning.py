import pandas as pd
import joblib

from rdkit import Chem

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X = pd.read_csv("data/morgan_fingerprints.csv")

df = pd.read_csv("data/cleaned_aromatase.csv")

df["Mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()].reset_index(drop=True)

y = df["pIC50"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("Test R²:", r2_score(y_test, pred))
print("Test RMSE:", mean_squared_error(y_test, pred) ** 0.5)

joblib.dump(best_model, "models/aromatase_rf_tuned.pkl")

print("Tuned model saved!")