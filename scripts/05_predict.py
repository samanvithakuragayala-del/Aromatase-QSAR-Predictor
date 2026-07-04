import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Load trained model
model = joblib.load("models/aromatase_rf.pkl")

# Example molecule (Letrozole)
smiles = "N#Cc1ccc(c(c1)C(c2ccc(C#N)cc2)n3cncn3)"

mol = Chem.MolFromSmiles(smiles)

fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=2048
)

X = pd.DataFrame([list(fp)])

prediction = model.predict(X)

print("Predicted pIC50:", prediction[0])