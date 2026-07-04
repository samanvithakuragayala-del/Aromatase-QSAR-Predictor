import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load cleaned dataset
df = pd.read_csv("data/cleaned_aromatase.csv")

# Convert SMILES to molecules
df["Mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()]

# Generate Morgan fingerprints
fingerprints = []

for mol in df["Mol"]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fingerprints.append(list(fp))

fp_df = pd.DataFrame(fingerprints)

print("Fingerprint shape:", fp_df.shape)

fp_df.to_csv("data/morgan_fingerprints.csv", index=False)
print("Morgan fingerprints saved successfully!")