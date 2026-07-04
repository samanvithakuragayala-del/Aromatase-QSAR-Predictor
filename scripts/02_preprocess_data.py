import pandas as pd
import numpy as np


# Load the dataset
df = pd.read_csv("data/raw/aromatase_raw.csv", sep=";")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Display first 5 rows
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/aromatase_raw.csv", sep=";")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing SMILES or Standard Value
df = df.dropna(subset=["Smiles", "Standard Value"])

# Keep only required columns
df = df[[
    "Molecule ChEMBL ID",
    "Smiles",
    "Standard Value"
]]

print("Cleaned dataset shape:", df.shape)
print(df.head())

# Save cleaned dataset
df["pIC50"] = -np.log10(df["Standard Value"] * 1e-9)

print(df[["Molecule ChEMBL ID", "Standard Value", "pIC50"]].head())
print("Cleaned dataset saved successfully!")


# Convert Standard Value to numeric
df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")

# Remove missing values after conversion
df = df.dropna(subset=["Standard Value"])

df = df[df["Standard Value"] > 0]# Convert IC50 (nM) to pIC50
df["pIC50"] = -np.log10(df["Standard Value"] * 1e-9)

print(df[["Molecule ChEMBL ID", "Standard Value", "pIC50"]].head())

from rdkit import Chem
from rdkit.Chem import Descriptors

# Convert SMILES to RDKit molecules
df["Mol"] = df["Smiles"].apply(lambda x: Chem.MolFromSmiles(str(x)))

# Remove invalid molecules
df = df[df["Mol"].notnull()]

# Calculate descriptors
df["MolWt"] = df["Mol"].apply(Descriptors.MolWt)
df["LogP"] = df["Mol"].apply(Descriptors.MolLogP)
df["HDonors"] = df["Mol"].apply(Descriptors.NumHDonors)
df["HAcceptors"] = df["Mol"].apply(Descriptors.NumHAcceptors)

print(df[["Molecule ChEMBL ID", "MolWt", "LogP", "HDonors", "HAcceptors"]].head())

df.to_csv("data/cleaned_aromatase.csv", index=False)
print("Updated cleaned dataset saved!")