import pandas as pd

url = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.csv?"
    "target_chembl_id=CHEMBL1978"
)

print("Downloading Aromatase bioactivity data...")

df = pd.read_csv(url)

print("Number of records:", len(df))

df.to_csv("../data/raw/aromatase_raw.csv", index=False)

print("Dataset saved successfully!")