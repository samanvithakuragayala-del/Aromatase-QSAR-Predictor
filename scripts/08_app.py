import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# -------------------------
# Load Model
# -------------------------
model = joblib.load("models/aromatase_rf_tuned.pkl")


# -------------------------
# Get Molecule Name
# -------------------------
def get_molecule_name(smiles):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
        f"smiles/{smiles}/property/Title/JSON"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["Title"]

        return "Unknown Molecule"

    except:
        return "Unknown Molecule"


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Aromatase QSAR Predictor",
    layout="wide"
)

st.title("🧬 Aromatase QSAR Predictor")

st.write(
    "Predict Aromatase inhibitory activity (pIC50) from a SMILES string."
)

smiles = st.text_input("Enter SMILES")

if st.button("Predict"):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("Invalid SMILES")

    else:

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )

        X = pd.DataFrame([list(fp)])

        prediction = model.predict(X)[0]

        tree_predictions = np.array(
            [tree.predict(X)[0] for tree in model.estimators_]
        )

        std = np.std(tree_predictions)

        confidence = max(
            0,
            min(100, 100 - std * 20)
        )

        distance = np.mean(np.abs(X.values))

        molecule_name = get_molecule_name(smiles)

        st.success(
            f"Predicted pIC50 = {prediction:.2f}"
        )

        st.subheader("Prediction Confidence")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={
                    "text": "Confidence (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "royalblue"
                    },
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "red"
                        },
                        {
                            "range": [50, 75],
                            "color": "yellow"
                        },
                        {
                            "range": [75, 100],
                            "color": "green"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Applicability Domain")

        if distance < 0.25:
            st.success("🟢 Inside Applicability Domain")

        elif distance < 0.40:
            st.warning("🟡 Borderline Applicability Domain")

        else:
            st.error("🔴 Outside Applicability Domain")

                    # -------------------------
        # Molecule Information
        # -------------------------

        st.subheader("Molecule Information")

        st.metric("Molecule Name", molecule_name)

        # -------------------------
        # Chemical Structure
        # -------------------------

        st.subheader("Chemical Structure")

        img = Draw.MolToImage(
            mol,
            size=(350, 350)
        )

        st.image(img)

        # -------------------------
        # Molecular Descriptors
        # -------------------------

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)

        st.subheader("Molecular Properties")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Molecular Weight",
                f"{mw:.2f}"
            )

            st.metric(
                "LogP",
                f"{logp:.2f}"
            )

            st.metric(
                "TPSA",
                f"{tpsa:.2f}"
            )

            st.metric(
                "Ring Count",
                rings
            )

        with col2:

            st.metric(
                "Formula",
                formula
            )

            st.metric(
                "H-Bond Donors",
                hbd
            )

            st.metric(
                "H-Bond Acceptors",
                hba
            )

            st.metric(
                "Rotatable Bonds",
                rotatable
            )

        # -------------------------
        # Lipinski Rule
        # -------------------------

        st.subheader("Drug-Likeness")

        lipinski = (
            mw < 500 and
            logp < 5 and
            hbd <= 5 and
            hba <= 10
        )

        if lipinski:
            st.success(
                "✅ Passes Lipinski Rule of Five"
            )
        else:
            st.error(
                "❌ Fails Lipinski Rule of Five"
            )

        # -------------------------
        # Predicted Activity
        # -------------------------

        st.subheader("Predicted Activity")

        if prediction >= 8:
            st.success("🟢 Highly Active")

        elif prediction >= 6:
            st.warning("🟡 Moderately Active")

        else:
            st.error("🔴 Weak Activity")

                    # -------------------------
        # Bioavailability Radar
        # -------------------------

        st.subheader("Bioavailability Radar")

        radar_values = [
            min(mw / 500, 1),
            min(logp / 5, 1),
            min(tpsa / 150, 1),
            min(hbd / 10, 1),
            min(hba / 10, 1),
            min(rotatable / 10, 1)
        ]

        radar_labels = [
            "MolWt",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
            "Rotatable"
        ]

        radar = go.Figure()

        radar.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_labels,
                fill="toself",
                name="Compound"
            )
        )

        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=500
        )

        st.plotly_chart(
            radar,
            use_container_width=True
        )

        # -------------------------
        # Download Report
        # -------------------------

        report = f"""
=============================
 Aromatase QSAR Report
=============================

Molecule Name : {molecule_name}

SMILES :
{smiles}

Predicted pIC50 :
{prediction:.2f}

Prediction Confidence :
{confidence:.1f} %

Applicability Domain :
{"Inside" if distance < 0.25 else "Borderline" if distance < 0.40 else "Outside"}

Molecular Formula :
{formula}

Molecular Weight :
{mw:.2f}

LogP :
{logp:.2f}

TPSA :
{tpsa:.2f}

H-Bond Donors :
{hbd}

H-Bond Acceptors :
{hba}

Rotatable Bonds :
{rotatable}

Ring Count :
{rings}

Lipinski Rule :
{"PASS" if lipinski else "FAIL"}
"""

        st.download_button(
            label="📥 Download QSAR Report",
            data=report,
            file_name="qsar_report.txt",
            mime="text/plain"
        )

        st.success("✅ Prediction completed successfully!")