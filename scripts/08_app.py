import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

import plotly.graph_objects as go

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Aromatase QSAR Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model = joblib.load("models/aromatase_rf.pkl")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("🧪 Aromatase QSAR")

st.sidebar.markdown("---")

st.sidebar.subheader("Developer")

st.sidebar.success("Samanvitha Kuragayala")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.info("""
Random Forest Regression

Morgan Fingerprints

2048 bits

Radius = 2
""")

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.write(
"""
AI-powered QSAR application for predicting
Aromatase inhibitor activity using
Machine Learning and RDKit.
"""
)

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------

st.title("🧪 Aromatase QSAR Predictor")

st.write(
"Predict molecular activity against the Aromatase enzyme using Machine Learning."
)

st.markdown("---")

# -------------------------------------------------------
# SMILES INPUT
# -------------------------------------------------------

smiles = st.text_input(
    "Enter SMILES",
    placeholder="Example: CCO"
)

# -------------------------------------------------------
# FEATURE GENERATION
# -------------------------------------------------------

def smiles_to_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    arr = np.array(fp)

    return arr.reshape(1, -1)

# -------------------------------------------------------
# MOLECULE PROPERTIES
# -------------------------------------------------------

def molecule_properties(mol):

    return {

        "Molecular Weight":
        round(Descriptors.MolWt(mol),2),

        "LogP":
        round(Crippen.MolLogP(mol),2),

        "TPSA":
        round(rdMolDescriptors.CalcTPSA(mol),2),

        "H Bond Donors":
        Lipinski.NumHDonors(mol),

        "H Bond Acceptors":
        Lipinski.NumHAcceptors(mol),

        "Rotatable Bonds":
        Lipinski.NumRotatableBonds(mol),

        "Heavy Atoms":
        mol.GetNumHeavyAtoms(),

        "Rings":
        rdMolDescriptors.CalcNumRings(mol)
    }

# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------

if st.button(
    "🔬 Predict Activity",
    use_container_width=True
):

    if smiles.strip() == "":
        st.warning("Please enter a SMILES string.")
        st.stop()

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("Invalid SMILES.")
        st.stop()

    X = smiles_to_features(smiles)

    prediction = float(model.predict(X)[0])

        # -------------------------------------------------------
    # MOLECULE IMAGE
    # -------------------------------------------------------

    st.markdown("---")
    st.header("🧬 Molecule Information")

    col1, col2 = st.columns([1, 1])

    with col1:

        img = Draw.MolToImage(mol, size=(450,450))
        st.image(img, caption="2D Molecular Structure")

    with col2:

        props = molecule_properties(mol)

        df = pd.DataFrame(
            props.items(),
            columns=["Property", "Value"]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # -------------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------------

    st.markdown("---")
    st.header("📊 Activity Dashboard")

    if prediction >= 8:

        risk = "🟢 Highly Active"

        color = "green"

    elif prediction >= 6:

        risk = "🟡 Moderately Active"

        color = "orange"

    else:

        risk = "🔴 Low Activity"

        color = "red"

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Predicted pIC50",
            f"{prediction:.2f}"
        )

    with metric2:

        st.metric(
            "Prediction",
            risk
        )

    # -------------------------------------------------------
    # SPEEDOMETER GAUGE
    # -------------------------------------------------------

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text":"Predicted Activity"},
            gauge={
                "axis":{"range":[0,10]},
                "bar":{"color":"royalblue"},
                "steps":[
                    {"range":[0,4],"color":"red"},
                    {"range":[4,7],"color":"yellow"},
                    {"range":[7,10],"color":"green"}
                ]
            }
        )
    )

    gauge.update_layout(height=420)

    st.plotly_chart(
        gauge,
        use_container_width=True
    )
        # -------------------------------------------------------
    # MOLECULAR DESCRIPTOR CHART
    # -------------------------------------------------------

    st.markdown("---")
    st.header("📈 Molecular Descriptor Summary")

    descriptor_df = pd.DataFrame({
        "Descriptor": [
            "Molecular Weight",
            "LogP",
            "TPSA",
            "H Bond Donors",
            "H Bond Acceptors",
            "Rotatable Bonds",
            "Heavy Atoms",
            "Rings"
        ],
        "Value": [
            props["Molecular Weight"],
            props["LogP"],
            props["TPSA"],
            props["H Bond Donors"],
            props["H Bond Acceptors"],
            props["Rotatable Bonds"],
            props["Heavy Atoms"],
            props["Rings"]
        ]
    })

    st.bar_chart(
        descriptor_df.set_index("Descriptor")
    )

    # -------------------------------------------------------
    # PREDICTION HISTORY
    # -------------------------------------------------------

    st.markdown("---")
    st.header("📝 Prediction History")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "SMILES": smiles,
        "Predicted pIC50": round(prediction,3),
        "Activity": risk
    })

    history = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------
    # DOWNLOAD HISTORY
    # -------------------------------------------------------

    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    # -------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------

    st.markdown("---")
    st.header("🧠 Interpretation")

    if prediction >= 8:

        st.success("""
This compound shows **high predicted activity** against the Aromatase target.

It could be considered a promising lead molecule for further computational or experimental validation.
""")

    elif prediction >= 6:

        st.info("""
This compound shows **moderate predicted activity**.

Structural optimization may improve its predicted potency.
""")

    else:

        st.warning("""
This compound shows **low predicted activity**.

It may require significant structural modification before further investigation.
""")
            # -------------------------------------------------------
    # DRUG LIKENESS
    # -------------------------------------------------------

    st.markdown("---")
    st.header("💊 Drug-Likeness Summary")

    violations = 0

    if props["Molecular Weight"] > 500:
        violations += 1

    if props["LogP"] > 5:
        violations += 1

    if props["H Bond Donors"] > 5:
        violations += 1

    if props["H Bond Acceptors"] > 10:
        violations += 1

    lip1, lip2 = st.columns(2)

    with lip1:

        st.metric(
            "Lipinski Violations",
            violations
        )

    with lip2:

        if violations == 0:
            st.success("Excellent Drug-Likeness")

        elif violations <= 2:
            st.info("Acceptable Drug-Likeness")

        else:
            st.error("Poor Drug-Likeness")

    # -------------------------------------------------------
    # MODEL INFORMATION
    # -------------------------------------------------------

    st.markdown("---")

    with st.expander("ℹ Model Information"):

        st.write("""
**Algorithm**

Random Forest Regression

**Fingerprint**

Morgan Fingerprint

Radius = 2

2048 Bits

**Target**

Aromatase Activity Prediction

**Framework**

RDKit + Scikit-learn + Streamlit
""")

    # -------------------------------------------------------
    # ABOUT
    # -------------------------------------------------------

    st.markdown("---")

    with st.expander("📖 About this Application"):

        st.write("""
This application predicts the biological activity
of molecules against the Aromatase enzyme.

Workflow:

• Enter SMILES

• Generate Morgan Fingerprints

• Predict activity using a trained Random Forest model

• Visualize molecular properties

• Evaluate drug-likeness

• Export prediction history
""")

    # -------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------

    st.markdown("---")

    st.caption(
        "Developed by Samanvitha Kuragayala | B.Pharm | QSAR & AI in Drug Discovery"
    )