import streamlit as st

# Page configuration and main title
st.set_page_config(page_title="Visualization Gallery", layout="wide")

st.title("Yue Hao Data Visualization Gallery")
st.caption("Select a project from the sidebar to explore.")

st.divider()

# Three-column layout: one card per project with description and navigation link
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("01 — Network Visualization")
    st.write("Two interactive PyVis networks: a chromosome interaction network from simulated osteosarcoma copy-number profiles, and an *Arabidopsis* metabolic network with nodes colored by strength of natural selection.")
    st.page_link("pages/Project1_Network_Interaction.py", label="Open", icon="🕸️")

with col2:
    st.subheader("02 — Drug Decision ML")
    st.write("Tree-based machine learning models (XGBoost and Random Forest) predicting glioma drug response using gene expression data, with ROC curves, confusion matrices, 5-fold CV results and feature importance.")
    st.page_link("pages/Project2_Drug_Decision_ML.py", label="Open", icon="🧬")

with col3:
    st.subheader("03 — 3D Brain Visualization")
    st.write("Interactive 3D cortical surface showing glioblastoma tumor occurrence across different brain regions, drawn from a 2024 meta-analysis of 6,224 cases. Frontal and temporal lobes account for ~60% of cases combined.")
    st.page_link("pages/Project3_3D_Brain_Map.py", label="Open", icon="🧠")
