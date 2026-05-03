import os
import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Drug Decision ML", layout="wide")


@st.cache_data
def load_artifacts():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "xgboost_artifacts.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


a = load_artifacts()
labels = a["class_labels"]

MODELS = [
    ("Base XGBoost",      "base_xgb",  "#7ba7d1"),
    ("Tuned XGBoost",     "tuned_xgb", "#4c72b0"),
    ("Base Random Forest","base_rf",   "#e08a8a"),
    ("Tuned Random Forest","tuned_rf", "#c44e52"),
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Project 2 · Drug Sensitivity Prediction with Machine Learning")
st.caption(
    f"{a['task']}  ·  "
    f"Drugs: {' vs '.join(a['drugs'])}  ·  "
    f"{a['n_samples']} cell lines  ·  {a['n_features']} gene features (top-500 variable)"
)

st.markdown("This project involves predicting drug performance in 29 glioma cell lines using gene expression data. " \
"I integrated transcriptomic profiles from the **Cancer Cell Line Encyclopedia ([CCLE](https://sites.broadinstitute.org/ccle/))** with drug sensitivity data from the **Genomics of Drug Sensitivity in Cancer ([GDSC2](https://pharmacodb.ca/datasets/5))** database. "\
"As a first step, I framed the problem as a binary classification task to determine which of two drugs would be more effective." \
" I utilized the top 500 highly variable genes as features and applied tree-based machine learning models, XGBoost and Random Forest. Both models were optimized using random search for hyperparameter tuning. " \
"XGBoost yielded the best performance, achieving an average accuracy of 0.867 across five rounds of stratified cross-validation.")

st.markdown("**Interpretation:** In this case study, I compared Sepantronium bromide, which induces oxidative stress-mediated DNA damage<sup>[[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687778/),"
"[2](https://pubmed.ncbi.nlm.nih.gov/33322336/),"
"[3](https://pubmed.ncbi.nlm.nih.gov/37017374/)]</sup>, "
"and Staurosporine, which causes cell cycle arrest<sup>[[4](https://pubmed.ncbi.nlm.nih.gov/9858877/)]</sup>. " \
"Among the genes discriminating sensitivity between these two drugs, **MCM7** (essential for cell cycle transition)<sup>[[5](https://pubmed.ncbi.nlm.nih.gov/32089988/)]</sup> "
"and **H2AFX** (a primary responder to DNA double-strand breaks)<sup>[[6](https://pubmed.ncbi.nlm.nih.gov/34885784/)]</sup> emerged as key features. These findings align with the known mechanisms of action for each drug, highlighting the interpretability of tree-based models in biological contexts.", unsafe_allow_html=True)


st.markdown("<style>[data-testid='stMetricDelta'] svg { display: none; }</style>", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
cols = st.columns(5)
for col, (label, key, _) in zip(cols, MODELS):
    scores = a[f"{key}_cv_scores"]
    col.metric(label, f"{np.mean(scores):.3f}", f"±{np.std(scores):.3f}", delta_color="green")
cols[4].metric("Classes", len(labels))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["ROC Curves", "Model Comparison", "Confusion Matrices", "Feature Importance"]
)

# ── Tab 1: ROC Curves ─────────────────────────────────────────────────────────
with tab1:
    roc_cols = st.columns(2)
    for col, cls in zip(roc_cols, labels):
        fig_roc = go.Figure()
        for model_name, _, color in MODELS:
            d = a["roc_data"][model_name][cls]
            fig_roc.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"], mode="lines",
                name=f"{model_name} (AUC={d['auc']:.2f})",
                line=dict(color=color, width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            xaxis_range=[0, 1], yaxis_range=[0, 1.02],
            height=450, legend=dict(x=0.55, y=0.1),
            title=f"ROC — {cls} (one-vs-rest)",
            margin=dict(t=50, b=50),
        )
        col.plotly_chart(fig_roc, use_container_width=True)

# ── Tab 2: Model Comparison ───────────────────────────────────────────────────
with tab2:
    model_names = [m[0] for m in MODELS]
    means = [np.mean(a[f"{k}_cv_scores"]) for _, k, _ in MODELS]
    stds  = [np.std(a[f"{k}_cv_scores"])  for _, k, _ in MODELS]
    colors = [c for _, _, c in MODELS]

    fig_cmp = go.Figure(go.Bar(
        x=model_names, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors,
    ))
    for name, mean, std in zip(model_names, means, stds):
        fig_cmp.add_annotation(x=name, y=mean + std + 0.03, text=f"{mean:.3f}", showarrow=False)
    fig_cmp.update_layout(
        yaxis_title="5-Fold CV Accuracy", yaxis_range=[0, 1.15],
        height=400, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.subheader("Classification Reports")
    report_model = st.radio(
        "Model", [m[0] for m in MODELS], horizontal=True, key="report_model_sel"
    )
    report_key = next(k for m, k, _ in MODELS if m == report_model)
    report = a[f"{report_key}_classification_report"]
    rows = []
    for cls in labels:
        r = report.get(cls, {})
        rows.append({
            "Class": cls,
            "Precision": f"{r.get('precision', 0):.2f}",
            "Recall": f"{r.get('recall', 0):.2f}",
            "F1-Score": f"{r.get('f1-score', 0):.2f}",
            "Support": int(r.get("support", 0)),
        })
    st.table(rows)

# ── Tab 3: Confusion Matrices ─────────────────────────────────────────────────
with tab3:
    row1 = st.columns(2)
    row2 = st.columns(2)
    grid = [row1[0], row1[1], row2[0], row2[1]]

    for cell, (model_name, key, color) in zip(grid, MODELS):
        cm = np.array(a[f"{key}_confusion_matrix"])
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm, texttemplate="%{text}",
            colorscale=[[0, "white"], [1, color]],
            showscale=False,
        ))
        fig.update_layout(
            title=dict(text=model_name, font=dict(size=13)),
            xaxis_title="Predicted", yaxis_title="True",
            height=300, margin=dict(t=40, b=60, l=80, r=10),
            xaxis=dict(tickangle=-20),
        )
        cell.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Fold Accuracy (5-Fold Stratified CV)")
    folds = [f"Fold {i+1}" for i in range(5)]
    fig_folds = go.Figure()
    for model_name, key, color in MODELS:
        fig_folds.add_trace(go.Bar(
            name=model_name, x=folds, y=a[f"{key}_cv_scores"],
            marker_color=color, opacity=0.85,
        ))
    fig_folds.update_layout(
        barmode="group", yaxis_title="Accuracy",
        yaxis_range=[0, 1.05], height=350,
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_folds, use_container_width=True)

# ── Tab 4: Feature Importance ─────────────────────────────────────────────────
with tab4:
    model_sel = st.radio(
        "Model", [m[0] for m in MODELS], horizontal=True, key="fi_model_sel"
    )
    key_sel = next(k for m, k, _ in MODELS if m == model_sel)
    color_sel = next(c for m, _, c in MODELS if m == model_sel)

    top_n = st.slider("Show top N genes", 5, 20, 15)
    feats = a[f"{key_sel}_feature_importance_top20"][:top_n]
    genes = [f["gene"] for f in feats]
    descs = [f["description"] for f in feats]
    imps  = [f["importance"] for f in feats]
    hover = [f"<b>{d}</b><br>ID: {g}<br>Importance: {v:.4f}" for g, d, v in zip(genes, descs, imps)]

    fig_fi = go.Figure(go.Bar(
        x=imps[::-1], y=descs[::-1],
        orientation="h",
        marker_color=color_sel,
        hovertext=hover[::-1], hoverinfo="text",
    ))
    fig_fi.update_layout(
        xaxis_title="Feature Importance",
        yaxis_title="Gene",
        height=500, margin=dict(t=20, b=40, l=160),
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("Hover over bars to see gene ID.")

st.divider()
st.markdown("**References**")
st.markdown(
    "1. Danielpour et al. Early Cellular Responses of Prostate Carcinoma Cells to Sepantronium Bromide (YM155) Involve Suppression of mTORC1 by AMPK. "
    "[*Sci Rep.* 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687778/)\n"
    "2. Majera and Mistrik. Effect of Sepantronium Bromide (YM-155) on DNA Double-Strand Breaks Repair in Cancer Cells. "
    "[*Int. J. Mol. Sci.* 2020](https://pubmed.ncbi.nlm.nih.gov/33322336/)\n"
    "3. West et al. A Cell Type Selective YM155 Prodrug Targets Receptor-Interacting Protein Kinase 2 to Induce Brain Cancer Cell Death. "
    "[*J Am Chem Soc.* 2023](https://pubmed.ncbi.nlm.nih.gov/37017374/)\n"
    "4. Begemann et al. Growth Inhibition Induced by Ro 31-8220 and Calphostin C in Human Glioblastoma Cell Lines Is Associated with Apoptosis and Inhibition of CDC2 Kinase. "
    "[*Anticancer Res.* 1998](https://pubmed.ncbi.nlm.nih.gov/9858877/)\n"
    "5. Yu et al. MCMs in Cancer: Prognostic Potential and Mechanisms. "
    "[*Anal Cell Pathol.* 2020](https://pubmed.ncbi.nlm.nih.gov/32089988/)\n"
    "6. Merighi et al. The Phosphorylated Form of the Histone H2AX (γH2AX) in the Brain from Embryonic Life to Old Age. "
    "[*Molecules.* 2021](https://pubmed.ncbi.nlm.nih.gov/34885784/)"
)
