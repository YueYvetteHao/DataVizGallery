import numpy as np
import streamlit as st
import plotly.graph_objects as go
from nilearn import datasets, surface

st.set_page_config(page_title="3D Brain Visualization", layout="wide")

# Aggregate lobar counts from: Valenzuela-Fuenzalida et al., JCM 2024, n=6,224
# Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11204640/
# Hemispheric split: right 51.5% / left 48.5% (p=0.352, no significant preference)
JCM_2024 = {
    "frontal":  {"total": 1812, "left_frac": 0.485},
    "temporal": {"total": 1609, "left_frac": 0.485},
    "parietal": {"total":  874, "left_frac": 0.485},
    "occipital":{"total":  388, "left_frac": 0.485},
    "insula":   {"total":  101, "left_frac": 0.485},
}

# Destrieux region keyword → clinical lobe mapping
LOBE_KEYWORDS = {
    "frontal":  ["front"],
    "temporal": ["temp"],
    "parietal": ["pariet"],
    "occipital": ["occip", "cuneus"],
    "insula":   ["insul", "g_ins"],
}



@st.cache_data
def load_brain_data():
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    pial_l_coords, pial_l_faces = surface.load_surf_mesh(fs.pial_left)
    pial_r_coords, pial_r_faces = surface.load_surf_mesh(fs.pial_right)
    infl_l_coords, infl_l_faces = surface.load_surf_mesh(fs.infl_left)
    infl_r_coords, infl_r_faces = surface.load_surf_mesh(fs.infl_right)
    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels_l = np.array(destrieux.map_left)
    labels_r = np.array(destrieux.map_right)
    region_names = list(destrieux.labels)
    return (
        pial_l_coords, pial_l_faces,
        pial_r_coords, pial_r_faces,
        infl_l_coords, infl_l_faces,
        infl_r_coords, infl_r_faces,
        labels_l, labels_r, region_names,
    )


@st.cache_data
def compute_occurrence(region_names):
    n = len(region_names)
    lh_counts = np.zeros(n)
    rh_counts = np.zeros(n)

    regions_per_lobe = {lobe: 0 for lobe in JCM_2024}
    for ridx, rname in enumerate(region_names):
        if ridx == 0:
            continue
        rlow = rname.lower()
        for lobe, kws in LOBE_KEYWORDS.items():
            if lobe in JCM_2024 and any(kw in rlow for kw in kws):
                regions_per_lobe[lobe] += 1

    for ridx, rname in enumerate(region_names):
        if ridx == 0:
            continue
        rlow = rname.lower()
        for lobe, data in JCM_2024.items():
            kws = LOBE_KEYWORDS.get(lobe, [])
            if any(kw in rlow for kw in kws):
                n_reg = regions_per_lobe[lobe] or 1
                lh_counts[ridx] += data["total"] * data["left_frac"] / n_reg
                rh_counts[ridx] += data["total"] * (1 - data["left_frac"]) / n_reg

    max_count = max(lh_counts.max(), rh_counts.max(), 1)
    return lh_counts / max_count, rh_counts / max_count, lh_counts, rh_counts


# ── Load data ─────────────────────────────────────────────────────────────────
(
    pial_l_coords, pial_l_faces,
    pial_r_coords, pial_r_faces,
    infl_l_coords, infl_l_faces,
    infl_r_coords, infl_r_faces,
    labels_l, labels_r, region_names,
) = load_brain_data()

lh_occ, rh_occ, lh_raw, rh_raw = compute_occurrence(tuple(region_names))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Brain Visualization")
    surf_type  = st.radio("Surface", ["Pial", "Inflated"], horizontal=True)
    hemisphere = st.radio("Hemisphere", ["Both", "Left", "Right"], horizontal=True)
    opacity    = st.slider("Surface opacity", 0.2, 1.0, 0.55, 0.05)
    colorscale = st.selectbox("Color scale", ["Blues", "Reds", "Plasma"], index=0)

# ── Select surface ────────────────────────────────────────────────────────────
if surf_type == "Pial":
    l_coords, l_faces = pial_l_coords, pial_l_faces
    r_coords, r_faces = pial_r_coords, pial_r_faces
else:
    l_coords = infl_l_coords.copy()
    l_coords[:, 0] -= 45
    r_coords = infl_r_coords.copy()
    r_coords[:, 0] += 45
    l_faces, r_faces = infl_l_faces, infl_r_faces


def make_vertex_arrays(labels, occ, raw):
    vals = np.array([occ[min(int(v), len(occ) - 1)] for v in labels])
    hover = [
        f"<b>{region_names[min(int(v), len(region_names) - 1)]}</b>"
        f"<br>Cases (est.): {int(raw[min(int(v), len(raw) - 1)])}"
        for v in labels
    ]
    return vals, hover


def mesh_trace(coords, faces, labels, occ, raw, name, showscale=False):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    vals, hover = make_vertex_arrays(labels, occ, raw)
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=vals,
        intensitymode="vertex",
        colorscale="Blues" if colorscale == "Blues" else colorscale,
        cmin=0.0, cmax=1.0,
        opacity=opacity,
        hovertext=hover,
        hoverinfo="text",
        showscale=showscale,
        colorbar=dict(title="Relative<br>Occurrence", thickness=14, len=0.6) if showscale else None,
        lighting=dict(ambient=0.5, diffuse=0.85, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
        name=name,
    )


# ── Build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()
if hemisphere in ("Both", "Left"):
    fig.add_trace(mesh_trace(l_coords, l_faces, labels_l, lh_occ, lh_raw, "Left",
                             showscale=(hemisphere == "Left")))
if hemisphere in ("Both", "Right"):
    fig.add_trace(mesh_trace(r_coords, r_faces, labels_r, rh_occ, rh_raw, "Right",
                             showscale=True))

frames = [
    go.Frame(layout=dict(scene_camera=dict(
        eye=dict(x=2.2 * np.sin(np.radians(a)),
                 y=2.2 * np.cos(np.radians(a)),
                 z=0.6)
    )))
    for a in np.linspace(0, 360, 72, endpoint=False)
]
fig.frames = frames

fig.update_layout(
    height=640,
    margin=dict(l=0, r=0, t=40, b=0),
    title=dict(
        text=f"GBM Tumor Location — {surf_type} Surface  (n=6,224 Valenzuela-Fuenzalida et al., JCM 2024)",
        x=0.1,  font_size=15,
    ),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="white",
        camera=dict(eye=dict(x=0, y=2.2, z=0.6)),
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.1, y=0.02, xanchor="left", yanchor="bottom",
        buttons=[
            dict(label="▶ Rotate", method="animate",
                 args=[None, {"frame": {"duration": 60, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0},
                                "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )],
)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Project 3 · 3D Brain Visualization")
st.caption(
    "GBM tumor locations mapped to Destrieux cortical atlas  ·  "
    "Drag to rotate  ·  ▶ Rotate for auto-spin  ·  Hover for region & case count"
)

st.markdown(
    """
    This page visualizes the cortical distribution of **glioblastoma (GBM) tumor occurrence**
    across brain regions using an interactive 3D surface rendering.
    Brain surface meshes and the Destrieux cortical parcellation (74 regions per hemisphere)
    are loaded via [nilearn](https://nilearn.github.io/); the 3D figure is rendered with
    [Plotly](https://plotly.com/python/).
    Data are from a [2024 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11204640/) of 123 studies (Valenzuela-Fuenzalida et al., *J. Clin. Med.*, n=6,224),
    with the assumption that lobar counts are distributed evenly across corresponding [Destrieux atlas](https://nilearn.github.io/dev/modules/description/destrieux_surface.html) regions.

    **Key finding:** the **frontal** and **temporal lobes** account for the majority of GBM
    cases (~60% combined), with the frontal lobe being the single most affected region.
    The **parietal lobe** shows intermediate involvement, while the **occipital lobe** and
    **insula** are comparatively rare sites of origin.
    No significant hemispheric preference is observed (right ≈ 51.5%, left ≈ 48.5%).
    """
)

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cases", "6,224")
c2.metric("Left Cases (est.)", f"{int(lh_raw.sum()):,}")
c3.metric("Right Cases (est.)", f"{int(rh_raw.sum()):,}")
lobe_totals = {
    lobe: int(
        lh_raw[[i for i, r in enumerate(region_names) if any(kw in r.lower() for kw in kws)]].sum()
        + rh_raw[[i for i, r in enumerate(region_names) if any(kw in r.lower() for kw in kws)]].sum()
    )
    for lobe, kws in LOBE_KEYWORDS.items() if lobe != "insula"
}
c4.metric("Most Affected Lobe", max(lobe_totals, key=lobe_totals.get).capitalize())

st.plotly_chart(fig, width="stretch")

# ── Lobe summary ──────────────────────────────────────────────────────────────
st.subheader("Cases by Lobe")
lobe_data = []
for lobe, kws in LOBE_KEYWORDS.items():
    idxs = [i for i, r in enumerate(region_names) if any(kw in r.lower() for kw in kws)]
    lh = int(lh_raw[idxs].sum()) if idxs else 0
    rh = int(rh_raw[idxs].sum()) if idxs else 0
    lobe_data.append({"Lobe": lobe.capitalize(), "Left (est.)": lh, "Right (est.)": rh, "Total": lh + rh})
st.table(lobe_data)

# ── Top regions ───────────────────────────────────────────────────────────────
#st.subheader(f"Top {top_n} Cortical Regions by Case Count")
#ranked_idx = np.argsort(lh_raw + rh_raw)[::-1]
#rows = [
#    {
#        "Rank": rank,
#        "Region": region_names[idx],
#        "Left (est.)": int(lh_raw[idx]),
#        "Right (est.)": int(rh_raw[idx]),
#        "Total (est.)": int(lh_raw[idx] + rh_raw[idx]),
#    }
#    for rank, idx in enumerate(ranked_idx[1:top_n + 1], 1)
#]
#st.table(rows)
