import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from scipy.stats import chisquare

st.set_page_config(page_title="Network Interaction", layout="wide")


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX",
]
CHR_LABELS = [c.replace("chr", "") for c in CHROMS]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_sv_log():
    return pd.read_csv(os.path.join(DATA_DIR, "simulated_sv_log.csv"))


@st.cache_data
def load_network_data():
    nodes = pd.read_csv(
        os.path.join(DATA_DIR, "AtAl_KaKs_per_rxn_numgene.txt"),
        sep=r"\s+", header=None, names=["ID", "KaKs", "numgene"], engine="python",
    )
    edges = pd.read_csv(
        os.path.join(DATA_DIR, "AtAl_KaKs_edges.txt"),
        sep=r"\s+", header=None, names=["Source", "Target"], engine="python",
    )
    return nodes, edges


# ── Chromosome data transformers ──────────────────────────────────────────────

@st.cache_data
def get_translocation_matrix():
    sv_log = load_sv_log()
    trans = pd.DataFrame(0, index=CHROMS, columns=CHROMS)
    df = sv_log[sv_log["type"] == "translocation"]
    for _, row in df.iterrows():
        d, r = row["donor"], row["recipient"]
        if d in trans.index and r in trans.columns:
            trans.loc[d, r] += 1
            trans.loc[r, d] += 1
    trans.index = CHR_LABELS
    trans.columns = CHR_LABELS
    return trans


@st.cache_data
def get_intra_sv_counts():
    sv_log = load_sv_log()
    INTRA_TYPES = ["full_deletion", "partial_deletion", "duplication", "fragmentation", "fusion"]

    non_fusion = sv_log[
        sv_log["type"].isin(["full_deletion", "partial_deletion", "duplication", "fragmentation"])
    ][["type", "chrom"]].dropna()

    fusions = sv_log[sv_log["type"] == "fusion"]
    fusion_chroms = pd.concat([
        fusions[["chrom1"]].rename(columns={"chrom1": "chrom"}),
        fusions[["chrom2"]].rename(columns={"chrom2": "chrom"}),
    ]).dropna()
    fusion_chroms["type"] = "fusion"

    all_intra = pd.concat([non_fusion, fusion_chroms])
    grp = all_intra.groupby(["type", "chrom"]).size()

    counts = {t: {c: 0 for c in CHROMS} for t in INTRA_TYPES}
    for (t, c), n in grp.items():
        if t in counts and c in counts[t]:
            counts[t][c] = n

    df = pd.DataFrame(counts, index=CHROMS)[INTRA_TYPES]
    df.index = CHR_LABELS
    return df


# ── Plot builders ──────────────────────────────────────────────────────────────

def plot_translocation_heatmap(trans_matrix, intra_df):
    labels = trans_matrix.columns.tolist()
    n = len(labels)
    z = trans_matrix.values.astype(float)
    mask = np.triu(np.ones_like(z, dtype=bool), k=0)
    z_masked = z.copy()
    z_masked[mask] = np.nan

    text = [
        [str(int(z[i, j])) if not mask[i, j] else "" for j in range(n)]
        for i in range(n)
    ]
    # customdata[i][j] = [x-chr label, y-chr label]
    custom = [[[labels[j], labels[i]] for j in range(n)] for i in range(n)]

    INTRA_TYPES  = ["full_deletion", "partial_deletion", "duplication", "fragmentation", "fusion"]
    INTRA_LABELS = ["Full deletion", "Partial deletion", "Duplication", "Fragmentation", "Fusion"]
    INTRA_COLORS = ["#8B0000", "#e8a0a0", "#2ecc71", "#3498db", "#e67e22"]

    sv_hover = []
    for lbl in labels:
        lines = [f"<b>chr{lbl} — intra-chr SVs</b>"]
        for t, l, c in zip(INTRA_TYPES, INTRA_LABELS, INTRA_COLORS):
            val = int(intra_df.loc[lbl, t])
            if val > 0:
                lines.append(f"<span style='color:{c}'>&#9632;</span> {l}: {val}")
        sv_hover.append("<br>".join(lines))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z_masked,
        x=list(range(n)),
        y=list(range(n)),
        colorscale=[[0, "#fff5f5"], [1, "firebrick"]],
        colorbar=dict(title="Count", len=0.55, thickness=12,
                      tickfont=dict(color="black"), title_font=dict(color="black")),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=8, color="black"),
        xgap=0,
        ygap=0,
        hovertemplate="<b>chr%{customdata[0]}</b> × <b>chr%{customdata[1]}</b><br>Translocations: %{z:.0f}<extra></extra>",
        customdata=custom,
        showscale=True,
        name="",
    ))

    # Invisible row of points just below the last heatmap row, one per chromosome.
    # Hovering near the x-axis chromosome labels triggers their SV breakdown tooltip.
    fig.add_trace(go.Scatter(
        x=list(range(n)),
        y=[n] * n,
        mode="markers",
        marker=dict(size=24, color="rgba(0,0,0,0)", symbol="square"),
        hovertext=sv_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
        name="",
    ))

    fig.update_layout(
        title=dict(text="", font=dict(color="black")),
        xaxis=dict(tickvals=list(range(n)), ticktext=labels,
                   title=dict(text="Chromosome", font=dict(color="black")),
                   tickfont=dict(size=9, color="black"),
                   showgrid=False, zeroline=False),
        yaxis=dict(tickvals=list(range(n)), ticktext=labels,
                   title=dict(text="Chromosome", font=dict(color="black")),
                   tickfont=dict(size=9, color="black"),
                   showgrid=False, zeroline=False,
                   range=[n + 0.5, -0.5]),
        height=500,
        margin=dict(l=50, r=20, t=50, b=55),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_sv_barchart(intra_df):
    INTRA_TYPES  = ["full_deletion", "partial_deletion", "duplication", "fragmentation", "fusion"]
    INTRA_LABELS = ["Full deletion", "Partial deletion", "Duplication", "Fragmentation", "Fusion"]
    INTRA_COLORS = ["#8B0000", "#e8a0a0", "#2ecc71", "#3498db", "#e67e22"]
    INTRA_DARK   = ["white", "#a03030", "#1a7a44", "#1a4f80", "#9f5408"]

    chroms = intra_df.index.tolist()
    total_per_chr  = intra_df.sum(axis=1)
    avg_prop       = intra_df.sum(axis=0) / intra_df.values.sum()

    # Chi-square test per (type, chromosome): observed vs expected from avg proportion
    # Expected = avg_prop[t] * total_per_chr[c]; table = [O, N-O] vs [E, N-E]
    significant = set()
    for ti, t in enumerate(INTRA_TYPES):
        for c in chroms:
            N = total_per_chr[c]
            E = avg_prop[t] * N
            if N == 0 or E < 1 or E >= N:
                continue
            O = intra_df.loc[c, t]
            try:
                _, p = chisquare([O, N - O], f_exp=[E, N - E])
                if p < 0.05:
                    significant.add((ti, c))
            except Exception:
                pass

    # Cumulative bar bottoms for star placement
    cum = np.zeros(len(chroms))
    bottoms = {}
    for ti, t in enumerate(INTRA_TYPES):
        bottoms[ti] = cum.copy()
        cum = cum + intra_df[t].values

    fig = go.Figure()
    for ti, (t, label, color) in enumerate(zip(INTRA_TYPES, INTRA_LABELS, INTRA_COLORS)):
        fig.add_trace(go.Bar(
            x=chroms, y=intra_df[t].tolist(),
            name=label,
            marker_color=color,
            marker_line_color="white",
            marker_line_width=0.4,
            textfont=dict(color="black"),
        ))

    # Overlay "*" per category at midpoint of each significant segment
    for ti, t in enumerate(INTRA_TYPES):
        sig_x, sig_y = [], []
        for ci, c in enumerate(chroms):
            if (ti, c) in significant:
                h = intra_df.loc[c, t]
                if h > 0:
                    sig_x.append(c)
                    sig_y.append(bottoms[ti][ci] + h / 2 - 2)
        if sig_x:
            fig.add_trace(go.Scatter(
                x=sig_x, y=sig_y,
                mode="text", text=["*"] * len(sig_x),
                textfont=dict(size=20, color=INTRA_DARK[ti]),
                showlegend=False, hoverinfo="skip",
            ))

    # Legend footnote for significance marker
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="asterisk", size=10, color="black", line=dict(width=2, color="black")),
        name="chi-square p < 0.05",
        showlegend=True, hoverinfo="skip",
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Intra-chromosomal Structural Variation Events per Chromosome",
                   font=dict(color="black")),
        font=dict(color="black"),
        xaxis=dict(title=dict(text="Chromosome", font=dict(color="black")),
                   tickfont=dict(size=9, color="black"),
                   tickmode="array", tickvals=chroms, ticktext=chroms),
        yaxis=dict(title=dict(text="Structural Variation Events", font=dict(color="black")),
                   tickfont=dict(size=9, color="black")),
        legend=dict(title=dict(text="SV Type", font=dict(color="black")),
                    font=dict(size=9, color="black")),
        height=460,
        margin=dict(l=60, r=20, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_chr_pyvis_net(trans_matrix, gravity, central_gravity, spring_length, spring_strength, damping):
    net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="#111111", notebook=False)
    labels = trans_matrix.columns.tolist()

    G = nx.Graph()
    G.add_nodes_from(labels)
    for i, c1 in enumerate(labels):
        for j, c2 in enumerate(labels):
            if j <= i:
                continue
            w = int(trans_matrix.loc[c1, c2])
            if w > 0:
                G.add_edge(c1, c2, weight=w)

    degrees = dict(G.degree(weight="weight"))
    min_deg = min(degrees.values()) if degrees else 0
    max_deg = max(degrees.values()) if degrees else 1
    deg_range = max_deg - min_deg or 1
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    # Interpolate #fff5f5 → #8B0000 (dark red), normalized over actual degree range
    def _node_color(ratio):
        r = int(0xFF + (0x8B - 0xFF) * ratio)
        g = int(0xF5 + (0x00 - 0xF5) * ratio)
        b = int(0xF5 + (0x00 - 0xF5) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    for node in labels:
        deg = degrees.get(node, 0)
        ratio = (deg - min_deg) / deg_range
        size = 35 + 20 * ratio
        color = _node_color(ratio)
        net.add_node(node, label=f"chr{node}", size=size,
                     title=f"chr{node}: {deg} translocations", color=color)

    for u, v in G.edges():
        w = G[u][v]["weight"]
        net.add_edge(u, v, value=w, title=f"{w} translocations",
                     width=0.5 + 4.0 * (w / max_w))

    net.set_options(f"""
    {{
      "physics": {{
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {{
          "gravitationalConstant": {gravity},
          "centralGravity": {central_gravity},
          "springLength": {spring_length},
          "springConstant": {spring_strength},
          "damping": {damping}
        }},
        "stabilization": {{"enabled": true, "iterations": 300, "fit": true}}
      }},
      "interaction": {{"hover": true, "tooltipDelay": 100, "navigationButtons": true}},
      "nodes": {{"shape": "circle", "font": {{"size": 40, "color": "#111111", "strokeWidth": 3, "strokeColor": "#ffffff"}}, "borderWidth": 2}},
      "edges": {{"smooth": {{"type": "dynamic"}}, "color": {{"inherit": "both"}}}}
    }}
    """)
    return net


def build_arabidopsis_net(nodes_df, edges_df, layout_solver, gravity, node_distance, central_gravity, spring_length, spring_strength, damping):
    nodes_df = nodes_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    node_ids = set(nodes_df["ID"])
    valid_edges = edges_df[edges_df["Source"].isin(node_ids) & edges_df["Target"].isin(node_ids)]
    connected = set(valid_edges["Source"]) | set(valid_edges["Target"])
    nodes_df = nodes_df[nodes_df["ID"].isin(connected)].reset_index(drop=True)

    kaks_mean = nodes_df["KaKs"].mean()
    kaks_max_dev = (nodes_df["KaKs"] - kaks_mean).abs().max() or 1

    # Diverging color: red (#d62728) ← white ← mean → white → blue (#1f77b4)
    def _kaks_color(kaks):
        ratio = (kaks - kaks_mean) / kaks_max_dev  # [-1, 1]
        if ratio <= 0:
            t = -ratio
            r = int(255 + (214 - 255) * t)
            g = int(255 + (39  - 255) * t)
            b = int(255 + (40  - 255) * t)
        else:
            t = ratio
            r = int(255 + (31  - 255) * t)
            g = int(255 + (119 - 255) * t)
            b = int(255 + (180 - 255) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    net = Network(height="600px", width="100%", directed=True, notebook=False,
                  bgcolor="#ffffff", font_color="#111111")

    for _, row in nodes_df.iterrows():
        nid   = row["ID"]
        kaks  = row["KaKs"]
        ng    = row["numgene"]
        size  = 8 + 4 * (ng ** 0.5)
        color = _kaks_color(kaks)
        title = f"{nid}  Ka/Ks: {kaks:.4f}"
        net.add_node(nid, label=nid, title=title, size=size, color=color)

    for _, row in valid_edges.iterrows():
        net.add_edge(row["Source"], row["Target"])

    if layout_solver == "barnesHut":
        solver_block = (
            f'"barnesHut": {{"gravitationalConstant": {gravity * 20},'
            f'"centralGravity": {central_gravity},'
            f'"springLength": {spring_length},"springConstant": {spring_strength},"damping": {damping}}}'
        )
    elif layout_solver == "repulsion":
        solver_block = (
            f'"repulsion": {{"nodeDistance": {node_distance},'
            f'"centralGravity": {central_gravity},'
            f'"springLength": {spring_length},"springConstant": {spring_strength},"damping": {damping}}}'
        )
    else:
        solver_block = (
            f'"forceAtlas2Based": {{"gravitationalConstant": {gravity},'
            f'"centralGravity": {central_gravity},'
            f'"springLength": {spring_length},"springConstant": {spring_strength},"damping": {damping}}}'
        )

    net.set_options(f"""
    {{
      "physics": {{
        "enabled": true,
        "solver": "{layout_solver}",
        {solver_block},
        "stabilization": {{"enabled": true, "iterations": 200, "fit": true}}
      }},
      "interaction": {{
        "dragNodes": true,
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      }},
      "nodes": {{
        "borderWidth": 1,
        "font": {{"size": 10, "color": "#111111"}}
      }},
      "edges": {{
        "color": {{"color": "#aaaaaa"}},
        "smooth": {{"type": "dynamic"}},
        "arrows": {{"to": {{"enabled": true, "scaleFactor": 0.4}}}}
      }}
    }}
    """)

    color_info = {
        "title": "Ka/Ks",
        "high": "#1f77b4",
        "mid":  "#ffffff",
        "low":  "#d62728",
        "max":  f"{nodes_df['KaKs'].max():.3f}",
        "min":  f"{nodes_df['KaKs'].min():.3f}",
        "mid_val": f"{kaks_mean:.3f}",
    }
    return net, color_info


def render_net_html(net, height=460, colorbar=None):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)
    if colorbar is not None:
        if "mid" in colorbar:
            gradient = f"linear-gradient(to bottom,{colorbar['high']},{colorbar['mid']},{colorbar['low']})"
            extra = f'\n  <span style="font-size:10px;color:#888">avg {colorbar["mid_val"]}</span>'
        else:
            gradient = f"linear-gradient(to bottom,{colorbar['high']},{colorbar['low']})"
            extra = ""
        inject = f"""
<div style="position:fixed;top:16px;right:16px;
            display:flex;flex-direction:column;align-items:center;gap:3px;
            z-index:9999;pointer-events:none;
            background:rgba(255,255,255,0.88);padding:7px 10px;
            border-radius:6px;border:1px solid #ddd;
            font-family:sans-serif;font-size:11px;color:#333;">
  <span style="font-weight:600;margin-bottom:2px;text-align:center;line-height:1.4">{colorbar['title']}</span>
  <span>{colorbar['max']}</span>
  <div style="width:12px;height:100px;
              background:{gradient};
              border:1px solid #bbb;border-radius:2px;"></div>
  <span>{colorbar['min']}</span>{extra}
</div>
"""
        html = html.replace("</body>", inject + "</body>")
    components.html(html, height=height, scrolling=False)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Example 1 · Chr Network Physics")
    chr_gravity = st.slider("Gravitational Constant", -200, -10, -80, step=5,
                            key="chr_gravity",
                            help="Repulsion between nodes. More negative = more spread out.")
    chr_central_gravity = st.slider("Central Gravity", 0.001, 0.100, 0.010, step=0.001, format="%.3f",
                                    key="chr_central_gravity",
                                    help="Pull toward center. Higher = tighter cluster.")
    chr_spring_length = st.slider("Spring Length (px)", 50, 300, 120, step=10,
                                  key="chr_spring_length",
                                  help="Ideal edge length in pixels.")
    chr_spring_strength = st.slider("Spring Strength", 0.01, 0.30, 0.05, step=0.01, format="%.2f",
                                    key="chr_spring_strength",
                                    help="Edge stiffness. Higher = edges pull harder.")
    chr_damping = st.slider("Damping", 0.10, 1.00, 0.50, step=0.05, format="%.2f",
                            key="chr_damping",
                            help="Velocity damping. Higher = settles faster.")
    st.divider()
    st.header("Example 2 · Network Physics")
    _LAYOUTS = {"Force Atlas 2": "forceAtlas2Based", "Barnes-Hut": "barnesHut", "Repulsion": "repulsion"}
    layout_label = st.selectbox("Layout", list(_LAYOUTS.keys()), key="ex2_layout")
    layout_solver = _LAYOUTS[layout_label]
    if layout_solver == "repulsion":
        node_distance = st.slider("Node Distance (px)", 50, 400, 120, step=10,
                                  key="node_distance",
                                  help="Minimum spacing between nodes.")
        gravity = -50
    else:
        gravity = st.slider("Gravitational Constant", -200, -10, -45, step=5,
                            key="gravity",
                            help="Repulsion between nodes. More negative = more spread out.")
        node_distance = 120
    central_gravity = st.slider("Central Gravity", 0.001, 0.100, 0.080, step=0.001, format="%.3f",
                                key="central_gravity",
                                help="Pull toward center. Higher = tighter cluster.")
    spring_length = st.slider("Spring Length (px)", 50, 300, 100, step=10,
                              key="spring_length",
                              help="Ideal edge length in pixels.")
    spring_strength = st.slider("Spring Strength", 0.01, 0.30, 0.08, step=0.01, format="%.2f",
                                key="spring_strength",
                                help="Edge stiffness. Higher = edges pull harder.")
    damping = st.slider("Damping", 0.10, 1.00, 0.40, step=0.05, format="%.2f",
                        key="damping",
                        help="Velocity damping. Higher = settles faster.")


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("Project 1 · Network Interaction")
st.markdown("This page demonstrates interactive network visualizations using [PyVis](https://pyvis.readthedocs.io/), with physics parameters adjustable via Streamlit sliders. Two examples are shown: a chromosome interaction network based on simulated cancer cell structural variations, and an *Arabidopsis* metabolic network with node colors reflecting natural selection strength.")

# ═══ Example 1 ════════════════════════════════════════════════════════════════
st.subheader("Example 1 · Comprehensive Visualization of Chromosomal Interactions")
st.markdown(
    "Cancers like [osteosarcoma](https://pubmed.ncbi.nlm.nih.gov/39814020/) exhibit complex strural variation landscapes. Here I simulated the copy-number profiles for 100 cells to demonstrate the visualization of inter-chromosomal interactions, as well as intra-chromosomal structural variation statistics." 
)

trans_matrix = get_translocation_matrix()
intra_df = get_intra_sv_counts()

st.markdown("Chromosomal translocations: genetic abnormalities where a chromosome attaches to another. The network layout emphasizes clusters of highly interacting chromosomes, while the heatmap provides precise counts.")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Chromosomal Interaction Heatmap**")
    st.caption("Heatmap shows which chromosomes are most likely to translocate with each other.")
    st.plotly_chart(plot_translocation_heatmap(trans_matrix, intra_df), use_container_width=True)

with col2:
    st.markdown("**Chromosomal Interaction Network**")
    st.caption("Network visualization makes \"hub\" chromosomes stand out more — node size and color reflect weighted degree; edge width reflects count.")
    net_chr = build_chr_pyvis_net(trans_matrix,
                                  chr_gravity, chr_central_gravity,
                                  chr_spring_length, chr_spring_strength, chr_damping)
    deg_sums = trans_matrix.sum(axis=1)
    render_net_html(net_chr, height=510, colorbar={
        "title": "Weighted<br>Degree",
        "low": "#fff5f5", "high": "#8B0000",
        "min": int(deg_sums.min()), "max": int(deg_sums.max()),
    })
    st.caption("Weighted degree = total translocations involving each chromosome.")

#st.markdown("**Intra-Chromosomal SV Events**")
st.plotly_chart(plot_sv_barchart(intra_df), use_container_width=True)
st.markdown("Each bar segment represents a type of intra-chromosomal SV event. "
            "A star indicates a significant deviation from the average proportion across chromosomes (chi-square test, p < 0.05).")
st.divider()

# ═══ Example 2 ════════════════════════════════════════════════════════════════
st.subheader("Example 2 · *Arabidopsis* 🌱 Network Visualization")
st.markdown("In this example, I recreated the metabolic network figure in **[Hao et al., *Genome Biol Evol.*, 2018](https://pubmed.ncbi.nlm.nih.gov/29617811/#&gid=article-figures&pid=scfigsc-3-uid-2)**, " \
"which was originally visualized using [Gephi](https://gephi.org/). The network approach allows us to directly observe how natural selection shapes metabolism. The network here is based on the *Arabidopsis* [AraGEM](https://pubmed.ncbi.nlm.nih.gov/20044452/) metabolic reconstruction, each node represents an enzyme and each edge represents a reaction. By calculating the strength of selection for each component, I’ve mapped the evolutionary constraints between two plant species (*A. thaliana* and *A. lyrata*) onto the network: red identifies areas of high constraint (stringent selection), while blue highlights regions where selection has relaxed. For page load efficiency, I sampled 20% of the nodes and their connected edges from the original dataset, but preserved the same layout and node coloring based on Ka/Ks values.")
st.markdown("Interestingly, a group of interconnected reactions showed relaxed selective constraints (colored in blue). Among these is R08280, an enzymatic reaction that utilizes glutathione to transform aldophosphamide.")

st.caption("Node color: **red** = Ka/Ks below average, **white** = average, **blue** = above average. Node size reflects number of genes encoding each enzyme.")

nodes_df, edges_df = load_network_data()
net2, kaks_color_info = build_arabidopsis_net(
    nodes_df, edges_df,
    layout_solver, gravity, node_distance,
    central_gravity, spring_length, spring_strength, damping,
)
render_net_html(net2, height=650, colorbar=kaks_color_info)

st.subheader("Spearman’s Correlations of Selective Constraints (Ka/Ks) and Network Statistics")
st.caption("Calculated for the full network, 1,068 nodes and 14,864 edges.")
_t2_metrics = [
    "Selection and Node Degree Spearman’s Correlation ρ", "    p-value",
    "Selection and Clustering Coefficient Spearman’s Correlation ρ", "    p-value",
    "Selection and Betweenness Centrality Spearman’s Correlation ρ", "    p-value",
]
_t2_values = ["0.0856", "0.002*", "0.1459", "0.000*", "0.0067", "0.443"]
_t2_cell_colors = [
    ["#f2f2f2" if i % 2 == 0 else "white" for i in range(len(_t2_metrics))],
    ["#d4edda" if "*" in v else ("#f2f2f2" if i % 2 == 0 else "white")
     for i, v in enumerate(_t2_values)],
]
fig_t2 = go.Figure(go.Table(
    header=dict(
        values=["Metric", "At-Al Ka/Ks"],
        fill_color="#4c72b0",
        font=dict(color="white", size=12),
        align=["left", "center"],
        height=34,
    ),
    cells=dict(
        values=[_t2_metrics, _t2_values],
        fill_color=_t2_cell_colors,
        font=dict(color="black", size=12),
        align=["left", "center"],
        height=30,
    ),
    columnwidth=[150, 150],
))
fig_t2.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=230)
st.plotly_chart(fig_t2, use_container_width=True)
st.caption("ρ = Spearman's correlation coefficient. * p < 0.05. "
           "Source: [Hao et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5887293/)")

st.markdown("""
Three topological properties of each enzyme node were correlated with its Ka/Ks value to test whether a reaction's structural role in the metabolic network predicts the intensity of natural selection acting on it.

- **Node Degree** counts the number of direct connections (reactions) a node shares with its immediate neighbors. A high-degree node participates in many biochemical transformations and is considered a local hub. The small but significant positive correlation (ρ = 0.086, p = 0.002) indicates that more-connected enzymes tend to evolve slightly faster — a counterintuitive result that may reflect **functional redundancy**: when many alternative pathways exist, any single enzyme faces relaxed purifying selection.

- **Clustering Coefficient** measures how tightly interconnected a node's neighbors are — specifically, the fraction of a node's neighbors that are also connected to each other. A high value signals that the node is embedded in a dense, modular sub-network. This metric shows the strongest and most significant correlation with Ka/Ks (ρ = 0.146, p < 0.001), suggesting that enzymes situated within cohesive metabolic modules experience more relaxed selection, likely because the module as a whole can compensate for changes in any individual member.

- **Betweenness Centrality** quantifies how frequently a node lies on the shortest paths between all other pairs of nodes in the network. Reactions with high betweenness act as critical bridges linking otherwise distant parts of metabolism. Notably, this metric shows no significant association with Ka/Ks (ρ = 0.007, p = 0.443), implying that a reaction's importance as a topological connector does not, by itself, translate into stronger purifying selection — the evolutionary constraint appears to be shaped more by local neighborhood structure than by global bridging roles.

Taken together, these results suggest that *local* network context (degree and clustering) has a detectable but modest influence on selective pressure, while *global* topological centrality does not. The findings are consistent with a model in which metabolic robustness — rather than essentiality per se — is the primary driver of relaxed selection in plant metabolism.
""")


