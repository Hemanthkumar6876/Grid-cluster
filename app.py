"""GridCluster AI - Renewable Microgrid Behavioural Segmentation (Streamlit app).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import auth
import data_gen
import pipeline as P
import report as R
import ui

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "sample_microgrid_data.csv"

st.set_page_config(page_title="GridCluster AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
S = st.session_state

PAGES = ["📊 Dashboard", "📈 Optimal K", "🎯 K-Means", "🔵 Clusters",
         "🔍 Analysis", "⚖️ Compare Sites", "✅ Validation", "🔮 New Site", "📄 Reports", "👤 Profile"]

SOLAR, WIND, HYB, BATT = "#F59E0B", "#3B82F6", "#10B981", "#22C55E"
BASE_COL = {"Solar": SOLAR, "Wind": WIND, "Hybrid": HYB}
ALT_COL = {"Solar": "#EA580C", "Wind": "#6366F1", "Hybrid": "#0D9488"}


# ============================================================================= helpers
def ccolor(name: str) -> str:
    for key in BASE_COL:
        if key in name:
            return ALT_COL[key] if any(t in name for t in ("Small", "Tier 2", "Tier 3")) else BASE_COL[key]
    return "#8B5CF6"


def cmap() -> dict[str, str]:
    return {n: ccolor(n) for n in S.res.names.values()}


def fig_style(fig: go.Figure, h: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=12, r=12, t=44, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12.5, color="#334155"),
                      legend=dict(orientation="h", y=-0.2, x=0), title=dict(font=dict(size=15, color="#0F1F44"), x=0.01))
    if any(isinstance(t, (go.Scatter3d, go.Surface, go.Mesh3d)) for t in fig.data):
        grid = "#C9D5EA"
        fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)",
                                     xaxis=dict(backgroundcolor="#EAF0FA", gridcolor=grid, showbackground=True),
                                     yaxis=dict(backgroundcolor="#E3ECF9", gridcolor=grid, showbackground=True),
                                     zaxis=dict(backgroundcolor="#F0F4FC", gridcolor=grid, showbackground=True)),
                          margin=dict(l=0, r=0, t=44, b=0))
    return fig


def bars3d(z, xl, yl, zlabel="", colorscale="Viridis") -> go.Figure:
    """True 3D bar chart from Mesh3d cubes (rows -> y, columns -> x)."""
    z = np.asarray(z, float)
    X, Y, Z, I, Ii, Jj, Kk = [], [], [], [], [], [], []
    tri = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    n, w = 0, 0.36
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            h = float(z[r, c])
            X += [c - w, c + w, c + w, c - w] * 2
            Y += [r - w, r - w, r + w, r + w] * 2
            Z += [0] * 4 + [h] * 4
            I += [h] * 8
            for p, q, s in tri:
                Ii.append(n + p)
                Jj.append(n + q)
                Kk.append(n + s)
            n += 8
    f = go.Figure(go.Mesh3d(x=X, y=Y, z=Z, i=Ii, j=Jj, k=Kk, intensity=I, colorscale=colorscale, flatshading=True,
                            lighting=dict(ambient=0.55, diffuse=0.95, specular=0.5, roughness=0.4),
                            colorbar=dict(title=zlabel, thickness=12)))
    f.update_layout(scene=dict(xaxis=dict(tickvals=list(range(len(xl))), ticktext=list(xl), title=""),
                               yaxis=dict(tickvals=list(range(len(yl))), ticktext=list(yl), title=""), zaxis=dict(title=zlabel)))
    return f


def donut(title: str = "Cluster Distribution", h: int = 320) -> go.Figure:
    res = S.res
    pr = res.profiles
    f = go.Figure(go.Pie(labels=[f"Cluster {c}: {pr.loc[c, 'name']}" for c in pr.index], values=pr["n_sites"], hole=0.64, sort=False,
                         marker=dict(colors=[ccolor(pr.loc[c, "name"]) for c in pr.index], line=dict(color="#fff", width=2)),
                         textinfo="percent", textfont=dict(color="#fff", size=13)))
    f.add_annotation(text=f"<b>{int(pr['n_sites'].sum())}</b><br><span style='font-size:11px;color:#64748B'>sites</span>", showarrow=False, font=dict(size=22, color="#0F1F44"))
    f.update_layout(title=title, legend=dict(orientation="v", x=1.0, y=0.5))
    return fig_style(f, h)


def header(title: str, sub: str) -> None:
    a, b = st.columns([3.2, 1.3])
    a.markdown(f'<div class="ptitle">{title}</div><div class="psub">{sub}</div>', unsafe_allow_html=True)
    u = S.user
    role = "Demo guest" if u.get("guest") else "Analyst"
    b.markdown(f'<div class="userchip"><div style="text-align:right"><b>{html.escape(u["name"])}</b><br><span>{role}</span></div>'
               f'<div class="av">{html.escape(u["name"][:1].upper())}</div></div>', unsafe_allow_html=True)
    st.write("")


def load_default() -> pd.DataFrame:
    if not SAMPLE.exists():
        SAMPLE.parent.mkdir(exist_ok=True)
        data_gen.generate_dataset().to_csv(SAMPLE, index=False)
    return pd.read_csv(SAMPLE)


def set_dataset(df: pd.DataFrame, source: str) -> None:
    prep = P.prepare(df)
    S.update(raw=df, source=source, prep=prep, res=P.cluster(prep, prep.recommended_k), k_choice=prep.recommended_k,
             init_choice="k-means++", report_files=None)


def ensure_data() -> None:
    if "prep" not in S:
        with st.spinner("Loading the dataset and training K-Means…"):
            set_dataset(load_default(), "sample_microgrid_data.csv")


def run_clustering(k: int, init: str = "k-means++") -> None:
    S.res = P.cluster(S.prep, int(k), init=init)
    S.k_choice = int(k)
    S.report_files = None


def run_clicked() -> None:
    run_clustering(int(S.get("k_choice") or S.res.k), S.get("init_choice", "k-means++"))


def describe(r: pd.Series) -> str:
    if r["solar_share"] >= 0.65:
        mix = "High solar generation, low wind contribution."
    elif r["wind_share"] >= 0.65:
        mix = "High wind generation, limited solar contribution."
    else:
        mix = "Balanced generation mix of solar and wind."
    return f"{mix} {r['load_level']} load; renewables cover {100 * r['renewable_ratio']:.0f}% of demand on average."


def go_public(page: str) -> None:
    S.public = page


def do_guest() -> None:
    S.user = auth.guest()


def do_logout() -> None:
    for k in list(S.keys()):
        del S[k]
    S.public = "home"


# ============================================================================= public pages
def auth_art(title: str, sub: str) -> str:
    return (f'<div class="authart"><div class="art">{ui.hero_art(640)}</div><div class="shade"></div>'
            f'<div class="txt">{ui.brand(True, 40)}<h2>{title}</h2><p>{sub}</p></div>'
            '<div class="ft"><div><span>☀️</span>Solar<br>Generation</div><div><span>🌬️</span>Wind<br>Generation</div>'
            '<div><span>⚡</span>Load<br>Analysis</div><div><span>🌤️</span>Weather<br>Data</div><div><span>🧠</span>ML<br>Clustering</div></div></div>')


def page_home() -> None:
    a, _, c, d = st.columns([3, 3.5, 1, 1])
    a.markdown(ui.brand(False, 36), unsafe_allow_html=True)
    c.button("Login", on_click=go_public, args=("login",), width="stretch", key="nav_login")
    d.button("Sign Up", on_click=go_public, args=("signup",), type="primary", width="stretch", key="nav_signup")
    st.write("")
    st.markdown(
        f'<div class="hero"><div class="art">{ui.hero_art(520)}</div><div class="shade"></div><div class="txt">'
        '<h1>Discover Hidden Patterns in Renewable Microgrids</h1>'
        '<p>Analyze generation, load and weather behavior to automatically segment microgrid sites using K-Means clustering.</p></div></div>',
        unsafe_allow_html=True)
    st.write("")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Get Started", on_click=go_public, args=("signup",), type="primary", width="stretch", key="hero_start")
    b2.button("Try the Demo", on_click=do_guest, width="stretch", key="hero_demo")
    st.write("")
    cards = [("📊", "Microgrid Analysis", "Explore generation, load and weather data for every site.", "#DBEAFE"),
             ("🎯", "K-Means Clustering", "Group similar microgrids with automatic K selection.", "#EDE9FE"),
             ("🏷️", "Site-Type Classification", "Clusters are named Solar, Wind or Hybrid from the data.", "#DCFCE7"),
             ("📈", "Behavioral Visualization", "Interactive 2D/3D charts, profiles and comparisons.", "#FEF3C7")]
    for col, (ic, t, d_, bg) in zip(st.columns(4), cards):
        col.markdown(f'<div class="card fcard"><div class="fi" style="background:{bg}">{ic}</div><h4>{t}</h4><p>{d_}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><div class="card-t">How it works</div>' + ui.flow([
        ("🗂️", "Upload data", "#DBEAFE"), ("🧹", "Clean", "#DCFCE7"), ("🧬", "Engineer features", "#FEF3C7"), ("📈", "Choose K", "#EDE9FE"),
        ("🎯", "K-Means", "#FEE2E2"), ("✅", "Validate", "#CCFBF1"), ("🔮", "Predict new sites", "#E0E7FF")]) + "</div>", unsafe_allow_html=True)
    st.caption("GridCluster AI · Renewable Microgrid Behavioral Segmentation")


def page_login() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    left.markdown(auth_art("Renewable Microgrid<br>Behavioral Segmentation",
                           "Discover behavioral patterns across renewable microgrid sites using Machine Learning."), unsafe_allow_html=True)
    with right:
        st.write("")
        with st.form("login_form"):
            st.markdown('<div class="authhead">Welcome Back</div><div class="authsub">Sign in to continue your analysis</div>', unsafe_allow_html=True)
            email = st.text_input("Email Address", placeholder="Enter your email")
            pwd = st.text_input("Password", type="password", placeholder="Enter your password")
            ok = st.form_submit_button("Sign In", type="primary", width="stretch")
        if ok:
            if time.time() < S.get("lock_until", 0):
                st.error("Too many failed attempts. Please wait a few seconds and try again.")
            else:
                user = auth.verify(email, pwd) if email and pwd else None
                if user:
                    S.user, S.fails = user, 0
                    st.rerun()
                S.fails = S.get("fails", 0) + 1
                if S.fails >= 5:
                    S.lock_until, S.fails = time.time() + 30, 0
                st.error("Invalid email or password.")
        st.button("Continue as Guest (demo, no account)", on_click=do_guest, width="stretch", key="guest_login")
        st.caption("Accounts are stored locally on this machine with salted password hashes.")
        c1, c2 = st.columns([2, 1])
        c1.markdown("<div style='padding-top:8px;color:#64748B'>Don't have an account?</div>", unsafe_allow_html=True)
        c2.button("Create Account", on_click=go_public, args=("signup",), width="stretch", key="to_signup")
        st.button("← Back to home", on_click=go_public, args=("home",), key="back_home1")


def page_signup() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    left.markdown(auth_art("Renewable Microgrid<br>Behavioral Segmentation",
                           "Discover behavioral patterns across renewable microgrid sites using Machine Learning."), unsafe_allow_html=True)
    with right:
        with st.form("signup_form"):
            st.markdown('<div class="authhead">Create Your Account</div><div class="authsub">Join us and start your analysis</div>', unsafe_allow_html=True)
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="Enter your email")
            pwd = st.text_input("Password", type="password", placeholder="At least 8 characters, letters and numbers")
            pwd2 = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            agree = st.checkbox("I understand this is a local account stored on this machine")
            ok = st.form_submit_button("Create Account", type="primary", width="stretch")
        if ok:
            if pwd != pwd2:
                st.error("Passwords do not match.")
            elif not agree:
                st.error("Please tick the checkbox to continue.")
            else:
                done, msg = auth.create_user(name, email, pwd)
                if done:
                    S.user = auth.verify(email, pwd)
                    st.rerun()
                st.error(msg)
        c1, c2 = st.columns([2, 1])
        c1.markdown("<div style='padding-top:8px;color:#64748B'>Already have an account?</div>", unsafe_allow_html=True)
        c2.button("Sign In", on_click=go_public, args=("login",), width="stretch", key="to_login")
        st.button("← Back to home", on_click=go_public, args=("home",), key="back_home2")


# ============================================================================= sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown(ui.brand(True, 34), unsafe_allow_html=True)
        st.write("")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        p, r = S.prep, S.res
        st.markdown(f'<div class="sb-ds">ACTIVE DATASET<br><b>{html.escape(S.source)}</b><br>{p.clean_report["n_sites"]} sites · '
                    f'{p.clean_report["rows_clean"]:,} rows<br>K = {r.k} · silhouette {r.metrics["silhouette"]:.2f}</div>', unsafe_allow_html=True)
        st.write("")
        st.button("↪  Logout", on_click=do_logout, width="stretch", key="logout")


# ============================================================================= DASHBOARD
def page_dashboard() -> None:
    prep, res = S.prep, S.res
    header("Dashboard", "Overview of your microgrid analysis")
    c = st.columns(4)
    c[0].markdown(ui.kpi("🏭", "Total Microgrid Sites", f"{prep.clean_report['n_sites']}", "after cleaning", "#DBEAFE"), unsafe_allow_html=True)
    c[1].markdown(ui.kpi("🧩", "Features Used", f"{len(prep.feature_cols)}", f"of {prep.features.shape[1]} engineered", "#DCFCE7"), unsafe_allow_html=True)
    c[2].markdown(ui.kpi("🎯", "Clusters", f"{res.k}", f"recommended K = {prep.recommended_k}", "#EDE9FE"), unsafe_allow_html=True)
    c[3].markdown(ui.kpi("📐", "Silhouette Score", f"{res.metrics['silhouette']:.2f}", f"Davies–Bouldin {res.metrics['davies_bouldin']:.2f}", "#FEF3C7"), unsafe_allow_html=True)
    st.write("")

    sites = list(prep.features.index)
    opts = ["All sites"] + [f"Cluster {i} – {res.names[i]}" for i in res.names] + [f"Site {s}" for s in sites]
    pick = st.selectbox("Show profiles for", opts, key="dash_scope")
    sel, label = None, "all sites"
    if pick.startswith("Cluster"):
        cl = int(pick.split()[1])
        sel, label = list(prep.features.index[res.labels == cl]), res.names[cl]
    elif pick.startswith("Site"):
        sel = [pick.split(" ", 1)[1]]
        label = sel[0]

    hp = P.hourly_profile(prep.clean, ["solar_generation", "wind_generation", "battery_output", "load", "temperature",
                                       "irradiance", "wind_speed", "humidity"], sel)
    a, b = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=hp.index, y=hp["solar_generation"], name="Solar", line=dict(color=SOLAR, width=2.6))
    f.add_scatter(x=hp.index, y=hp["wind_generation"], name="Wind", line=dict(color=WIND, width=2.6))
    f.add_scatter(x=hp.index, y=hp["battery_output"], name="Battery", line=dict(color=BATT, width=2, dash="dot"))
    f.update_layout(title=f"Generation Profile – {label}", xaxis_title="Hour of day", yaxis_title="Mean kW")
    a.plotly_chart(fig_style(f), width="stretch")
    f = go.Figure(go.Scatter(x=hp.index, y=hp["load"], name="Load", line=dict(color="#EF4444", width=2.8), fill="tozeroy", fillcolor="rgba(239,68,68,.10)"))
    f.update_layout(title=f"Load Profile – {label}", xaxis_title="Hour of day", yaxis_title="Mean kW")
    b.plotly_chart(fig_style(f), width="stretch")

    a, b = st.columns([1.6, 1])
    wc = [("temperature", "Temperature (°C)", "#EF4444"), ("irradiance", "Irradiance (W/m²)", SOLAR), ("humidity", "Humidity (%)", "#14B8A6"), ("wind_speed", "Wind speed (m/s)", WIND)]
    f = go.Figure()
    for col, nm, colr in wc:
        if col in hp.columns:
            v = hp[col]
            f.add_scatter(x=hp.index, y=(v - v.min()) / (v.max() - v.min() + 1e-9), name=nm, customdata=v, line=dict(color=colr, width=2.4),
                          hovertemplate="%{customdata:.1f}<extra>" + nm + "</extra>")
    f.update_layout(title=f"Weather Trends – {label} (each series scaled 0–1)", xaxis_title="Hour of day", yaxis_title="Relative level")
    a.plotly_chart(fig_style(f, 320), width="stretch")
    b.plotly_chart(donut(), width="stretch")

    with st.expander("🧊 3D generation landscape (site × hour)"):
        gp = prep.clean.assign(h=prep.clean.timestamp.dt.hour, g=prep.clean.solar_generation + prep.clean.wind_generation)
        piv = gp.pivot_table(index="site_id", columns="h", values="g", aggfunc="mean")
        piv = piv.reindex(prep.features.index[np.argsort(res.labels, kind="stable")])
        f = go.Figure(go.Surface(z=piv.values, x=list(piv.columns), y=list(range(len(piv))), colorscale="Viridis",
                                 contours=dict(z=dict(show=True, usecolormap=True, project_z=True)), colorbar=dict(title="kW", thickness=12)))
        step = max(1, len(piv) // 10)
        f.update_layout(scene=dict(xaxis_title="Hour of day", zaxis_title="kW",
                                   yaxis=dict(title="Site (grouped by cluster)", tickvals=list(range(0, len(piv), step)), ticktext=list(piv.index[::step])),
                                   camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9))))
        st.plotly_chart(fig_style(f, 520), width="stretch")


# ============================================================================= DATASET
def _stats_cards(raw: pd.DataFrame, n_sites: int) -> None:
    c = st.columns(4)
    c[0].metric("Rows", f"{len(raw):,}")
    c[1].metric("Microgrids", n_sites)
    c[2].metric("Columns", raw.shape[1])
    c[3].metric("Missing values", int(raw.isna().sum().sum()))


def page_dataset() -> None:
    prep = S.prep
    header("Upload Dataset", "Upload your microgrid data in CSV or Excel format")
    a, b = st.columns([1.5, 1])
    with a.container(border=True):
        up = st.file_uploader("Drag & drop your file here", type=["csv", "xlsx", "xls"], help="Supported formats: CSV, XLS, XLSX (max 200 MB)")
        c1, c2 = st.columns(2)
        if c1.button("Use sample dataset", width="stretch"):
            set_dataset(load_default(), "sample_microgrid_data.csv")
            st.rerun()
        c2.download_button("⬇ Download sample CSV", load_default().to_csv(index=False).encode(), "sample_microgrid_data.csv", "text/csv", width="stretch")
    raw, rep, src = S.raw, prep.clean_report, S.source
    if up is not None:
        try:
            raw_up = pd.read_excel(up) if up.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(up)
            _, rep_up = P.clean_data(raw_up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this file: {e}")
            return
        raw, rep, src = raw_up, rep_up, up.name
    with b:
        st.markdown('<div class="card"><div class="card-t">Dataset Information</div>'
                    f'<div class="ok">✓ {"Dataset uploaded" if up is not None else "Dataset loaded"} successfully</div>'
                    f'<div style="margin-top:12px;font-size:.9rem;color:#334155;line-height:1.9">'
                    f'<b>File:</b> {html.escape(src)}<br><b>Rows:</b> {len(raw):,}<br><b>Microgrids:</b> {rep["n_sites"]}<br>'
                    f'<b>Features (columns):</b> {raw.shape[1]}</div></div>', unsafe_allow_html=True)
    if up is not None:
        if st.button("▶ Run analysis on this dataset", type="primary"):
            try:
                with st.spinner("Cleaning, engineering features and training K-Means…"):
                    set_dataset(raw, up.name)
                st.success("Analysis complete – explore the other pages.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with st.expander("Expected columns"):
        st.markdown("**Required:** `site_id`, `timestamp`, `solar_generation`, `wind_generation`, `load`  \n"
                    "**Optional:** `battery_output`, `temperature`, `irradiance`, `wind_speed`, `humidity`, `latitude`, `longitude` (enables the map), "
                    "`site_type` (enables validation against known types). Column names are case-insensitive.")
    st.markdown('<div class="card-t" style="margin-top:14px">Preview Dataset</div>', unsafe_allow_html=True)
    st.dataframe(raw.head(200), width="stretch", height=300)


# ============================================================================= PREPROCESSING
def page_preprocessing() -> None:
    prep = S.prep
    r = prep.clean_report
    header("Data Preprocessing", "Clean and prepare the raw time-series before feature engineering")
    st.markdown('<div class="card"><div class="card-t">Data Preprocessing</div>' + ui.flow([
        ("🗂️", "Raw Data", "#DBEAFE"), ("🧹", "Cleaning", "#FEF3C7"), ("🧩", "Missing Values", "#DCFCE7"), ("🛡️", "Invalid Values", "#FEE2E2"),
        ("✅", "Clean Data", "#CCFBF1")]) + "</div>", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-t">Data Summary</div>'
                    f'<div class="bar-row"><div class="lab">Total records</div><div class="val" style="flex:1;text-align:right">{r["rows_raw"]:,}</div></div>'
                    f'<div class="bar-row"><div class="lab">Missing values</div><div class="val" style="flex:1;text-align:right">{r["missing_total"]}</div></div>'
                    f'<div class="bar-row"><div class="lab">Duplicate rows</div><div class="val" style="flex:1;text-align:right">{r["duplicates_removed"]}</div></div>'
                    f'<div class="bar-row"><div class="lab">Negative values</div><div class="val" style="flex:1;text-align:right">{r["negative_values_clipped"]}</div></div>'
                    f'<div class="bar-row"><div class="lab">Microgrid sites</div><div class="val" style="flex:1;text-align:right">{r["n_sites"]}</div></div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-t">Cleaning Steps</div>' + ui.checklist([
            "Remove duplicate site–timestamp rows", "Parse timestamps and drop invalid rows",
            "Fill missing values (time interpolation per site, then median)", "Clip negative generation/load and humidity outside 0–100%",
            "Drop sites with fewer than 24 hourly rows"]) + "</div>", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown('<div class="card-t">Before vs After</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Metric": ["Missing values", "Duplicates", "Negative values", "Records"],
                                   "Before": [r["missing_total"], r["duplicates_removed"], r["negative_values_clipped"], r["rows_raw"]],
                                   "After": [0, 0, 0, r["rows_clean"]]}), hide_index=True, width="stretch")
    with b:
        mb = pd.Series(r["missing_by_col"])
        mb = mb[mb > 0]
        if len(mb):
            f = px.bar(x=mb.index, y=mb.values, title="Missing values by column", color_discrete_sequence=["#3B82F6"])
            f.update_layout(yaxis_title="Missing values", xaxis_title="")
            st.plotly_chart(fig_style(f, 260), width="stretch")
        else:
            st.info("No missing values to show.")


# ============================================================================= FEATURES
def page_features() -> None:
    prep = S.prep
    header("Feature Engineering", "Generate behavioral features from generation, load and weather data")
    groups = ["Generation", "Load", "Weather", "Time-based"]
    cols = st.columns(4)
    for col, g in zip(cols, groups):
        feats = [c for c in prep.features.columns if P.FEATURE_INFO[c][0] == g]
        items = "".join(f'<div class="chk {"" if c in prep.feature_cols else "off"}"><i>{"✓" if c in prep.feature_cols else "–"}</i><span>{html.escape(P.FEATURE_INFO[c][1])}</span></div>' for c in feats)
        col.markdown(f'<div class="card"><div class="card-t">{g} Features</div>{items}</div>', unsafe_allow_html=True)
    st.caption("✓ used for clustering · – dropped as a near-duplicate (|r| > 0.97 with an earlier feature)")
    st.markdown('<div class="card">' + ui.flow([("🗂️", "Original Data", "#DBEAFE"), ("🧬", "Feature Engineering", "#EDE9FE"),
                                                   ("📏", "Standardization", "#FEF3C7"), ("🤖", f"{len(prep.feature_cols)} ML Features", "#DCFCE7")]) + "</div>", unsafe_allow_html=True)
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Feature table", "Standardization", "Correlation", "ML dataset"])
    with t1:
        st.dataframe(prep.features.round(3), width="stretch", height=340)
        gp = prep.clean.assign(h=prep.clean.timestamp.dt.hour, g=prep.clean.solar_generation + prep.clean.wind_generation)
        piv = gp.pivot_table(index="site_id", columns="h", values="g", aggfunc="mean")
        f = px.imshow(piv, aspect="auto", color_continuous_scale="YlGnBu", labels=dict(x="Hour", y="Site", color="kW"), title="Daily generation profile per site")
        st.plotly_chart(fig_style(f, 440), width="stretch")
    with t2:
        feat = st.selectbox("Feature", prep.feature_cols)
        i = prep.feature_cols.index(feat)
        a, b = st.columns(2)
        a.plotly_chart(fig_style(px.histogram(prep.features, x=feat, nbins=20, title="Before (raw units)", color_discrete_sequence=["#3B82F6"]), 300), width="stretch")
        b.plotly_chart(fig_style(px.histogram(x=prep.X[:, i], nbins=20, title="After StandardScaler (mean 0, sd 1)", color_discrete_sequence=["#10B981"]), 300), width="stretch")
    with t3:
        f = px.imshow(prep.features.corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_style(f, 600), width="stretch")
    with t4:
        ml = pd.DataFrame(prep.X, index=prep.features.index, columns=prep.feature_cols)
        st.dataframe(ml.round(3), width="stretch", height=380)
        st.download_button("⬇ Download ML dataset (CSV)", ml.to_csv().encode(), "ml_dataset.csv", "text/csv")


# ============================================================================= OPTIMAL K
def apply_k(k: int) -> None:
    run_clustering(k)


def page_optimal_k() -> None:
    prep, res = S.prep, S.res
    header("Optimal K", "Use the Elbow method and Silhouette score to find the best number of clusters")
    kt = prep.k_table
    a, b = st.columns(2)
    f = go.Figure(go.Scatter(x=kt.k, y=kt.inertia, mode="lines+markers", line=dict(color="#2563EB", width=3), marker=dict(size=8)))
    f.add_vline(x=prep.elbow_k, line_dash="dash", line_color="#EF4444", annotation_text=f"Elbow K = {prep.elbow_k}")
    f.update_layout(title="Elbow Method (inertia)", xaxis_title="K", yaxis_title="Inertia")
    a.plotly_chart(fig_style(f, 340), width="stretch")
    f = go.Figure(go.Scatter(x=kt.k, y=kt.silhouette, mode="lines+markers", line=dict(color="#10B981", width=3), marker=dict(size=8)))
    f.add_vline(x=prep.best_sil_k, line_dash="dash", line_color="#EF4444", annotation_text=f"Best K = {prep.best_sil_k}")
    f.update_layout(title="Silhouette Score vs K", xaxis_title="K", yaxis_title="Silhouette")
    b.plotly_chart(fig_style(f, 340), width="stretch")
    st.write("")
    a, b = st.columns([2.2, 1])
    with a:
        st.markdown(f'<div class="card"><div class="card-t">Recommended K</div><div style="display:flex;align-items:center;gap:16px">'
                    f'<div style="width:64px;height:64px;border-radius:50%;background:#2563EB;color:#fff;font-weight:800;font-size:1.9rem;display:flex;align-items:center;justify-content:center">{prep.recommended_k}</div>'
                    f'<div style="color:#475569;font-size:.92rem;line-height:1.55">The elbow method points to <b>K = {prep.elbow_k}</b>; the silhouette score peaks at '
                    f'<b>K = {prep.best_sil_k}</b> ({kt.set_index("k").silhouette.max():.2f}). The elbow value is preferred unless its silhouette is below 85% of the best.</div></div></div>', unsafe_allow_html=True)
    with b:
        st.write("")
        st.button(f"Apply K = {prep.recommended_k}", on_click=apply_k, args=(prep.recommended_k,), type="primary", width="stretch", key="apply_k")
        if res.k == prep.recommended_k:
            st.caption("✓ Currently applied")
    with st.expander("Davies–Bouldin and Calinski–Harabasz"):
        a, b = st.columns(2)
        a.plotly_chart(fig_style(px.line(kt, x="k", y="davies_bouldin", markers=True, title="Davies–Bouldin (lower is better)"), 280), width="stretch")
        b.plotly_chart(fig_style(px.line(kt, x="k", y="calinski_harabasz", markers=True, title="Calinski–Harabasz (higher is better)"), 280), width="stretch")
        st.dataframe(kt.round(3), hide_index=True, width="stretch")


# ============================================================================= K-MEANS
def page_kmeans() -> None:
    prep, res = S.prep, S.res
    header("K-Means Clustering", "Cluster your microgrid sites using the K-Means algorithm")
    kmax = int(prep.k_table["k"].max())
    a, b = st.columns([1, 1.3])
    with a.container(border=True):
        st.markdown('<div class="card-t">Settings</div>', unsafe_allow_html=True)
        st.segmented_control("Number of clusters (K)", list(range(2, min(6, kmax) + 1)), key="k_choice", selection_mode="single")
        init = st.selectbox("Initialization", ["k-means++", "random"], key="init_choice")
        k = S.get("k_choice") or res.k
        st.button("▶ Run Clustering", type="primary", width="stretch", key="run_km", on_click=run_clicked)
        st.caption(f"Recommended K = {prep.recommended_k}")
    with b:
        st.markdown('<div class="card-t">Clustering Results</div>', unsafe_allow_html=True)
        st.success("K-Means completed ✓")
        c = st.columns(4)
        c[0].metric("Clusters", res.k)
        c[1].metric("Iterations", res.n_iter)
        c[2].metric("Inertia", f"{res.inertia:.2f}")
        c[3].metric("Silhouette", f"{res.metrics['silhouette']:.2f}")
        if k != res.k:
            st.warning(f"K = {k} is selected but the results shown are for K = {res.k}. Click **Run Clustering**.")
    st.write("")
    st.markdown('<div class="card-t">Cluster Assignments</div>', unsafe_allow_html=True)
    asg = res.assignments.copy()
    q = st.text_input("Filter by site id or cluster name", placeholder="e.g. MG001 or Wind")
    if q:
        asg = asg[asg.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(asg, width="stretch", height=340, hide_index=True)


# ============================================================================= CLUSTERS (visualization)
def page_clusters() -> None:
    prep, res = S.prep, S.res
    COL = cmap()
    header("Cluster Visualization", "Each point represents a microgrid site projected with PCA")
    view = st.radio("View", ["2D PCA", "3D PCA"], horizontal=True, label_visibility="collapsed")
    names = [res.names[i] for i in res.labels]
    a, b = st.columns([2.1, 1])
    if view == "2D PCA":
        ev = prep.pca.explained_variance_ratio_
        df = pd.DataFrame(prep.coords, columns=["PC1", "PC2"], index=prep.features.index)
        df["Cluster"] = names
        df["Site"] = df.index
        df["Solar kW"] = prep.features["avg_solar"].round(1)
        df["Wind kW"] = prep.features["avg_wind"].round(1)
        df["Load kW"] = prep.features["avg_load"].round(1)
        known = prep.meta["site_type"].reindex(df.index) if "site_type" in prep.meta else None
        sym = None
        if known is not None:
            df["Known type"] = known
            sym = "Known type" if st.toggle("Marker shape = known site type", value=False) else None
        f = px.scatter(df, x="PC1", y="PC2", color="Cluster", color_discrete_map=COL, symbol=sym, hover_name="Site",
                       hover_data={"PC1": False, "PC2": False, "Solar kW": True, "Wind kW": True, "Load kW": True}, title="PCA projection (2D)")
        f.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        cen = prep.pca.transform(res.centers)
        f.add_scatter(x=cen[:, 0], y=cen[:, 1], mode="markers+text", text=[f"C{i}" for i in range(res.k)], textposition="top center",
                      marker=dict(symbol="x", size=15, color="#0F1F44"), name="Centroids")
        f.update_layout(xaxis_title=f"PC1 ({100 * ev[0]:.0f}% variance)", yaxis_title=f"PC2 ({100 * ev[1]:.0f}% variance)")
        a.plotly_chart(fig_style(f, 520), width="stretch")
    else:
        from sklearn.decomposition import PCA
        p3 = PCA(3, random_state=P.SEED).fit(prep.X)
        c3, cen3 = p3.transform(prep.X), p3.transform(res.centers)
        hulls = st.toggle("Show translucent cluster hulls", value=True)
        f = go.Figure()
        for cid, nm in res.names.items():
            m = res.labels == cid
            col = ccolor(nm)
            f.add_trace(go.Scatter3d(x=c3[m, 0], y=c3[m, 1], z=c3[m, 2], mode="markers", name=f"C{cid} {nm}", text=prep.features.index[m],
                                     hovertemplate="%{text}<extra>" + nm + "</extra>", marker=dict(size=7, color=col, line=dict(width=1.5, color="white"))))
            if hulls and m.sum() >= 4:
                f.add_trace(go.Mesh3d(x=c3[m, 0], y=c3[m, 1], z=c3[m, 2], alphahull=0, color=col, opacity=0.14, showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter3d(x=cen3[:, 0], y=cen3[:, 1], z=cen3[:, 2], mode="markers+text", name="Centroids", text=[f"C{i}" for i in range(res.k)],
                                 textfont=dict(color="#0F1F44", size=14), marker=dict(size=10, symbol="diamond", color="#0F1F44")))
        ev3 = p3.explained_variance_ratio_
        f.update_layout(title="PCA projection (3D) – drag to rotate",
                        scene=dict(xaxis_title=f"PC1 ({100 * ev3[0]:.0f}%)", yaxis_title=f"PC2 ({100 * ev3[1]:.0f}%)", zaxis_title=f"PC3 ({100 * ev3[2]:.0f}%)",
                                   camera=dict(eye=dict(x=1.4, y=1.25, z=0.85))))
        a.plotly_chart(fig_style(f, 560), width="stretch")
    b.plotly_chart(donut(h=340), width="stretch")
    pr = res.profiles
    b.markdown('<div class="card">' + "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:.88rem;margin:5px 0"><span><span style="color:{ccolor(pr.loc[c, "name"])}">●</span> Cluster {c}</span>'
        f'<b>{int(pr.loc[c, "n_sites"])} sites</b></div>' for c in pr.index) + "</div>", unsafe_allow_html=True)


# ============================================================================= ANALYSIS
def page_analysis() -> None:
    prep, res = S.prep, S.res
    header("Cluster Analysis", "Behavioral patterns and characteristics of each cluster")
    pr = res.profiles
    mx = prep.features["avg_load"].max()
    cols = st.columns(min(res.k, 3))
    for n, (cid, r) in enumerate(pr.iterrows()):
        cols[n % len(cols)].markdown(
            f'<div class="card" style="margin-bottom:14px"><div class="ctitle" style="color:{ccolor(r["name"])}">Cluster {cid} – {r["name"]}</div>'
            f'{ui.pill(str(int(r["n_sites"])) + " sites")}<div style="height:6px"></div>'
            + ui.bar("Solar Generation", r["solar_share"], SOLAR) + ui.bar("Wind Generation", r["wind_share"], WIND)
            + ui.bar("Battery", min(1, r["battery_share"] * 5), BATT, f"{100 * r['battery_share']:.1f}%")
            + ui.bar("Average Load", r["avg_load"] / mx, "#EF4444", f"{r['avg_load']:.0f} kW")
            + f'<div class="desc">{describe(r)}</div></div>', unsafe_allow_html=True)
    st.caption("Names are generated automatically from each cluster's mean solar/wind share and battery use – nothing is hard-coded.")
    t1, t2, t3, t4 = st.tabs(["Radar", "Centroid heatmap", "Profile table", "Site map"])
    with t1:
        feats = [c for c in ["solar_share", "wind_share", "renewable_ratio", "avg_load", "load_variability", "avg_wind_speed", "avg_irradiance", "battery_share"] if c in prep.features]
        mm = (pr[feats] - prep.features[feats].min()) / (prep.features[feats].max() - prep.features[feats].min() + 1e-9)
        f = go.Figure()
        for cid, row in mm.iterrows():
            f.add_scatterpolar(r=list(row.values) + [row.values[0]], theta=feats + [feats[0]], name=pr.loc[cid, "name"], fill="toself", opacity=0.45,
                               line=dict(color=ccolor(pr.loc[cid, "name"])))
        f.update_layout(title="Cluster fingerprints (min–max scaled)")
        st.plotly_chart(fig_style(f, 460), width="stretch")
    with t2:
        z = pd.DataFrame(res.centers, columns=prep.feature_cols, index=[f"C{i} {res.names[i]}" for i in range(res.k)])
        st.plotly_chart(fig_style(px.imshow(z, color_continuous_scale="RdBu_r", zmin=-2.5, zmax=2.5, aspect="auto", labels=dict(color="z-score")), 150 + 70 * res.k), width="stretch")
        f3 = bars3d(res.centers, prep.feature_cols, [f"C{i}" for i in range(res.k)], "z-score", "Viridis")
        f3.update_layout(title="3D centroid bars", scene=dict(camera=dict(eye=dict(x=1.5, y=-1.9, z=1.0))))
        st.plotly_chart(fig_style(f3, 540), width="stretch")
    with t3:
        st.dataframe(pr.drop(columns=["solar_var_level", "load_level"]).round(3), width="stretch")
    with t4:
        if {"latitude", "longitude"} <= set(prep.meta.columns) and prep.meta[["latitude", "longitude"]].notna().all(axis=1).any():
            m = prep.meta.copy()
            m["Cluster"] = [res.names[i] for i in res.labels]
            m["site"] = m.index
            m = m.dropna(subset=["latitude", "longitude"])
            if st.toggle("Show basemap tiles (needs internet access)", value=True):
                f = px.scatter_map(m, lat="latitude", lon="longitude", color="Cluster", color_discrete_map=cmap(), hover_name="site", zoom=4, map_style="carto-positron")
                f.update_traces(marker=dict(size=14))
            else:
                f = px.scatter(m, x="longitude", y="latitude", color="Cluster", color_discrete_map=cmap(), hover_name="site")
                f.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
                f.update_yaxes(scaleanchor="x")
            st.plotly_chart(fig_style(f, 540), width="stretch")
        else:
            st.info("Add `latitude` and `longitude` columns to your dataset to see sites on a map.")


# ============================================================================= COMPARE
def page_compare() -> None:
    prep, res = S.prep, S.res
    header("Site Comparison", "Compare two microgrid sites and analyze their similarity")
    sites = list(prep.features.index)
    a, b = st.columns(2)
    sa = a.selectbox("Site A", sites, index=0)
    sb = b.selectbox("Site B", sites, index=min(len(sites) - 1, len(sites) // 2 + 5))
    fa, fb = prep.features.loc[sa], prep.features.loc[sb]
    ca, cb = int(res.labels[sites.index(sa)]), int(res.labels[sites.index(sb)])
    sim = P.site_similarity(prep, sa, sb)
    mx = prep.features["avg_load"].max()

    def card(s, f_, c):
        return (f'<div class="card"><div class="ctitle" style="color:{ccolor(res.names[c])}">Site {s}</div>{ui.pill(f"Cluster {c} · {res.names[c]}")}<div style="height:6px"></div>'
                + ui.bar("Solar", f_["solar_share"], SOLAR) + ui.bar("Wind", f_["wind_share"], WIND)
                + ui.bar("Load", f_["avg_load"] / mx, "#EF4444", f"{f_['avg_load']:.1f} kW")
                + (ui.bar("Temp", min(1, f_["avg_temperature"] / 45), "#F97316", f"{f_['avg_temperature']:.1f} °C") if "avg_temperature" in f_ else "") + "</div>")
    c1, c2, c3 = st.columns([1.2, 1.3, 1.2])
    c1.markdown(card(sa, fa, ca), unsafe_allow_html=True)
    feats = [c for c in ["solar_share", "wind_share", "renewable_ratio", "avg_load", "load_variability", "avg_wind_speed", "avg_irradiance", "battery_share", "morning_gen", "evening_load"] if c in prep.features]
    F = prep.features[feats]
    mm = (F - F.min()) / (F.max() - F.min() + 1e-9)
    f = go.Figure()
    for s, colr in ((sa, SOLAR), (sb, WIND)):
        r = mm.loc[s]
        f.add_scatterpolar(r=list(r.values) + [r.values[0]], theta=feats + [feats[0]], name=s, fill="toself", opacity=0.5, line=dict(color=colr))
    f.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
    c2.plotly_chart(fig_style(f, 330), width="stretch")
    c3.markdown(card(sb, fb, cb), unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card"><div class="card-t">Behavioral Similarity</div>{ui.bar("", sim / 100, "#2563EB", f"{sim:.0f}%")}'
                '<div class="desc">100% = identical feature vectors; 0% = the most different pair of sites in this dataset.</div></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Hourly profiles", "All features"])
    with t1:
        f = go.Figure()
        for s, colr in ((sa, SOLAR), (sb, WIND)):
            h = P.hourly_profile(prep.clean, ["solar_generation", "wind_generation", "load"], [s])
            f.add_scatter(x=h.index, y=h.solar_generation + h.wind_generation, name=f"{s} generation", line=dict(color=colr, width=3))
            f.add_scatter(x=h.index, y=h.load, name=f"{s} load", line=dict(color=colr, dash="dot"))
        f.update_layout(xaxis_title="Hour of day", yaxis_title="Mean kW")
        st.plotly_chart(fig_style(f, 380), width="stretch")
    with t2:
        cmp_ = pd.DataFrame({sa: fa, sb: fb})
        cmp_["difference"] = cmp_[sb] - cmp_[sa]
        st.dataframe(cmp_.round(3), width="stretch", height=480)


# ============================================================================= VALIDATION
def page_validation() -> None:
    prep, res = S.prep, S.res
    header("Validation", "Evaluate cluster quality and compare with known site types")
    m = res.metrics
    c = st.columns(4)
    sil = m["silhouette"]
    badge = ui.pill("Strong", "green") if sil >= 0.5 else (ui.pill("Acceptable", "amber") if sil >= 0.25 else ui.pill("Weak", "red"))
    c[0].markdown(ui.kpi("📐", "Silhouette Score", f"{sil:.2f}", ">0.5 strong · >0.25 acceptable", "#DBEAFE") + badge, unsafe_allow_html=True)
    c[1].markdown(ui.kpi("📉", "Davies–Bouldin Index", f"{m['davies_bouldin']:.2f}", "lower is better", "#DCFCE7"), unsafe_allow_html=True)
    c[2].markdown(ui.kpi("📊", "Calinski–Harabasz", f"{m['calinski_harabasz']:.1f}", "higher is better", "#FEF3C7"), unsafe_allow_html=True)
    c[3].markdown(ui.kpi("🧮", "Inertia", f"{m['inertia']:.1f}", "within-cluster SS", "#EDE9FE"), unsafe_allow_html=True)
    st.write("")
    v = res.validation
    if not v:
        st.info("External validation needs a `site_type` column (known Solar / Wind / Hybrid labels) with at least two types.")
        return
    a, b = st.columns([1.5, 1])
    with a:
        st.markdown('<div class="card-t">Known vs Discovered Types</div>', unsafe_allow_html=True)
        t = v["table"].copy()
        t["Result"] = np.where(t["match"], "✓", "✗")
        t = t.rename(columns={"site_id": "Site", "known_type": "Known Type", "cluster": "Cluster", "cluster_majority_type": "Cluster majority type"})
        if st.toggle("Show mismatches only"):
            t = t[~t["match"]]
        st.dataframe(t[["Site", "Known Type", "Cluster", "cluster_name", "Cluster majority type", "Result"]], hide_index=True, width="stretch", height=380)
    with b:
        pur = v["purity"]
        f = go.Figure(go.Pie(values=[pur, 1 - pur], hole=0.74, sort=False, direction="clockwise", textinfo="none", marker=dict(colors=["#2563EB", "#E2E8F0"])))
        f.add_annotation(text=f"<b>{100 * pur:.0f}%</b>", showarrow=False, font=dict(size=34, color="#0F1F44"))
        f.update_layout(title="Cluster Purity", showlegend=False)
        st.plotly_chart(fig_style(f, 280), width="stretch")
        st.markdown(f'<div class="card">{ui.bar("Agreement", v["agreement"], "#10B981")}{ui.bar("Adjusted Rand", max(v["ari"], 0), "#6366F1", f"{v["ari"]:.2f}")}'
                    f'{ui.bar("Norm. Mutual Info", v["nmi"], "#F59E0B", f"{v["nmi"]:.2f}")}<div class="desc">{int(v["table"]["match"].sum())} of {len(v["table"])} sites fall in the cluster that matches their known type.</div></div>', unsafe_allow_html=True)
    st.write("")
    ct = v["crosstab"].copy()
    ct.index = [f"C{i} {res.names[i]}" for i in ct.index]
    if st.toggle("3D confusion bars", value=False):
        f = bars3d(ct.to_numpy(), list(ct.columns), [s.split(" ", 1)[0] for s in ct.index], "Sites", "Viridis")
        f.update_layout(scene=dict(xaxis_title="Known type", yaxis_title="Cluster", camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
    else:
        f = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", aspect="auto", labels=dict(x="Known type", y="Cluster", color="Sites"))
    f.update_layout(title="Confusion matrix: known type vs cluster")
    st.plotly_chart(fig_style(f, 380), width="stretch")


# ============================================================================= NEW SITE
def page_newsite() -> None:
    prep, res = S.prep, S.res
    header("New Microgrid Analysis", "Enter new microgrid data to predict its behavioral cluster")
    t1, t2 = st.tabs(["Enter site values", "Upload site time-series"])
    with t1:
        a, b = st.columns([1.2, 1])
        with a:
            with st.form("new_site"):
                c1, c2 = st.columns(2)
                solar = c1.number_input("Solar generation (kWh/day)", 0.0, 100000.0, 850.0, 10.0)
                wind = c2.number_input("Wind generation (kWh/day)", 0.0, 100000.0, 120.0, 10.0)
                load = c1.number_input("Average load (kWh/day)", 1.0, 100000.0, 600.0, 10.0)
                temp = c2.number_input("Average temperature (°C)", -20.0, 55.0, 31.0, 0.5)
                irr = c1.number_input("Peak (midday) irradiance (W/m²)", 0.0, 1500.0, 720.0, 10.0)
                ws = c2.number_input("Average wind speed (m/s)", 0.0, 30.0, 4.2, 0.1)
                go_ = st.form_submit_button("Analyze Site", type="primary", width="stretch")
        with b:
            if go_:
                o = P.predict_new_site(prep, res, solar, wind, load, irr / np.pi, ws, temp)
                c = o["cluster"]
                st.markdown(f'<div class="big"><div class="t">Predicted Cluster</div><div class="n">{o["name"]}</div>'
                            f'<div class="sub">Cluster {c} · similarity {o["similarity"][c]:.0f}%</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown('<div class="card">' + ui.bar("Solar Dominance", o["solar_share"], SOLAR) + ui.bar("Wind Contribution", o["wind_share"], WIND)
                            + ui.bar("Renewable / Load", min(1, o["renewable_ratio"] / 2), "#8B5CF6", f"{o['renewable_ratio']:.2f}×")
                            + f'<div class="bar-row"><div class="lab">Load Pattern</div><b>{o["load_level"]}</b></div></div>', unsafe_allow_html=True)
                st.session_state["_last_pred"] = o
            else:
                st.markdown('<div class="card"><div class="card-t">Result</div><p>Fill in the site values and click <b>Analyze Site</b> to see the predicted cluster.</p></div>', unsafe_allow_html=True)
        o = S.get("_last_pred")
        if go_ and o:
            d = pd.DataFrame({"Cluster": [f"C{i} {res.names[i]}" for i in range(res.k)], "Similarity %": o["similarity"].round(1)})
            f = px.bar(d, x="Similarity %", y="Cluster", orientation="h", color="Cluster", color_discrete_map={f"C{i} {res.names[i]}": ccolor(res.names[i]) for i in range(res.k)}, range_x=[0, 100])
            f.update_layout(showlegend=False, title="Similarity to every cluster")
            st.plotly_chart(fig_style(f, 260), width="stretch")
            st.caption(f"Distance to each centroid uses the {o['n_dims']} features that can be derived from these inputs (standardized). Irradiance is converted to a daily mean (peak ÷ π).")
    with t2:
        st.write("Upload hourly data for one or more **new** sites (same columns as the training file).")
        up = st.file_uploader("Site data (CSV / Excel)", type=["csv", "xlsx"], key="newsite")
        if st.button("Try with 3 example unseen sites"):
            ex = data_gen.generate_dataset(1, 1, 1, days=14, seed=7, n_duplicates=0)
            ex["site_id"] = ex["site_id"].map({"MG001": "NEW-A", "MG002": "NEW-B", "MG003": "NEW-C"})
            S.example_new = ex
        df = None
        if up is not None:
            df = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
        elif "example_new" in S:
            df = S.example_new
        if df is not None:
            try:
                st.dataframe(P.predict_from_timeseries(prep, res, df), width="stretch", hide_index=True)
                if "site_type" in df.columns:
                    st.caption("Known types in the example: " + ", ".join(f"{s}={t}" for s, t in df.groupby("site_id")["site_type"].first().items()))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))


# ============================================================================= REPORTS
def page_reports() -> None:
    prep, res = S.prep, S.res
    header("Reports", "Generate and download comprehensive analysis reports")
    st.markdown("""<style>
    .st-key-dl_pdf button{background:#EF4444 !important;color:#fff !important;border:0 !important;}
    .st-key-dl_csv button{background:#10B981 !important;color:#fff !important;border:0 !important;}
    .st-key-dl_xlsx button{background:#2563EB !important;color:#fff !important;border:0 !important;}
    </style>""", unsafe_allow_html=True)
    if not S.get("report_files"):
        with st.spinner("Building CSV, Excel and PDF…"):
            S.report_files = {"csv": R.to_csv(prep, res), "xlsx": R.to_excel(prep, res), "pdf": R.to_pdf(prep, res)}
    files = S.report_files
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown('<div class="card"><div class="card-t">Report Sections</div>' + ui.checklist([
            "Dataset summary", "Feature statistics (Excel)", "Optimal K (elbow & silhouette)", "Cluster profiles", "Cluster assignments",
            "Validation metrics", "Visualizations (PDF)", "Conclusions"]) + "</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card-t">Export Report</div>', unsafe_allow_html=True)
        st.download_button("⬇ Download PDF", files["pdf"], "microgrid_report.pdf", "application/pdf", width="stretch", key="dl_pdf")
        st.download_button("⬇ Download CSV (assignments)", files["csv"], "microgrid_clusters.csv", "text/csv", width="stretch", key="dl_csv")
        st.download_button("⬇ Download Excel (full report)", files["xlsx"], "microgrid_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", key="dl_xlsx")
    st.write("")
    with st.expander("Dataset summary", expanded=True):
        st.dataframe(R._summary_df(prep, res), hide_index=True, width="stretch")
    with st.expander("Conclusions", expanded=True):
        for line in P.conclusions(prep, res):
            st.markdown(f"- {line}")


# ============================================================================= PROFILE
def page_profile() -> None:
    u = S.user
    header("Profile", "Manage your account")
    a, b = st.columns([1, 1.6])
    with a:
        st.markdown(f'<div class="card" style="text-align:center"><div class="userchip" style="justify-content:center"><div class="av" style="width:84px;height:84px;font-size:2.2rem">{html.escape(u["name"][:1].upper())}</div></div>'
                    f'<div class="ctitle" style="margin-top:12px;font-size:1.2rem">{html.escape(u["name"])}</div><div class="desc">{html.escape(u["email"])}</div>'
                    f'<div style="margin-top:10px">{ui.pill("Member since " + u.get("created", "today"))}</div></div>', unsafe_allow_html=True)
        st.write("")
        st.button("↪  Logout", on_click=do_logout, width="stretch", key="logout_profile")
    with b:
        if u.get("guest"):
            st.info("You are using the demo as a guest. Create an account to save a profile.")
            st.button("Create an account", on_click=lambda: (do_logout(), go_public("signup")), key="guest_to_signup")
        else:
            with st.form("name_form"):
                st.markdown('<div class="card-t">Account Settings</div>', unsafe_allow_html=True)
                nm = st.text_input("Display name", value=u["name"])
                if st.form_submit_button("Update name"):
                    if auth.update_name(u["email"], nm):
                        S.user = {**u, "name": nm.strip()}
                        st.rerun()
                    st.error("Please enter at least 2 characters.")
            with st.form("pwd_form"):
                st.markdown('<div class="card-t">Change Password</div>', unsafe_allow_html=True)
                old = st.text_input("Current password", type="password")
                new = st.text_input("New password", type="password")
                if st.form_submit_button("Change password"):
                    done, msg = auth.change_password(u["email"], old, new)
                    (st.success if done else st.error)(msg)


PAGE_FUNCS = dict(zip(PAGES, [page_dashboard, page_optimal_k, page_kmeans, page_clusters,
                              page_analysis, page_compare, page_validation, page_newsite, page_reports, page_profile]))

# ============================================================================= router
S.setdefault("user", None)
S.setdefault("public", "home")
if S.user is None:
    ui.inject(public=True)
    {"home": page_home, "login": page_login, "signup": page_signup}[S.public]()
else:
    ui.inject()
    S.setdefault("page", PAGES[0])
    if S.page not in PAGES:
        S.page = PAGES[0]
    ensure_data()
    sidebar()
    PAGE_FUNCS[S.page]()
