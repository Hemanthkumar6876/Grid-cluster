"""ML pipeline for Renewable Microgrid Behavioural Segmentation.

Data -> cleaning -> feature engineering -> standardisation -> K selection ->
K-Means -> automatic cluster naming/profiling -> validation -> prediction.

Everything here is UI-independent so it can be unit-tested and reused from
``train.py`` and ``app.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
REQUIRED = ["site_id", "timestamp", "solar_generation", "wind_generation", "load"]
NUMERIC = [
    "solar_generation", "wind_generation", "battery_output", "load",
    "temperature", "irradiance", "wind_speed", "humidity",
]
EPS = 1e-3

# feature -> (group, human description)
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "avg_solar": ("Generation", "Average solar generation (kW)"),
    "max_solar": ("Generation", "Maximum solar generation (kW)"),
    "solar_variability": ("Generation", "Solar variability (coeff. of variation)"),
    "avg_wind": ("Generation", "Average wind generation (kW)"),
    "max_wind": ("Generation", "Maximum wind generation (kW)"),
    "wind_variability": ("Generation", "Wind variability (coeff. of variation)"),
    "solar_share": ("Generation", "Solar share of total generation"),
    "wind_share": ("Generation", "Wind share of total generation"),
    "renewable_ratio": ("Generation", "Renewable contribution (generation / load)"),
    "battery_share": ("Generation", "Battery discharge as share of load"),
    "avg_load": ("Load", "Average load (kW)"),
    "peak_load": ("Load", "Peak load (kW)"),
    "load_variability": ("Load", "Load variability (coeff. of variation)"),
    "peak_to_avg": ("Load", "Peak-to-average load ratio"),
    "avg_temperature": ("Weather", "Average temperature (°C)"),
    "avg_irradiance": ("Weather", "Average irradiance (W/m²)"),
    "avg_wind_speed": ("Weather", "Average wind speed (m/s)"),
    "avg_humidity": ("Weather", "Average humidity (%)"),
    "morning_gen": ("Time-based", "Mean generation 06–12 h (kW)"),
    "afternoon_gen": ("Time-based", "Mean generation 12–18 h (kW)"),
    "evening_load": ("Time-based", "Mean load 18–23 h (kW)"),
}


# --------------------------------------------------------------------------- #
# 1. Cleaning
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")

    rep: dict = {"rows_raw": int(len(df))}
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    rep["bad_rows_removed"] = int(df["timestamp"].isna().sum() + df["site_id"].isna().sum())
    df = df.dropna(subset=["timestamp", "site_id"]).copy()
    df["site_id"] = df["site_id"].astype(str).str.strip()

    if "battery_output" not in df.columns:
        df["battery_output"] = 0.0
    num = [c for c in NUMERIC if c in df.columns]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.drop_duplicates(subset=["site_id", "timestamp"], keep="first")
    rep["duplicates_removed"] = int(n0 - len(df))

    rep["missing_by_col"] = {c: int(df[c].isna().sum()) for c in num}
    rep["missing_total"] = int(sum(rep["missing_by_col"].values()))

    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    for c in num:
        df[c] = df.groupby("site_id")[c].transform(lambda s: s.interpolate(limit_direction="both"))
        med = df[c].median()
        df[c] = df[c].fillna(0.0 if pd.isna(med) else med)

    neg = 0
    for c in ["solar_generation", "wind_generation", "load", "irradiance", "wind_speed"]:
        if c in df.columns:
            neg += int((df[c] < 0).sum())
            df[c] = df[c].clip(lower=0)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0, 100)
    rep["negative_values_clipped"] = neg

    counts = df.groupby("site_id").size()
    short = counts[counts < 24].index
    rep["sites_dropped_too_short"] = int(len(short))
    df = df[~df["site_id"].isin(short)].reset_index(drop=True)

    rep["rows_clean"] = int(len(df))
    rep["n_sites"] = int(df["site_id"].nunique())
    return df, rep


def site_meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["site_type", "latitude", "longitude"] if c in df.columns]
    if not cols:
        return pd.DataFrame(index=pd.Index(sorted(df["site_id"].unique()), name="site_id"))
    m = df.groupby("site_id")[cols].first()
    if "site_type" in m.columns:
        m["site_type"] = m["site_type"].astype("object").where(m["site_type"].notna(), None)
        m["site_type"] = m["site_type"].map(lambda x: str(x).strip().title() if x is not None else None)
    for c in ("latitude", "longitude"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["gen"] = d["solar_generation"] + d["wind_generation"]
    d["hour"] = d["timestamp"].dt.hour
    d["batt_dis"] = d["battery_output"].clip(lower=0)
    g = d.groupby("site_id")
    f = pd.DataFrame(index=pd.Index(sorted(d["site_id"].unique()), name="site_id"))

    f["avg_solar"] = g["solar_generation"].mean()
    f["max_solar"] = g["solar_generation"].max()
    f["solar_variability"] = (g["solar_generation"].std() / (f["avg_solar"] + EPS)).clip(upper=10)
    f["avg_wind"] = g["wind_generation"].mean()
    f["max_wind"] = g["wind_generation"].max()
    f["wind_variability"] = (g["wind_generation"].std() / (f["avg_wind"] + EPS)).clip(upper=10)
    tot = f["avg_solar"] + f["avg_wind"] + EPS
    f["solar_share"] = f["avg_solar"] / tot
    f["wind_share"] = f["avg_wind"] / tot

    f["avg_load"] = g["load"].mean()
    f["peak_load"] = g["load"].max()
    f["load_variability"] = g["load"].std() / (f["avg_load"] + EPS)
    f["peak_to_avg"] = f["peak_load"] / (f["avg_load"] + EPS)
    f["renewable_ratio"] = (f["avg_solar"] + f["avg_wind"]) / (f["avg_load"] + EPS)
    f["battery_share"] = g["batt_dis"].mean() / (f["avg_load"] + EPS)

    if "temperature" in d:
        f["avg_temperature"] = g["temperature"].mean()
    if "irradiance" in d:
        f["avg_irradiance"] = g["irradiance"].mean()
    if "wind_speed" in d:
        f["avg_wind_speed"] = g["wind_speed"].mean()
    if "humidity" in d:
        f["avg_humidity"] = g["humidity"].mean()

    f["morning_gen"] = d[(d.hour >= 6) & (d.hour < 12)].groupby("site_id")["gen"].mean()
    f["afternoon_gen"] = d[(d.hour >= 12) & (d.hour < 18)].groupby("site_id")["gen"].mean()
    f["evening_load"] = d[(d.hour >= 18) & (d.hour < 23)].groupby("site_id")["load"].mean()

    order = [c for c in FEATURE_INFO if c in f.columns]
    return f[order].fillna(0.0)


def hourly_profile(df: pd.DataFrame, cols: list[str], sites: list[str] | None = None) -> pd.DataFrame:
    d = df if sites is None else df[df["site_id"].isin(sites)]
    d = d.assign(hour=d["timestamp"].dt.hour)
    return d.groupby("hour")[[c for c in cols if c in d.columns]].mean()


# --------------------------------------------------------------------------- #
# 3. Preparation: scaling, K selection, PCA
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    clean: pd.DataFrame
    clean_report: dict
    meta: pd.DataFrame
    features: pd.DataFrame
    feature_cols: list[str]
    scaler: StandardScaler
    X: np.ndarray
    k_table: pd.DataFrame
    recommended_k: int
    elbow_k: int
    best_sil_k: int
    pca: PCA
    coords: np.ndarray
    dist: np.ndarray = field(repr=False, default=None)  # pairwise site distances


def _elbow(ks: np.ndarray, inertia: np.ndarray) -> int:
    if len(ks) < 3:
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-12)
    return int(ks[np.argmax((1 - x) - y)])


def _prune_collinear(f: pd.DataFrame, thr: float = 0.97) -> list[str]:
    """Drop features that are near-duplicates (|r| > thr) of an earlier one."""
    corr = f.corr().abs().fillna(0)
    keep: list[str] = []
    for c in f.columns:
        if f[c].std() < 1e-9:
            continue
        if all(corr.loc[c, k] <= thr for k in keep):
            keep.append(c)
    return keep


def prepare(raw: pd.DataFrame) -> Prepared:
    clean, rep = clean_data(raw)
    feats = engineer_features(clean)
    if len(feats) < 4:
        raise ValueError("At least 4 sites are needed for clustering.")
    cols = _prune_collinear(feats)          # features actually used for clustering
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[cols].to_numpy())

    kmax = int(min(8, len(feats) - 1))
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append(
            dict(
                k=k,
                inertia=float(km.inertia_),
                silhouette=float(silhouette_score(X, km.labels_)),
                davies_bouldin=float(davies_bouldin_score(X, km.labels_)),
                calinski_harabasz=float(calinski_harabasz_score(X, km.labels_)),
            )
        )
    kt = pd.DataFrame(rows)
    best_sil_k = int(kt.loc[kt["silhouette"].idxmax(), "k"])
    elbow = _elbow(kt["k"].to_numpy(), kt["inertia"].to_numpy())
    # Balanced rule: prefer the elbow K unless its silhouette is clearly worse (<85 % of the best).
    sil = kt.set_index("k")["silhouette"]
    rec = elbow if sil[elbow] >= 0.85 * sil.max() else best_sil_k

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    return Prepared(
        clean=clean, clean_report=rep, meta=site_meta(clean), features=feats,
        feature_cols=cols, scaler=scaler, X=X, k_table=kt,
        recommended_k=rec, elbow_k=elbow, best_sil_k=best_sil_k, pca=pca, coords=pca.transform(X),
        dist=cdist(X, X),
    )


# --------------------------------------------------------------------------- #
# 4. K-Means + automatic profiling
# --------------------------------------------------------------------------- #
@dataclass
class ClusterResult:
    k: int
    model: KMeans
    remap: np.ndarray            # raw sklearn label -> display label
    centers: np.ndarray          # scaled centres in display order
    labels: np.ndarray           # display labels per site
    n_iter: int
    inertia: float
    metrics: dict
    profiles: pd.DataFrame       # one row per cluster
    names: dict[int, str]
    assignments: pd.DataFrame    # site_id, cluster, cluster_name, ...
    validation: dict | None
    scale: float                 # RMS within-cluster distance (for similarity)


def _base_type(solar: float, wind: float) -> tuple[str, str]:
    if solar >= 0.65:
        return "Solar-Dominant", "☀️"
    if wind >= 0.65:
        return "Wind-Dominant", "🌬️"
    return "Hybrid", "🔋"


def _level(value: float, series: pd.Series) -> str:
    lo, hi = series.quantile([0.33, 0.67])
    return "Low" if value <= lo else ("High" if value >= hi else "Medium")


def _name_clusters(prof: pd.DataFrame, feats: pd.DataFrame) -> dict[int, str]:
    base, icons = {}, {}
    for c, r in prof.iterrows():
        b, ic = _base_type(r["solar_share"], r["wind_share"])
        base[c], icons[c] = b, ic
    names = {}
    for b in set(base.values()):
        members = [c for c in base if base[c] == b]
        if len(members) > 1:
            members = sorted(members, key=lambda c: prof.loc[c, "avg_load"], reverse=True)
            tiers = ["Large", "Small"] if len(members) == 2 else [f"Tier {i + 1}" for i in range(len(members))]
            for c, t in zip(members, tiers):
                names[c] = f"{b} · {t}"
        else:
            names[members[0]] = b
    overall_batt = feats["battery_share"].mean()
    for c in names:
        if prof.loc[c, "battery_share"] > max(0.05, 1.5 * overall_batt):
            names[c] += " + Storage"
        names[c] = f"{icons[c]} {names[c]}"
    return names


def cluster(prep: Prepared, k: int, n_init: int = 20, init: str = "k-means++") -> ClusterResult:
    km = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=SEED).fit(prep.X)
    f = prep.features.copy()
    f["_raw"] = km.labels_
    # order clusters by solar share (desc) so ids are stable & meaningful
    order = f.groupby("_raw")["solar_share"].mean().sort_values(ascending=False).index.to_numpy()
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[km.labels_]
    centers = km.cluster_centers_[order]
    f["cluster"] = labels
    f = f.drop(columns="_raw")

    prof = f.groupby("cluster")[list(prep.features.columns)].mean()
    names = _name_clusters(prof, prep.features)
    prof.insert(0, "n_sites", f.groupby("cluster").size())
    prof.insert(0, "name", [names[c] for c in prof.index])
    prof["solar_var_level"] = [_level(v, prep.features["solar_variability"]) for v in prof["solar_variability"]]
    prof["load_level"] = [_level(v, prep.features["avg_load"]) for v in prof["avg_load"]]

    m = {
        "silhouette": float(silhouette_score(prep.X, labels)),
        "davies_bouldin": float(davies_bouldin_score(prep.X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(prep.X, labels)),
        "inertia": float(km.inertia_),
    }
    d_own = np.linalg.norm(prep.X - centers[labels], axis=1)
    scale = float(np.sqrt(np.mean(d_own**2))) or 1.0

    asg = pd.DataFrame({"site_id": prep.features.index, "cluster": labels})
    asg["cluster_name"] = asg["cluster"].map(names)
    asg = asg.set_index("site_id").join(prep.meta)

    res = ClusterResult(
        k=k, model=km, remap=remap, centers=centers, labels=labels, n_iter=int(km.n_iter_),
        inertia=float(km.inertia_), metrics=m, profiles=prof, names=names,
        assignments=asg.reset_index(), validation=None, scale=scale,
    )
    res.validation = validate(prep, res)
    return res


# --------------------------------------------------------------------------- #
# 5. Validation against known site types
# --------------------------------------------------------------------------- #
def validate(prep: Prepared, res: ClusterResult) -> dict | None:
    meta = prep.meta
    if "site_type" not in meta.columns or meta["site_type"].dropna().nunique() < 2:
        return None
    known = meta["site_type"].reindex(prep.features.index)
    ok = known.notna().to_numpy()
    if ok.sum() < 4:
        return None
    k_known = known[ok].to_numpy()
    lab = res.labels[ok]
    ct = pd.crosstab(pd.Series(lab, name="Cluster"), pd.Series(k_known, name="Known type"))
    majority = ct.idxmax(axis=1).to_dict()
    purity = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
    table = pd.DataFrame(
        {"site_id": prep.features.index[ok], "known_type": k_known, "cluster": lab}
    )
    table["cluster_name"] = table["cluster"].map(res.names)
    table["cluster_majority_type"] = table["cluster"].map(majority)
    table["match"] = table["known_type"] == table["cluster_majority_type"]
    return {
        "crosstab": ct,
        "majority": majority,
        "purity": purity,
        "agreement": float(table["match"].mean()),
        "ari": float(adjusted_rand_score(k_known, lab)),
        "nmi": float(normalized_mutual_info_score(k_known, lab)),
        "homogeneity": float(homogeneity_score(k_known, lab)),
        "completeness": float(completeness_score(k_known, lab)),
        "table": table,
        "n_checked": int(ok.sum()),
    }


# --------------------------------------------------------------------------- #
# 6. Prediction for new sites
# --------------------------------------------------------------------------- #
def similarity_pct(d: np.ndarray | float, scale: float) -> np.ndarray | float:
    return 100.0 * np.exp(-0.5 * (np.asarray(d) / scale) ** 2)


def _nearest(prep: Prepared, res: ClusterResult, values: dict[str, float]) -> dict:
    """Nearest-centroid assignment using the features that are known.

    For a full feature vector this is identical to ``KMeans.predict``; when only
    a handful of readings are available the distance is computed over those
    dimensions only (unknown dimensions would otherwise inject noise).
    """
    cols = prep.feature_cols
    idx = [i for i, c in enumerate(cols) if c in values]
    mean, sd = prep.scaler.mean_[idx], prep.scaler.scale_[idx]
    z = (np.array([values[cols[i]] for i in idx], dtype=float) - mean) / sd
    d = np.linalg.norm(res.centers[:, idx] - z, axis=1)
    # within-cluster RMS distance on the same subset of dimensions
    own = np.linalg.norm(prep.X[:, idx] - res.centers[res.labels][:, idx], axis=1)
    s = float(np.sqrt(np.mean(own**2))) or 1.0
    best = int(np.argmin(d))
    sims = similarity_pct(d, s * 1.5)
    return {"cluster": best, "name": res.names[best], "distances": d, "similarity": sims,
            "n_dims": len(idx)}


def predict_new_site(prep: Prepared, res: ClusterResult, solar_kwh_day: float, wind_kwh_day: float,
                     load_kwh_day: float, irradiance: float, wind_speed: float, temperature: float) -> dict:
    solar, wind, load = solar_kwh_day / 24, wind_kwh_day / 24, max(load_kwh_day / 24, EPS)
    tot = solar + wind + EPS
    vals = {
        "avg_solar": solar, "avg_wind": wind, "avg_load": load,
        "solar_share": solar / tot, "wind_share": wind / tot,
        "renewable_ratio": (solar + wind) / load,
        "avg_irradiance": irradiance, "avg_wind_speed": wind_speed, "avg_temperature": temperature,
    }
    out = _nearest(prep, res, vals)
    out.update(
        solar_share=solar / tot, wind_share=wind / tot, renewable_ratio=(solar + wind) / load,
        load_level=_level(load, prep.features["avg_load"]),
    )
    return out


def predict_from_timeseries(prep: Prepared, res: ClusterResult, raw_new: pd.DataFrame) -> pd.DataFrame:
    clean, _ = clean_data(raw_new)
    feats = engineer_features(clean)
    rows = []
    for sid, r in feats.iterrows():
        vals = {c: float(r[c]) for c in prep.feature_cols if c in r.index}
        o = _nearest(prep, res, vals)
        rows.append({"site_id": sid, "cluster": o["cluster"], "cluster_name": o["name"],
                     "similarity_%": round(float(o["similarity"][o["cluster"]]), 1),
                     "solar_share_%": round(100 * float(r["solar_share"]), 1),
                     "wind_share_%": round(100 * float(r["wind_share"]), 1),
                     "avg_load_kW": round(float(r["avg_load"]), 1)})
    return pd.DataFrame(rows)


def site_similarity(prep: Prepared, a: str, b: str) -> float:
    ids = list(prep.features.index)
    d = prep.dist[ids.index(a), ids.index(b)]
    return float(max(0.0, 100 * (1 - d / prep.dist.max())))


# --------------------------------------------------------------------------- #
# 7. Narrative
# --------------------------------------------------------------------------- #
def conclusions(prep: Prepared, res: ClusterResult) -> list[str]:
    out = [
        f"{len(prep.features)} microgrid sites were described by {len(prep.feature_cols)} behavioural "
        f"features and grouped into {res.k} clusters with K-Means.",
        f"Silhouette score {res.metrics['silhouette']:.2f}, Davies–Bouldin {res.metrics['davies_bouldin']:.2f}: "
        + ("clusters are well separated." if res.metrics["silhouette"] >= 0.5
           else "clusters are reasonably separated." if res.metrics["silhouette"] >= 0.25
           else "cluster separation is weak."),
        f"Recommended K = {prep.recommended_k} (elbow method: K = {prep.elbow_k}; best silhouette: K = {prep.best_sil_k}).",
    ]
    for c, r in res.profiles.iterrows():
        out.append(
            f"Cluster {c} – {r['name']}: {int(r['n_sites'])} sites, solar {100 * r['solar_share']:.0f}% / "
            f"wind {100 * r['wind_share']:.0f}% of generation, average load {r['avg_load']:.1f} kW."
        )
    if res.validation:
        v = res.validation
        out.append(
            f"Against known site types, cluster purity is {100 * v['purity']:.1f}% "
            f"(ARI {v['ari']:.2f}, NMI {v['nmi']:.2f}), confirming that the learned clusters "
            "mirror real solar / wind / hybrid differences."
        )
    return out
