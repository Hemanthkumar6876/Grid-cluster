"""Report exporters: CSV, Excel and PDF (matplotlib only, no extra deps)."""
from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import pipeline as P  # noqa: E402

PALETTE = ["#F59E0B", "#10B981", "#3B82F6", "#8B5CF6", "#EF4444", "#14B8A6", "#EC4899"]


def _clean_name(s: str) -> str:
    return "".join(ch for ch in s if ord(ch) < 0x2000).strip()  # drop emoji for PDF fonts


def assignments_df(prep: P.Prepared, res: P.ClusterResult) -> pd.DataFrame:
    a = res.assignments.set_index("site_id")
    out = prep.features.join(a[["cluster", "cluster_name"]])
    return out.reset_index()


def to_csv(prep, res) -> bytes:
    return assignments_df(prep, res).to_csv(index=False).encode()


def _summary_df(prep, res) -> pd.DataFrame:
    r = prep.clean_report
    rows = [
        ("Sites", r["n_sites"]), ("Rows (raw)", r["rows_raw"]), ("Rows (clean)", r["rows_clean"]),
        ("Duplicates removed", r["duplicates_removed"]), ("Missing values filled", r["missing_total"]),
        ("Features engineered", prep.features.shape[1]), ("Features used for clustering", len(prep.feature_cols)),
        ("K (chosen)", res.k), ("Recommended K", prep.recommended_k), ("Elbow K", prep.elbow_k),
        ("Iterations", res.n_iter), ("Inertia", round(res.inertia, 3)),
        ("Silhouette", round(res.metrics["silhouette"], 4)),
        ("Davies-Bouldin", round(res.metrics["davies_bouldin"], 4)),
        ("Calinski-Harabasz", round(res.metrics["calinski_harabasz"], 3)),
    ]
    if res.validation:
        v = res.validation
        rows += [("Cluster purity", round(v["purity"], 4)), ("Adjusted Rand Index", round(v["ari"], 4)),
                 ("Normalised Mutual Info", round(v["nmi"], 4))]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def to_excel(prep, res) -> bytes:
    buf = io.BytesIO()
    prof = res.profiles.copy()
    prof.insert(0, "cluster", prof.index)
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        _summary_df(prep, res).to_excel(xw, sheet_name="Summary", index=False)
        prep.features.describe().T.round(3).to_excel(xw, sheet_name="Feature statistics")
        prep.k_table.round(4).to_excel(xw, sheet_name="K selection", index=False)
        prof.round(3).to_excel(xw, sheet_name="Cluster profiles", index=False)
        assignments_df(prep, res).round(3).to_excel(xw, sheet_name="Assignments", index=False)
        if res.validation:
            res.validation["table"].to_excel(xw, sheet_name="Validation", index=False)
            res.validation["crosstab"].to_excel(xw, sheet_name="Confusion")
        pd.DataFrame({"Conclusions": P.conclusions(prep, res)}).to_excel(xw, sheet_name="Conclusions", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 48)
    return buf.getvalue()


def _text_page(pdf, title, lines, size=10):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=17, weight="bold", color="#0F172A")
    y = 0.91
    for ln in lines:
        for w in textwrap.wrap(ln, 95) or [""]:
            fig.text(0.07, y, w, fontsize=size, family="DejaVu Sans")
            y -= 0.022
        y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)


def to_pdf(prep, res) -> bytes:
    buf = io.BytesIO()
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(res.profiles.index)}
    names = {c: _clean_name(n) for c, n in res.names.items()}
    with PdfPages(buf) as pdf:
        # 1 cover + dataset summary
        s = _summary_df(prep, res)
        lines = ["Dataset summary & model configuration", ""] + [f"{a:<34}{b}" for a, b in s.itertuples(index=False)]
        _text_page(pdf, "Renewable Microgrid Analysis Report", lines, size=10)

        # 2 K selection + PCA
        fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69))
        kt = prep.k_table
        ax[0, 0].plot(kt.k, kt.inertia, "o-", color="#3B82F6")
        ax[0, 0].axvline(prep.elbow_k, ls="--", c="grey")
        ax[0, 0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
        ax[0, 1].plot(kt.k, kt.silhouette, "o-", color="#10B981")
        ax[0, 1].axvline(prep.recommended_k, ls="--", c="grey")
        ax[0, 1].set(title="Silhouette vs K", xlabel="K", ylabel="Silhouette")
        for c in sorted(set(res.labels)):
            m = res.labels == c
            ax[1, 0].scatter(prep.coords[m, 0], prep.coords[m, 1], s=28, c=colors[c], label=f"C{c} {names[c]}")
        ax[1, 0].set(title="PCA projection of sites", xlabel="PC1", ylabel="PC2")
        ax[1, 0].legend(fontsize=6)
        sizes = res.profiles["n_sites"]
        ax[1, 1].bar([f"C{c}" for c in sizes.index], sizes.values, color=[colors[c] for c in sizes.index])
        ax[1, 1].set(title="Sites per cluster")
        fig.suptitle("K selection & cluster visualisation", fontsize=14, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

        # 3 profiles
        fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))
        pr = res.profiles
        x = np.arange(len(pr))
        ax[0].bar(x - 0.2, 100 * pr["solar_share"], 0.4, label="Solar share %", color="#F59E0B")
        ax[0].bar(x + 0.2, 100 * pr["wind_share"], 0.4, label="Wind share %", color="#3B82F6")
        ax[0].set_xticks(x, [f"C{c}\n{names[c]}" for c in pr.index], fontsize=7)
        ax[0].set(title="Generation mix per cluster", ylabel="% of generation")
        ax[0].legend()
        z = (pr[prep.feature_cols] - prep.features[prep.feature_cols].mean()) / prep.features[prep.feature_cols].std()
        im = ax[1].imshow(z.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax[1].set_yticks(range(len(prep.feature_cols)), prep.feature_cols, fontsize=7)
        ax[1].set_xticks(x, [f"C{c}" for c in pr.index])
        ax[1].set_title("Cluster centroids (z-scores)")
        fig.colorbar(im, ax=ax[1], shrink=0.7)
        fig.suptitle("Cluster profiles", fontsize=14, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

        # 4 validation + conclusions
        v = res.validation
        lines = [f"Silhouette {res.metrics['silhouette']:.3f} | Davies-Bouldin {res.metrics['davies_bouldin']:.3f} | "
                 f"Calinski-Harabasz {res.metrics['calinski_harabasz']:.1f} | Inertia {res.inertia:.1f}", ""]
        if v:
            lines += [f"Cluster purity {100 * v['purity']:.1f}%   ARI {v['ari']:.3f}   NMI {v['nmi']:.3f}", "",
                      "Known type vs cluster (confusion matrix):", v["crosstab"].to_string(), ""]
        else:
            lines += ["No 'site_type' column supplied - external validation skipped.", ""]
        lines += ["Conclusions", ""] + ["- " + _clean_name(c) for c in P.conclusions(prep, res)]
        _text_page(pdf, "Validation & conclusions", lines, size=9)
    return buf.getvalue()
