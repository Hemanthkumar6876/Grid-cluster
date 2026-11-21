"""Train the K-Means model on a dataset and save it.

    python train.py                       # uses data/sample_microgrid_data.csv
    python train.py my_data.csv --k 3

Writes ``models/kmeans_bundle.joblib`` (scaler + model + feature list + cluster
names) and ``models/training_summary.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import pipeline as P

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "data" / "sample_microgrid_data.csv"))
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: recommended K)")
    a = ap.parse_args()

    raw = pd.read_excel(a.csv) if a.csv.endswith((".xlsx", ".xls")) else pd.read_csv(a.csv)
    prep = P.prepare(raw)
    k = a.k or prep.recommended_k
    res = P.cluster(prep, k)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(
        {"scaler": prep.scaler, "model": res.model, "feature_cols": prep.feature_cols,
         "remap": res.remap, "cluster_names": res.names, "k": k},
        out / "kmeans_bundle.joblib",
    )
    summary = {
        "sites": int(len(prep.features)), "features_engineered": int(prep.features.shape[1]),
        "features_used": len(prep.feature_cols), "recommended_k": prep.recommended_k,
        "elbow_k": prep.elbow_k, "k": k, "iterations": res.n_iter,
        "metrics": {m: round(v, 4) for m, v in res.metrics.items()},
        "cluster_names": {int(c): n for c, n in res.names.items()},
        "validation": None if not res.validation else {
            m: round(res.validation[m], 4) for m in ("purity", "agreement", "ari", "nmi", "homogeneity", "completeness")
        },
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out / 'kmeans_bundle.joblib'}")


if __name__ == "__main__":
    main()
