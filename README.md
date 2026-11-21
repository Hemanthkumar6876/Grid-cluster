# ⚡ Renewable Microgrid Behavioral Segmentation

An interactive Streamlit web app with a dark, glossy **3D-style UI** (neon gradients, glass cards, 3D press buttons and rotatable 3D charts) that groups renewable-microgrid sites into **behavioural clusters**
(☀️ Solar-dominant · 🌬️ Wind-dominant · 🔋 Hybrid · …) with **K-Means**, automatically names and profiles
the clusters, validates them against known site types, and classifies new sites.

```
Data → Cleaning → Feature Engineering → Standardisation → K selection → K-Means
     → Auto cluster naming/profiling → Validation → Visualisation → New-site prediction → Report
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```
(or double-click `run.bat` on Windows / run `./run.sh` on Linux/macOS.)

The app starts on the bundled sample dataset (50 sites, 16,800 hourly rows) – everything works immediately.
Upload your own CSV/Excel on the **Upload Data** page.

## Pages

| Page | What it does |
|---|---|
| 🏠 Home | Overview, feature cards, workflow, *Start Analysis* |
| 📊 Dashboard | KPIs; generation, load and weather profiles for all sites / a cluster / one site; **3D generation landscape** (site × hour surface) |
| 📁 Upload Data | CSV/Excel upload, schema check, row/site/missing counts, preview, sample download |
| ⚙️ Feature Engineering | Cleaning report, 21 engineered features (generation, load, weather, time-based), standardisation view, correlations, ML dataset |
| 🤖 K-Means Clustering | Choose K, **Run K-Means** (iterations, inertia, silhouette), assignments; **Elbow + Silhouette + DB + CH** K-selection |
| 📈 Cluster Analysis | **3D PCA cluster view** (rotatable, with cluster hulls), PCA 2-D scatter, auto-generated cluster profiles, radar, centroid heatmap + **3D centroid bars**, site map |
| 🔍 Site Comparison | Pick two sites → side-by-side metrics, similarity %, radar, hourly profiles |
| ✅ Validation | Silhouette, Davies-Bouldin, Calinski-Harabasz, inertia; known-vs-cluster table, purity, ARI, NMI, confusion matrix (**3D bars**) |
| 🔮 New Site Prediction | Enter 6 values (or upload a time-series) → predicted cluster + similarity |
| 📑 Reports | One-click **CSV / Excel / PDF** report |

## Dataset format

Required columns: `site_id, timestamp, solar_generation, wind_generation, load`
Optional: `battery_output, temperature, irradiance, wind_speed, humidity` (more features),
`latitude, longitude` (enables the map), `site_type` (enables validation against known types).
Column names are case-insensitive. Missing values are interpolated per site; duplicates are removed.

## Project layout

```
app.py          Streamlit UI (10 pages)
pipeline.py     cleaning, features, K selection, K-Means, naming, validation, prediction
report.py       CSV / Excel / PDF exporters
data_gen.py     synthetic dataset generator (python data_gen.py)
train.py        trains + saves models/kmeans_bundle.joblib and training_summary.json
data/           sample_microgrid_data.csv
models/         trained model bundle + training summary
tests/          24 automated tests (pipeline + headless UI for every page)
docs/           screenshots
```

## Training & testing

```bash
python train.py                   # train on sample data (or: python train.py my.csv --k 3)
python -m pytest -q tests         # 24 tests
```

Sample-data results (K = 3): silhouette 0.27, Davies-Bouldin 1.40, **cluster purity 96 %, ARI 0.90, NMI 0.87**
against the known site types (48 / 50 sites in the right cluster).

## How it works (viva notes)

* **Features** – per-site means/maxima/variability of solar & wind, solar/wind share, renewable-to-load ratio,
  battery discharge share, load statistics (mean, peak, CV, peak-to-average), average weather, and time-of-day
  features (morning/afternoon generation, evening load). Near-duplicate features (|r|>0.97) are dropped before clustering.
* **K selection** – elbow (max distance to the chord of the inertia curve) and silhouette. The recommended K is the elbow K
  unless its silhouette is < 85 % of the best.
* **Auto naming** – each cluster's mean solar share ≥ 65 % → Solar-Dominant, wind share ≥ 65 % → Wind-Dominant,
  otherwise Hybrid; battery-heavy clusters get “+ Storage”; same-type clusters are split by load (Large/Small).
  Cluster ids are ordered by solar share so they stay stable between runs.
* **Validation** – internal metrics + external agreement with `site_type` (purity, ARI, NMI, confusion matrix).
* **New site** – nearest centroid in standardised space (identical to `KMeans.predict` for a full feature vector; for the
  manual form only the features derivable from the 6 inputs are used). Similarity = Gaussian kernel of centroid distance.
  Manual-form units: generation/load in kWh/day (÷24 → mean kW), irradiance = midday peak (÷π → daily mean).
* **Sample data is synthetic** (clearly labelled) so the project is self-contained; swap in real data via Upload.


## GridCluster AI UI (v3)

The interface now follows the 16-screen "GridCluster AI" design: landing page, login, sign-up, Dashboard, Dataset upload,
Preprocessing, Feature Engineering, Optimal K, K-Means, Cluster visualization (2D/3D PCA), Cluster analysis, Site comparison,
Validation, New-site analysis, Reports and Profile. The dataset, pipeline and exports are unchanged.

* **Login:** local accounts (SQLite, salted PBKDF2 hashes) in `data/users.db`; **"Continue as Guest"** opens the demo without an account.
  There is no Google/OAuth sign-in - that needs a Google Cloud client ID and cannot ship inside a zip.
* Light theme is set in `.streamlit/config.toml`.
* Tests: `python -m pytest -q tests`  (26 tests, UI rendered headlessly for every page).
