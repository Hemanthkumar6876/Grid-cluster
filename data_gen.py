"""Synthetic renewable-microgrid dataset generator.

Produces hourly records for solar-, wind- and hybrid-type microgrid sites with
realistic weather, load, a simple battery model, a few missing values and a few
duplicate rows (so the cleaning stage has something to do).

Run directly to write ``data/sample_microgrid_data.csv``:

    python data_gen.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    e = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _load_shape(hours: np.ndarray, sector: str) -> np.ndarray:
    h = hours.astype(float)
    if sector == "residential":
        s = 0.55 + 0.35 * np.exp(-0.5 * ((h - 7.5) / 1.6) ** 2) + 0.75 * np.exp(-0.5 * ((h - 20) / 2.2) ** 2)
    elif sector == "commercial":
        s = 0.5 + 0.7 * np.exp(-0.5 * ((h - 13) / 3.5) ** 2)
    else:  # industrial
        s = 0.9 + 0.15 * np.sin((h - 6) / 24 * 2 * np.pi)
    return s / s.mean()


def _wind_power(v: np.ndarray, cap: float) -> np.ndarray:
    cut_in, rated = 3.0, 11.0
    p = (v**3 - cut_in**3) / (rated**3 - cut_in**3)
    return cap * np.clip(p, 0.0, 1.0)


def generate_dataset(
    n_solar: int = 20,
    n_wind: int = 15,
    n_hybrid: int = 15,
    days: int = 14,
    seed: int = 42,
    missing_rate: float = 0.0015,
    n_duplicates: int = 12,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-03-01", periods=days * 24, freq="h")
    hours = ts.hour.to_numpy()
    n = len(ts)
    day_idx = np.arange(n) // 24

    cloud_day = rng.uniform(0.55, 1.0, size=days)[day_idx]
    clear_sky = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    temp_base = 24 + 8 * np.sin((hours - 9) / 24 * 2 * np.pi)

    types = ["Solar"] * n_solar + ["Wind"] * n_wind + ["Hybrid"] * n_hybrid
    rng.shuffle(types)

    centres = {"Solar": (27.0, 72.0), "Wind": (21.8, 70.2), "Hybrid": (24.5, 73.8)}
    sectors = ["residential", "commercial", "industrial"]

    frames = []
    for i, t in enumerate(types, start=1):
        sid = f"MG{i:03d}"
        if t == "Solar":
            cap_s, cap_w = rng.uniform(100, 200), rng.uniform(0, 18)
            mean_v, p_batt = rng.uniform(3.0, 4.8), 0.35
        elif t == "Wind":
            cap_s, cap_w = rng.uniform(0, 30), rng.uniform(100, 190)
            mean_v, p_batt = rng.uniform(6.8, 8.8), 0.25
        else:
            cap_s = rng.uniform(60, 115)
            cap_w = cap_s * rng.uniform(0.9, 1.6)
            mean_v, p_batt = rng.uniform(5.4, 6.8), 0.6

        # climate differs by site type: solar sites are sunnier and warmer, wind sites cloudier
        irr_factor = {"Solar": rng.uniform(1.05, 1.15), "Wind": rng.uniform(0.72, 0.85),
                      "Hybrid": rng.uniform(0.9, 1.0)}[t]
        t_off = {"Solar": 3.5, "Wind": -4.0, "Hybrid": 0.0}[t]
        irr = 1000 * clear_sky * cloud_day * irr_factor * (1 + rng.normal(0, 0.05, n))
        irr = np.clip(irr, 0, 1100)
        solar = cap_s * (irr / 1000) * 0.85 * (1 + rng.normal(0, 0.03, n))
        solar = np.clip(solar, 0, None)

        v = mean_v + 1.2 * np.cos((hours - 2) / 24 * 2 * np.pi) + _ar1(rng, n, 0.95, 0.55)
        v = np.clip(v, 0.2, 24)
        wind = np.clip(_wind_power(v, cap_w) * (1 + rng.normal(0, 0.03, n)), 0, None)

        gen_est = 0.17 * cap_s + 0.2 * cap_w
        avg_load = rng.uniform(0.6, 1.1) * gen_est + 8
        sector = rng.choice(sectors)
        load = avg_load * _load_shape(hours, sector) * (1 + _ar1(rng, n, 0.8, 0.03))
        load = np.clip(load, 1, None)

        battery = np.zeros(n)
        if rng.random() < p_batt:
            bp = rng.uniform(0.25, 0.5) * avg_load
            cap_kwh = bp * 4
            soc = 0.5 * cap_kwh
            gen = solar + wind
            for k in range(n):
                surplus = gen[k] - load[k]
                if surplus > 0:
                    ch = min(surplus, bp, (cap_kwh - soc) / 0.95)
                    soc += ch * 0.95
                    battery[k] = -ch
                else:
                    dis = min(-surplus, bp, soc)
                    soc -= dis
                    battery[k] = dis

        temp = temp_base + t_off + rng.normal(0, 0.8, n) + rng.normal(0, 1.0)
        hum = np.clip(80 - 1.2 * (temp - 20) + 12 * (1 - cloud_day) + rng.normal(0, 3, n), 20, 100)

        lat0, lon0 = centres[t]
        frames.append(
            pd.DataFrame(
                {
                    "site_id": sid,
                    "timestamp": ts,
                    "solar_generation": solar,
                    "wind_generation": wind,
                    "battery_output": battery,
                    "load": load,
                    "temperature": temp,
                    "irradiance": irr,
                    "wind_speed": v,
                    "humidity": hum,
                    "latitude": round(lat0 + rng.normal(0, 0.9), 4),
                    "longitude": round(lon0 + rng.normal(0, 1.1), 4),
                    "site_type": t,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)

    num_cols = ["solar_generation", "wind_generation", "load", "temperature", "irradiance", "wind_speed", "humidity"]
    for c in num_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, c] = np.nan

    if n_duplicates:
        dup = df.sample(n_duplicates, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)

    df[num_cols + ["battery_output"]] = df[num_cols + ["battery_output"]].round(2)
    return df


def main() -> None:
    out = Path(__file__).parent / "data" / "sample_microgrid_data.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows, {df['site_id'].nunique()} sites -> {out}")


if __name__ == "__main__":
    main()
