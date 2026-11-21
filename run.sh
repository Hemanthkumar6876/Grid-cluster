#!/usr/bin/env bash
# One-click launcher (Linux / macOS)
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
