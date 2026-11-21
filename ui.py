"""UI kit for the GridCluster AI interface: theme CSS, logo, hero artwork and small HTML components."""
from __future__ import annotations

import html

import streamlit as st

BLUE, BLUE_D = "#2563EB", "#1D4ED8"
NAVY = "#0B1B3A"
AMBER, GREEN, SKY, PURPLE, RED = "#F59E0B", "#10B981", "#38BDF8", "#8B5CF6", "#EF4444"


# ----------------------------------------------------------------------------- artwork
def logo_svg(size: int = 34) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22D3EE"/>'
        '<stop offset="1" stop-color="#2563EB"/></linearGradient></defs>'
        '<path d="M24 3 42 13.5v21L24 45 6 34.5v-21z" fill="url(#lg)"/>'
        '<path d="M26.5 10 15 26h8l-2 12 12-17h-8z" fill="#fff"/></svg>'
    )


def brand(dark: bool = True, size: int = 34) -> str:
    col = "#FFFFFF" if dark else NAVY
    return (f'<div style="display:flex;align-items:center;gap:10px">{logo_svg(size)}'
            f'<span style="font-weight:800;font-size:{size * 0.62:.0f}px;color:{col};letter-spacing:-.3px">GridCluster AI</span></div>')


def _turbine(x: float, y: float, s: float, dur: float) -> str:
    col = "#EAF2FF"
    blades = "".join(
        f'<rect x="-1.7" y="{-48 * s:.1f}" width="3.4" height="{48 * s:.1f}" rx="1.7" fill="{col}" transform="rotate({a})"/>'
        for a in (0, 120, 240)
    )
    return (f'<g transform="translate({x},{y})"><path d="M-2.4 {92 * s:.1f} L-1.4 0 H1.4 L2.4 {92 * s:.1f}z" fill="{col}" opacity=".96"/>'
            f'<g><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="{dur}s" repeatCount="indefinite"/>'
            f'{blades}<circle r="3.8" fill="#fff"/></g></g>')


def _panels(x: float, y: float, rows: int, cols: int) -> str:
    out = []
    for r in range(rows):
        for c in range(cols):
            px, py = x + c * 46 - r * 12, y + r * 20
            out.append(f'<polygon points="{px},{py} {px + 40},{py} {px + 34},{py + 15} {px - 6},{py + 15}" fill="#1E40AF" stroke="#93C5FD" stroke-width=".8"/>'
                       f'<line x1="{px + 20}" y1="{py}" x2="{px + 14}" y2="{py + 15}" stroke="#60A5FA" stroke-width=".5"/>')
    return "".join(out)


def hero_art(height: int = 460) -> str:
    """Animated landscape: wind turbines, solar array, small control building, power nodes."""
    t = "".join(_turbine(x, y, s, d) for x, y, s, d in [(560, 150, 1.25, 7), (650, 185, 0.95, 9), (735, 205, 0.75, 6), (470, 215, 0.65, 11)])
    return f"""
<svg viewBox="0 0 820 460" preserveAspectRatio="xMidYMid slice" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" style="display:block">
<defs>
 <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#050E24"/><stop offset=".55" stop-color="#17408B"/><stop offset="1" stop-color="#3B82C4"/></linearGradient>
 <radialGradient id="glow" cx=".18" cy=".72" r=".6"><stop offset="0" stop-color="#FDE68A" stop-opacity=".95"/><stop offset=".4" stop-color="#FB923C" stop-opacity=".35"/><stop offset="1" stop-color="#FB923C" stop-opacity="0"/></radialGradient>
 <linearGradient id="hill1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1D6F5A"/><stop offset="1" stop-color="#0B3B33"/></linearGradient>
 <linearGradient id="hill2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2E9A5F"/><stop offset="1" stop-color="#0F5132"/></linearGradient>
</defs>
<rect width="820" height="460" fill="url(#sky)"/>
<rect width="820" height="460" fill="url(#glow)"/>
<g fill="#fff" opacity=".7"><circle cx="90" cy="40" r="1.4"/><circle cx="210" cy="70" r="1"/><circle cx="330" cy="30" r="1.2"/><circle cx="440" cy="85" r="1"/><circle cx="620" cy="40" r="1.3"/><circle cx="760" cy="75" r="1"/><circle cx="520" cy="22" r="1"/></g>
<path d="M0 300 C120 255 230 290 340 262 C470 230 560 285 820 240 V460 H0z" fill="url(#hill1)"/>
{t}
<path d="M0 352 C150 318 260 350 400 332 C560 312 680 350 820 318 V460 H0z" fill="url(#hill2)"/>
{_panels(70, 352, 3, 5)}
<g transform="translate(470,330)"><rect width="74" height="42" rx="3" fill="#E2E8F0"/><rect x="6" y="8" width="22" height="14" fill="#38BDF8"/><rect x="34" y="8" width="22" height="14" fill="#38BDF8"/><rect x="60" y="8" width="8" height="30" fill="#64748B"/><rect x="-6" y="-8" width="86" height="10" rx="2" fill="#CBD5E1"/></g>
<g stroke="#FDE68A" stroke-width="1.2" fill="none" opacity=".85"><path d="M60 350 Q260 300 505 322"><animate attributeName="stroke-dasharray" values="0 400;40 360;0 400" dur="5s" repeatCount="indefinite"/></path><path d="M544 340 Q650 330 735 300"><animate attributeName="stroke-dasharray" values="0 300;30 270;0 300" dur="4s" repeatCount="indefinite"/></path></g>
<g fill="#FDE68A"><circle cx="60" cy="350" r="3.5"><animate attributeName="r" values="3;5.5;3" dur="2.4s" repeatCount="indefinite"/></circle><circle cx="505" cy="322" r="3.5"/><circle cx="735" cy="300" r="3.5"/></g>
</svg>"""


# ----------------------------------------------------------------------------- CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;}
.stApp {background:#EEF3FB;}
[data-testid="stHeader"] {background:transparent;pointer-events:none;} [data-testid="stHeader"] button,[data-testid="stToolbar"] a{pointer-events:auto;} [data-testid="stToolbar"]{display:none;} .st-key-logout button,.st-key-logout button p{color:#0F1F44 !important;font-weight:600;}
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.4rem;padding-bottom:2.5rem;max-width:1320px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0B1B3A 0%,#0A1633 100%);border-right:0;min-width:250px;}
[data-testid="stSidebar"] * {color:#CBD5F0;}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:3px;}
[data-testid="stSidebar"] [role="radiogroup"] > label {width:100%;padding:10px 14px;border-radius:10px;cursor:pointer;transition:background .15s;margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {background:rgba(255,255,255,.07);}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] > label p {font-weight:600;font-size:.93rem;color:#CBD5F0;margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {background:linear-gradient(90deg,#2563EB,#1D4ED8);box-shadow:0 6px 16px rgba(37,99,235,.45);}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {color:#fff;}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.12);}
[data-testid="stSidebar"] .stButton button {background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.18);border-radius:10px;font-weight:600;}
[data-testid="stSidebar"] .stButton button:hover {background:rgba(239,68,68,.25);border-color:#EF4444;}
.sb-ds {background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:10px 12px;font-size:.78rem;line-height:1.5;}
.sb-ds b {color:#fff;}

/* ---------- page header ---------- */
.ptitle {font-size:1.75rem;font-weight:800;color:#0F1F44;line-height:1.15;letter-spacing:-.4px;}
.psub {color:#64748B;font-size:.95rem;margin-top:3px;}
.userchip {display:flex;align-items:center;gap:10px;justify-content:flex-end;}
.userchip .av {width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#2563EB,#22D3EE);color:#fff !important;font-weight:800;display:flex;align-items:center;justify-content:center;}
.userchip b {color:#0F1F44;font-size:.9rem;} .userchip span {color:#94A3B8;font-size:.75rem;}

/* ---------- cards ---------- */
.card, .kpi, [data-testid="stMetric"] {background:#fff;border:1px solid #E2E8F4;border-radius:16px;box-shadow:0 1px 2px rgba(15,31,68,.04),0 8px 24px -12px rgba(15,31,68,.14);}
.card {padding:18px 20px;} .card h4 {margin:0 0 4px 0;color:#0F1F44;font-size:1.02rem;}
.card p {margin:0;color:#64748B;font-size:.88rem;}
.card-t {font-weight:700;color:#0F1F44;font-size:1rem;margin-bottom:8px;}
[data-testid="stMetric"] {padding:14px 18px;}
[data-testid="stMetricLabel"] p {color:#64748B !important;font-weight:600;}
[data-testid="stMetricValue"] {color:#0F1F44;font-weight:800;}
.kpi {display:flex;align-items:center;gap:14px;padding:16px 18px;}
.kpi .ic {width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;flex:none;}
.kpi .l {font-size:.78rem;color:#64748B;font-weight:600;} .kpi .v {font-size:1.9rem;font-weight:800;color:#0F1F44;line-height:1.1;}
.kpi .s {font-size:.72rem;color:#94A3B8;}
[data-testid="stPlotlyChart"] {background:#fff;border:1px solid #E2E8F4;border-radius:16px;padding:8px 8px 2px 8px;box-shadow:0 1px 2px rgba(15,31,68,.04),0 8px 24px -12px rgba(15,31,68,.14);}
[data-testid="stDataFrame"] {border:1px solid #E2E8F4;border-radius:14px;overflow:hidden;}

/* ---------- buttons / inputs ---------- */
[data-testid^="stBaseButton"] {border-radius:10px !important;font-weight:700 !important;}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {background:linear-gradient(180deg,#3B82F6,#2563EB) !important;border:0 !important;color:#fff !important;box-shadow:0 8px 18px -6px rgba(37,99,235,.6) !important;}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {filter:brightness(1.07);}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {background:#fff !important;color:#1E3A8A !important;border:1px solid #CBD5E6 !important;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="select"] > div {border-radius:10px !important;}
button[data-baseweb="tab"] {font-weight:700;} button[data-baseweb="tab"][aria-selected="true"] {color:#2563EB;}
.stTabs [data-baseweb="tab-highlight"] {background:#2563EB;}

/* ---------- components ---------- */
.bar-row {display:flex;align-items:center;gap:10px;margin:8px 0;font-size:.85rem;color:#334155;}
.bar-row .lab {width:132px;flex:none;color:#475569;font-weight:500;}
.bar-row .track {flex:1;background:#E8EEF8;border-radius:8px;height:10px;overflow:hidden;}
.bar-row .fill {height:100%;border-radius:8px;}
.bar-row .val {width:62px;text-align:right;font-weight:700;color:#0F1F44;font-variant-numeric:tabular-nums;}
.pill {display:inline-block;padding:3px 11px;border-radius:999px;font-size:.74rem;font-weight:700;}
.pill.blue {background:#DBEAFE;color:#1D4ED8;} .pill.green {background:#DCFCE7;color:#15803D;} .pill.amber {background:#FEF3C7;color:#B45309;} .pill.red {background:#FEE2E2;color:#B91C1C;}
.ok {display:inline-block;background:#DCFCE7;color:#15803D;font-weight:700;border-radius:8px;padding:2px 10px;font-size:.82rem;}
.flow {display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap;padding:6px 4px;}
.flow .st {text-align:center;flex:1;min-width:96px;} .flow .ci {width:54px;height:54px;border-radius:50%;margin:0 auto 6px auto;display:flex;align-items:center;justify-content:center;font-size:1.4rem;}
.flow .tx {font-size:.8rem;font-weight:600;color:#334155;} .flow .ar {color:#94A3B8;font-size:1.3rem;flex:none;}
.chk {display:flex;align-items:center;gap:10px;margin:9px 0;font-size:.9rem;color:#334155;}
.chk i {width:20px;height:20px;border-radius:50%;background:#DCFCE7;color:#15803D;font-style:normal;font-size:.72rem;font-weight:800;display:flex;align-items:center;justify-content:center;flex:none;}
.chk.off i {background:#F1F5F9;color:#94A3B8;}
.ctitle {font-weight:800;font-size:1.02rem;color:#0F1F44;margin-bottom:2px;}
.desc {color:#64748B;font-size:.82rem;margin-top:8px;line-height:1.45;}
.big {border-radius:18px;padding:20px 24px;background:linear-gradient(135deg,#EFF6FF,#ECFDF5);border:1px solid #BFDBFE;}
.big .t {font-size:.78rem;color:#2563EB;text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
.big .n {font-size:1.9rem;font-weight:800;color:#0F1F44;} .big .sub {color:#475569;}

/* ---------- public pages ---------- */
.nav {display:flex;align-items:center;justify-content:space-between;padding:6px 6px 14px 6px;}
.nav a {color:#334155;text-decoration:none;font-weight:600;margin:0 14px;font-size:.92rem;}
.hero {position:relative;overflow:hidden;border-radius:26px;min-height:440px;background:#071430;box-shadow:0 30px 60px -28px rgba(7,20,48,.8);}
.hero .art {position:absolute;inset:0;} .hero .art svg {width:100%;height:100%;}
.hero .shade {position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,14,36,.92) 0%,rgba(5,14,36,.65) 46%,rgba(5,14,36,0) 78%);}
.hero .txt {position:relative;z-index:2;padding:56px 52px;max-width:600px;}
.hero h1 {color:#fff;font-size:2.7rem;line-height:1.12;margin:0 0 14px 0;font-weight:800;letter-spacing:-1px;}
.hero p {color:#BFD0F0;font-size:1.05rem;line-height:1.55;margin:0;}
.authart {position:relative;overflow:hidden;border-radius:24px;min-height:640px;background:#071430;box-shadow:0 30px 60px -28px rgba(7,20,48,.8);}
.authart .art {position:absolute;inset:0;} .authart .art svg {width:100%;height:100%;}
.authart .shade {position:absolute;inset:0;background:linear-gradient(180deg,rgba(5,14,36,.88) 0%,rgba(5,14,36,.35) 55%,rgba(5,14,36,.7) 100%);}
.authart .txt {position:relative;z-index:2;padding:40px 42px;}
.authart h2 {color:#fff;font-size:1.85rem;line-height:1.2;margin:34px 0 10px 0;font-weight:800;}
.authart p {color:#BFD0F0;font-size:.98rem;line-height:1.55;max-width:420px;}
.authart .ft {position:absolute;z-index:2;left:0;right:0;bottom:26px;display:flex;justify-content:space-around;color:#DCE7FF;font-size:.78rem;text-align:center;}
.authart .ft div span {display:block;font-size:1.35rem;margin-bottom:4px;}
.authhead {font-size:1.7rem;font-weight:800;color:#0F1F44;margin:0 0 2px 0;} .authsub {color:#64748B;margin-bottom:12px;font-size:.92rem;}
[data-testid="stForm"] {background:#fff;border:1px solid #E2E8F4;border-radius:20px;padding:26px 28px;box-shadow:0 20px 40px -20px rgba(15,31,68,.25);}
.fcard {text-align:center;padding:20px 14px;} .fcard .fi {width:54px;height:54px;border-radius:14px;margin:0 auto 10px auto;display:flex;align-items:center;justify-content:center;font-size:1.6rem;}
.step-n {width:34px;height:34px;border-radius:50%;background:#2563EB;color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center;margin:0 auto 8px auto;}
</style>
"""

HIDE_SIDEBAR = "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {display:none !important;}</style>"


def inject(public: bool = False) -> None:
    st.markdown(CSS + (HIDE_SIDEBAR if public else ""), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- components
def kpi(icon: str, label: str, value: str, sub: str = "", bg: str = "#DBEAFE") -> str:
    return (f'<div class="kpi"><div class="ic" style="background:{bg}">{icon}</div><div><div class="l">{html.escape(label)}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div></div>')


def bar(label: str, frac: float, color: str, text: str | None = None) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    text = text if text is not None else f"{100 * frac:.0f}%"
    return (f'<div class="bar-row"><div class="lab">{label}</div><div class="track"><div class="fill" style="width:{100 * frac:.1f}%;background:{color}"></div></div>'
            f'<div class="val">{text}</div></div>')


def pill(text: str, kind: str = "blue") -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def flow(steps: list[tuple[str, str, str]]) -> str:
    """steps: (emoji, label, circle background colour)."""
    parts = []
    for i, (ic, tx, bg) in enumerate(steps):
        parts.append(f'<div class="st"><div class="ci" style="background:{bg}">{ic}</div><div class="tx">{tx}</div></div>')
        if i < len(steps) - 1:
            parts.append('<div class="ar">→</div>')
    return f'<div class="flow">{"".join(parts)}</div>'


def checklist(items: list[str], on: bool = True) -> str:
    cls = "chk" if on else "chk off"
    return "".join(f'<div class="{cls}"><i>✓</i><span>{html.escape(t)}</span></div>' for t in items)
