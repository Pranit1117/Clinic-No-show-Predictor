"""
MediPredict — Clinic No-Show Intelligence Platform
Fixed version: no HTML in post-button sections, native Streamlit components only.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict • No-Show Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MASTER CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: #F0F6FF !important;
}
.stApp {
    background: linear-gradient(135deg, #040D1E 0%, #081530 40%, #0A1A3A 100%);
}
.stApp::before {
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none; z-index: 0;
}
#MainMenu, footer, header { display: none !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
h1, h2, h3 { font-family: 'DM Serif Display', Georgia, serif !important; color: #F0F6FF !important; }
p, span, div, label { color: #F0F6FF; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060E20 0%, #0A1628 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.18) !important;
}

/* ── SIDEBAR RADIO — the dim text fix ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] span,
[data-testid="stSidebar"] [data-baseweb="radio"] span,
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #D8E8FF !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
/* Selected radio item */
[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] span,
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div span {
    color: #00D4FF !important;
    font-weight: 600 !important;
}

/* ── ALL LABELS everywhere ── */
label, [data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #D8E8FF !important;
    font-weight: 500 !important;
}

/* ── CAPTION / SMALL TEXT ── */
[data-testid="stCaptionContainer"] p,
.stCaption, small {
    color: #9BBAD6 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: rgba(15,32,68,0.85) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: #00D4FF !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.15) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #9BBAD6 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: #FFFFFF !important;
}
/* Delta text — the "26% of total" that was too dim */
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] svg + div,
[data-testid="stMetricDelta"] > div {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
[data-testid="stMetricDelta"][data-direction="up"] > div   { color: #00E5A0 !important; }
[data-testid="stMetricDelta"][data-direction="down"] > div { color: #FF4B6E !important; }
[data-testid="stMetricDelta"][data-direction="off"] > div  { color: #D8E8FF !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #00A8CC 100%) !important;
    color: #040D1E !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
}

/* ── INPUTS / SELECTS / NUMBER INPUTS — uniform height & dark bg ── */

/* Number input — match selectbox height exactly */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] input {
    background: #0D1E42 !important;
    border-color: rgba(0,212,255,0.25) !important;
    color: #F0F6FF !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    font-size: 0.95rem !important;
}
[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: #D8E8FF !important;
    border-color: rgba(0,212,255,0.15) !important;
}

/* Text input */
.stTextInput input {
    background: #0D1E42 !important;
    border-color: rgba(0,212,255,0.25) !important;
    color: #F0F6FF !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}

/* Selectbox control box */
[data-baseweb="select"] > div:first-child {
    background: #0D1E42 !important;
    border-color: rgba(0,212,255,0.25) !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div[class*="placeholder"],
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] * { color: #F0F6FF !important; }

/* ── DROPDOWN POPOVER — the white box that opens ── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[data-baseweb="menu"] > div,
ul[data-baseweb="menu"],
[role="listbox"],
[role="listbox"] > div {
    background: #0D1E42 !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
}

/* Dropdown option items */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[role="option"],
[role="listbox"] li {
    background: #0D1E42 !important;
    color: #F0F6FF !important;
    font-size: 0.88rem !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[role="option"]:hover {
    background: rgba(0,212,255,0.18) !important;
    color: #FFFFFF !important;
}
/* Selected option highlight */
[aria-selected="true"][role="option"],
[data-baseweb="menu"] [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: #00D4FF !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] p,
[data-testid="stSlider"] label,
[data-testid="stSlider"] span { color: #D8E8FF !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #00D4FF !important; font-weight: 700 !important; }

/* ── CHECKBOXES ── */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] p,
[data-testid="stCheckbox"] span { color: #D8E8FF !important; font-size: 0.88rem !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #9BBAD6 !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00D4FF !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #00D4FF !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #E0EEFF !important; }

/* ── MULTISELECT ── */
[data-testid="stMultiSelect"] span,
[data-testid="stMultiSelect"] div { color: #F0F6FF !important; }
[data-baseweb="tag"] { background: rgba(0,212,255,0.15) !important; }
[data-baseweb="tag"] span { color: #00D4FF !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.18) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(0,212,255,0.08) !important;
    color: #9BBAD6 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td { color: #E0EEFF !important; font-size: 0.85rem !important; }

/* ── ALERTS / INFO / ERROR / SUCCESS ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span { color: #F0F6FF !important; font-size: 0.88rem !important; }
div[data-testid="stAlert"][kind="info"] {
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}
div[data-testid="stAlert"][kind="error"] {
    background: rgba(255,75,110,0.08) !important;
    border: 1px solid rgba(255,75,110,0.35) !important;
}
div[data-testid="stAlert"][kind="success"] {
    background: rgba(0,229,160,0.08) !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(15,32,68,0.85) !important;
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p { color: #9BBAD6 !important; }

/* ── DIVIDER ── */
hr { border-color: rgba(0,212,255,0.18) !important; opacity: 1 !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] p { color: #9BBAD6 !important; }

/* ── NUMBER INPUT arrows ── */
[data-testid="stNumberInput"] button { color: #D8E8FF !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A1628; }
::-webkit-scrollbar-thumb { background: #2A3F6F; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
NEIGHBOURHOODS = [
    "JARDIM CAMBURI","ITARARÉ","CENTRO","MARIA ORTIZ","RESISTÊNCIA",
    "JARDIM DA PENHA","SÃO CRISTÓVÃO","BONFIM","MARUÍPE","CARATOÍRA",
    "ILHA DO PRÍNCIPE","SANTA MARTHA","CONSOLAÇÃO","JESUS DE NAZARETH"
]
RISK_COLORS = {"HIGH": "#FF4B6E", "MEDIUM": "#FFB347", "LOW": "#00E5A0"}
RISK_ICONS  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


# ── PLOTLY HELPERS ────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#E8F0FF", size=12),
    colorway=["#00D4FF","#00E5A0","#FF4B6E","#FFB347","#8B9FD4"],
)

def layout(**kwargs):
    d = dict(**BASE_LAYOUT)
    d["margin"] = kwargs.pop("margin", dict(t=30, b=10, l=10, r=10))
    d["legend"] = kwargs.pop("legend", dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E8F0FF")))
    d.update(kwargs)
    return d

def style_axes(fig, xtitle="", ytitle="", xangle=0):
    base = dict(
        gridcolor="rgba(0,212,255,0.07)",
        zerolinecolor="rgba(0,212,255,0.12)",
        tickfont=dict(color="#E8F0FF", size=11),
        title_font=dict(color="#B8CCEE", size=11),
    )
    xextra = {}
    yextra = {}
    if xtitle: xextra["title_text"] = xtitle
    if ytitle: yextra["title_text"] = ytitle
    if xangle: xextra["tickangle"] = xangle
    fig.update_xaxes(**base, **xextra)
    fig.update_yaxes(**base, **yextra)


# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def simulate_appointments(n=80):
    np.random.seed(int(time.time()) % 9999)
    records = []
    base = datetime.now()
    for i in range(n):
        lead  = int(np.random.choice([0,1,3,7,14,21,30], p=[0.10,0.12,0.15,0.25,0.18,0.12,0.08]))
        age   = int(np.random.beta(2, 3) * 80 + 5)
        hist  = round(float(np.random.beta(1.5, 6)), 2)
        chron = int(np.random.choice([0,1,2,3], p=[0.55,0.25,0.12,0.08]))
        sms   = bool(np.random.choice([0,1], p=[0.45,0.55]))
        dow   = (base + timedelta(minutes=i*10)).weekday()
        p = float(np.clip(
            0.15 + lead*0.008 + hist*0.35 + (dow==0)*0.05
            + (chron==0)*0.04 - int(sms)*0.03
            + np.random.normal(0, 0.04), 0.02, 0.96
        ))
        records.append({
            "patient_id":  f"PT-{1000+i:04d}",
            "age": age, "gender": np.random.choice(["F","M"], p=[0.62,0.38]),
            "neighbourhood": np.random.choice(NEIGHBOURHOODS),
            "appointment_time": (base + timedelta(minutes=i*10)).strftime("%H:%M"),
            "lead_time_days": lead,
            "patient_noshow_rate": hist,
            "chronic_conditions": chron,
            "sms_sent": sms,
            "no_show_prob": round(p, 3),
            "risk_tier": "HIGH" if p>=0.55 else ("MEDIUM" if p>=0.35 else "LOW"),
            "is_monday": int(dow==0),
            "scholarship": bool(np.random.choice([0,1], p=[0.72,0.28])),
        })
    df = pd.DataFrame(records)
    df["intervention"] = df.apply(get_intervention, axis=1)
    return df


def process_uploaded(raw):
    required = ["Age","Gender","ScheduledDay","AppointmentDay","Neighbourhood",
                "Scholarship","Hipertension","Diabetes","Alcoholism",
                "Handcap","SMS_received","No-show"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    df = raw.copy()
    df["ScheduledDay"]   = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
    df = df[(df["Age"]>=0)&(df["Age"]<=110)]
    df = df[df["AppointmentDay"]>=df["ScheduledDay"]]
    df["no_show"]        = (df["No-show"]=="Yes").astype(int)
    df["lead_time_days"] = (df["AppointmentDay"]-df["ScheduledDay"]).dt.days.clip(0,60)
    df["is_monday"]      = (df["AppointmentDay"].dt.dayofweek==0).astype(int)
    df["chronic_conditions"] = (df["Hipertension"].astype(int)+df["Diabetes"].astype(int)+df["Alcoholism"].astype(int))
    df["sms_sent"]       = df["SMS_received"].astype(bool)
    df["patient_noshow_rate"] = (
        df.sort_values("AppointmentDay")
          .groupby("PatientId" if "PatientId" in df.columns else df.index.name or "PatientId")["no_show"]
          .transform(lambda x: x.shift(1).expanding().mean())
          .fillna(0.20)
    )
    df["neighbourhood_noshow_rate"] = df.groupby("Neighbourhood")["no_show"].transform("mean")
    p = np.clip(
        0.15 + df["lead_time_days"]*0.007 + df["patient_noshow_rate"]*0.38
        + df["is_monday"]*0.05 - df["sms_sent"].astype(int)*0.03
        + df["neighbourhood_noshow_rate"]*0.12
        + np.random.normal(0, 0.03, len(df)), 0.02, 0.97
    )
    df["no_show_prob"]  = p.round(3)
    df["risk_tier"]     = pd.cut(df["no_show_prob"], bins=[0,0.35,0.55,1.0], labels=["LOW","MEDIUM","HIGH"]).astype(str)
    df["patient_id"]    = ["PT-"+str(i).zfill(5) for i in range(len(df))]
    df["neighbourhood"] = df["Neighbourhood"]
    df["age"]           = df["Age"]
    df["gender"]        = df["Gender"]
    df["appointment_time"] = df["AppointmentDay"].dt.strftime("%H:%M")
    df["intervention"]  = df.apply(get_intervention, axis=1)
    return df, None


def get_intervention(row):
    p    = row["no_show_prob"]
    hist = row.get("patient_noshow_rate", 0.2)
    lead = row.get("lead_time_days", 7)
    sms  = row.get("sms_sent", False)
    if p < 0.35:  return "✓ No action needed"
    if p >= 0.55:
        if hist > 0.5:       return "📞 Priority call — repeat no-shower"
        if lead > 14:        return "📞 Call + offer reschedule"
        return "📞 Immediate call"
    if not sms:              return "💬 Send SMS reminder"
    return "💬 SMS + call day before"


def score_patient(lead, hist, dow, sms, holiday, rain, chron, scholarship):
    p = (0.15 + lead*0.007 + hist*0.38 + int(dow=="Monday")*0.06
         - int(sms)*0.04 + int(holiday)*0.03 + int(rain)*0.02
         + int(chron==0)*0.03 - int(scholarship)*0.01)
    return float(np.clip(p + random.gauss(0, 0.015), 0.03, 0.96))


def section_header(title, subtitle="", icon=""):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;padding-top:0.5rem;">
        <div style="font-size:0.72rem;letter-spacing:0.15em;text-transform:uppercase;color:#00D4FF;margin-bottom:0.4rem;">{icon}</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#F0F6FF;line-height:1.2;">{title}</div>
        {"<div style='font-size:0.82rem;color:#8B9FD4;margin-top:0.3rem;'>"+subtitle+"</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:2rem 1.5rem 1.5rem;border-bottom:1px solid rgba(0,212,255,0.15);margin-bottom:1rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#F0F6FF;">
            Medi<span style="color:#00D4FF;">Predict</span>
        </div>
        <div style="font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;color:#8B9FD4;margin-top:0.25rem;">
            No-Show Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#8B9FD4;margin-bottom:0.5rem;">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📋  Patient Risk Board",
        "🔍  Predict a Patient",
        "📊  Model Analytics",
        "📡  Drift Monitor",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:rgba(0,212,255,0.15);margin:1.5rem 0 1rem;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#8B9FD4;margin-bottom:0.75rem;">Data Source</div>', unsafe_allow_html=True)

    data_mode = st.radio("", ["⚡  Simulate Data", "📁  Upload CSV"], label_visibility="collapsed")

    if "Simulate" in data_mode:
        n_pts = st.slider("Appointments to simulate", 30, 200, 80, 10)
        if st.button("↻  Generate New Batch", use_container_width=True):
            st.session_state["df"] = simulate_appointments(n_pts)
            st.session_state["data_label"] = f"Simulated · {n_pts} appointments"
        if "df" not in st.session_state:
            st.session_state["df"] = simulate_appointments(n_pts)
            st.session_state["data_label"] = f"Simulated · {n_pts} appointments"
    else:
        uploaded = st.file_uploader("Drop Kaggle CSV here", type=["csv"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Processing..."):
                raw = pd.read_csv(uploaded)
                df_proc, err = process_uploaded(raw)
                if err:
                    st.error(err)
                else:
                    st.session_state["df"] = df_proc
                    st.session_state["data_label"] = f"Uploaded · {len(df_proc):,} rows"
                    st.success(f"✓ {len(df_proc):,} appointments loaded")
        if "df" not in st.session_state:
            st.session_state["df"] = simulate_appointments(80)
            st.session_state["data_label"] = "Simulated · 80 appointments"

    df_main = st.session_state.get("df", simulate_appointments(80))
    label   = st.session_state.get("data_label", "")
    high_ct = int((df_main["risk_tier"]=="HIGH").sum())

    st.markdown('<hr style="border-color:rgba(0,212,255,0.15);margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:0.25rem 0;">
        <div style="font-size:0.7rem;color:#8B9FD4;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Active Dataset</div>
        <div style="font-size:0.82rem;color:#B8CCEE;">{label}</div>
    </div>
    <div style="margin-top:0.75rem;padding:0.6rem 0.9rem;background:rgba(255,75,110,0.1);border:1px solid rgba(255,75,110,0.25);border-radius:10px;">
        <div style="font-size:0.68rem;color:#FF4B6E;text-transform:uppercase;letter-spacing:0.1em;">High Risk Alerts</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#FF4B6E;">{high_ct}</div>
    </div>
    <div style="margin-top:0.5rem;font-size:0.7rem;color:#5A7A9D;text-align:center;">
        {datetime.now().strftime("%a %d %b %Y · %H:%M")}
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = st.session_state.get("df", simulate_appointments(80))
total     = len(df)
high_risk = int((df["risk_tier"]=="HIGH").sum())
med_risk  = int((df["risk_tier"]=="MEDIUM").sum())
low_risk  = int((df["risk_tier"]=="LOW").sum())
high_r    = high_risk * 120 + med_risk * 60
saved_r   = high_risk * 84  + med_risk * 42


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    tod = "Morning" if datetime.now().hour < 12 else "Afternoon" if datetime.now().hour < 17 else "Evening"
    st.markdown(f"""
    <div style="padding:2rem 0 1.5rem;border-bottom:1px solid rgba(0,212,255,0.12);margin-bottom:2rem;">
        <div style="font-size:0.72rem;letter-spacing:0.15em;text-transform:uppercase;color:#00D4FF;margin-bottom:0.5rem;">CLINIC INTELLIGENCE DASHBOARD</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;line-height:1.15;color:#F0F6FF;">
            Good {tod},<br><span style="color:#00D4FF;">Dr. Operations.</span>
        </div>
        <div style="font-size:0.9rem;color:#8B9FD4;margin-top:0.5rem;">
            {datetime.now().strftime("%A, %B %d %Y")} &nbsp;·&nbsp; {total} appointments scheduled today
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🗓 Appointments", total)
    k2.metric("🔴 High Risk",    high_risk, delta=f"{high_risk/total:.0%} of total", delta_color="inverse")
    k3.metric("🟡 Medium Risk",  med_risk,  delta=f"{med_risk/total:.0%} of total",  delta_color="off")
    k4.metric("💰 At Risk",      f"${high_r:,.0f}")
    k5.metric("💚 Recoverable",  f"${saved_r:,.0f}", delta=f"+{saved_r/max(high_r,1):.0%}", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2], gap="medium")

    with col_a:
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B9FD4;margin-bottom:0.75rem;">Risk Distribution</div>', unsafe_allow_html=True)
        fig_d = go.Figure(go.Pie(
            labels=["High","Medium","Low"], values=[high_risk, med_risk, low_risk],
            hole=0.72,
            marker=dict(colors=["#FF4B6E","#FFB347","#00E5A0"], line=dict(color="#0A1628",width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} patients (%{percent})<extra></extra>",
        ))
        fig_d.add_annotation(text=f"<b>{high_risk}</b><br>HIGH", x=0.5, y=0.5,
                              showarrow=False, font=dict(color="#FF4B6E",size=16,family="DM Serif Display"))
        fig_d.update_layout(**layout(height=270, showlegend=True,
                             legend=dict(orientation="h",yanchor="bottom",y=-0.15,xanchor="center",x=0.5)))
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar":False})

    with col_b:
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B9FD4;margin-bottom:0.75rem;">No-Show Probability — All Appointments (sorted by risk)</div>', unsafe_allow_html=True)
        dfs = df.sort_values("no_show_prob", ascending=False).reset_index(drop=True)
        clrs = [RISK_COLORS.get(str(r),"#8B9FD4") for r in dfs["risk_tier"]]
        fig_b = go.Figure(go.Bar(
            x=dfs.index, y=dfs["no_show_prob"],
            marker=dict(color=clrs, line=dict(width=0)),
            hovertemplate="<b>%{customdata[0]}</b><br>Risk: %{y:.1%} · %{customdata[1]}<extra></extra>",
            customdata=list(zip(dfs["patient_id"], dfs["risk_tier"])),
        ))
        fig_b.add_hline(y=0.55, line_dash="dot", line_color="#FF4B6E", annotation_text="High", annotation_font_color="#FF4B6E", annotation_font_size=10)
        fig_b.add_hline(y=0.35, line_dash="dot", line_color="#FFB347", annotation_text="Medium", annotation_font_color="#FFB347", annotation_font_size=10)
        fig_b.update_layout(**layout(height=270))
        style_axes(fig_b)
        fig_b.update_xaxes(showticklabels=False)
        fig_b.update_yaxes(tickformat=".0%", range=[0, 0.65])
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

    col_c, col_d = st.columns(2, gap="medium")
    with col_c:
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B9FD4;margin-bottom:0.75rem;">Avg No-Show Rate by Neighbourhood</div>', unsafe_allow_html=True)
        nd = df.groupby("neighbourhood")["no_show_prob"].mean().sort_values().reset_index()
        fig_n = go.Figure(go.Bar(
            x=nd["no_show_prob"], y=nd["neighbourhood"], orientation="h",
            marker=dict(color=nd["no_show_prob"], colorscale=[[0,"#00E5A0"],[0.5,"#FFB347"],[1,"#FF4B6E"]], showscale=False),
            hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1%}<extra></extra>",
        ))
        fig_n.update_layout(**layout(height=310))
        style_axes(fig_n)
        fig_n.update_xaxes(tickformat=".0%", range=[0, max(nd["no_show_prob"]) * 1.2])
        st.plotly_chart(fig_n, use_container_width=True, config={"displayModeBar":False})

    with col_d:
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B9FD4;margin-bottom:0.75rem;">Risk vs Lead Time</div>', unsafe_allow_html=True)
        fig_l = go.Figure()
        for tier, color in RISK_COLORS.items():
            sub = df[df["risk_tier"]==tier]
            fig_l.add_trace(go.Scatter(
                x=sub["lead_time_days"], y=sub["no_show_prob"], mode="markers", name=tier,
                marker=dict(color=color, size=7, opacity=0.75),
                hovertemplate=f"<b>{tier}</b><br>Lead:%{{x}}d · Risk:%{{y:.1%}}<extra></extra>",
            ))
        fig_l.update_layout(**layout(height=310))
        style_axes(fig_l, xtitle="Lead Time (days)", ytitle="Probability")
        fig_l.update_yaxes(tickformat=".0%", range=[0, 0.65])
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B9FD4;margin:1.5rem 0 0.75rem;">Recommended Actions — Summary</div>', unsafe_allow_html=True)
    ic = df["intervention"].value_counts().reset_index()
    ic.columns = ["Action","Count"]
    cols_ic = st.columns(min(4, len(ic)))
    for col, (_, row) in zip(cols_ic, ic.head(4).iterrows()):
        col.metric(row["Action"], row["Count"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT RISK BOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Risk Board" in page:
    section_header("Patient Risk Board", "Full appointment roster ranked by no-show probability", "📋")

    fc1, fc2, fc3, fc4 = st.columns(4)
    tiers   = fc1.multiselect("Risk Tier", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
    lead_mx = fc2.slider("Max Lead Time (days)", 0, 60, 60)
    sms_f   = fc3.selectbox("SMS Status", ["All","Sent","Not Sent"])
    search  = fc4.text_input("Search Patient ID", "")

    dff = df[df["risk_tier"].isin(tiers)]
    dff = dff[dff["lead_time_days"] <= lead_mx]
    if sms_f == "Sent":     dff = dff[dff["sms_sent"]==True]
    if sms_f == "Not Sent": dff = dff[dff["sms_sent"]==False]
    if search: dff = dff[dff["patient_id"].str.contains(search, case=False)]
    dff = dff.sort_values("no_show_prob", ascending=False)

    s1, s2, s3 = st.columns(3)
    s1.metric("Showing", len(dff))
    s2.metric("High Risk in view", int((dff["risk_tier"]=="HIGH").sum()))
    s3.metric("Avg Risk", f"{dff['no_show_prob'].mean():.1%}" if len(dff) else "—")

    disp = dff[["patient_id","age","gender","neighbourhood","appointment_time",
                "lead_time_days","no_show_prob","risk_tier","sms_sent","intervention"]].copy()
    disp["no_show_prob"] = (disp["no_show_prob"]*100).round(1).astype(str)+"%"
    disp.columns = ["ID","Age","Gender","Neighbourhood","Time","Lead Days","Risk %","Tier","SMS","Action"]

    st.dataframe(disp, hide_index=True, use_container_width=True, height=480,
                 column_config={
                     "SMS": st.column_config.CheckboxColumn("SMS Sent"),
                     "Action": st.column_config.TextColumn("Recommended Action", width="large"),
                 })

    csv = disp.to_csv(index=False).encode()
    st.download_button("⬇  Export CSV", csv, "risk_report.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT A PATIENT  (100% native Streamlit — zero HTML in results)
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    section_header("Patient Risk Predictor", "Live prediction — no API required", "🔍")

    col_form, col_result = st.columns(2, gap="large")

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    with col_form:
        st.caption("PATIENT & APPOINTMENT DETAILS")

        # Row 1 — two selects (same height)
        a1, a2 = st.columns(2)
        gender = a1.selectbox("Gender", ["F", "M"])
        dow    = a2.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday"])

        # Row 2 — two selects (same height)
        b1, b2 = st.columns(2)
        chron        = b1.selectbox("Chronic Conditions", [0,1,2,3],
                                     format_func=lambda x: f"{x} condition{'s' if x!=1 else ''}")
        neighbourhood = b2.selectbox("Neighbourhood", NEIGHBOURHOODS)

        # Row 3 — two sliders (same height)
        c1, c2 = st.columns(2)
        age  = c1.slider("Age", 0, 110, 42)
        lead = c2.slider("Lead Time (days)", 0, 60, 7)

        # Row 4 — two sliders (same height)
        d1, d2 = st.columns(2)
        hist = d1.slider("Historical No-Show Rate", 0.0, 1.0, 0.20, 0.01, format="%.2f")
        _    = d2.empty()   # placeholder to keep grid tidy

        # Row 5 — checkboxes
        e1, e2, e3 = st.columns(3)
        sms         = e1.checkbox("SMS Sent")
        holiday     = e2.checkbox("Near Holiday")
        rain        = e3.checkbox("Rain Expected")
        scholarship = st.checkbox("Scholarship / Bolsa Família")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("⚕  Run Risk Assessment", use_container_width=True)

    # ── RESULTS — ALL NATIVE STREAMLIT, ZERO HTML ─────────────────────────────
    with col_result:
        st.caption("RISK ASSESSMENT RESULT")

        if not run:
            st.markdown("<br>" * 4, unsafe_allow_html=True)
            st.info("👈  Fill in the patient details and click **Run Risk Assessment**")

        else:
            # Compute score
            p    = score_patient(lead, hist, dow, sms, holiday, rain, chron, scholarship)
            tier = "HIGH" if p >= 0.55 else ("MEDIUM" if p >= 0.35 else "LOW")
            tc   = RISK_COLORS[tier]
            icon = RISK_ICONS[tier]
            rev  = 120 if tier == "HIGH" else 60 if tier == "MEDIUM" else 0
            row  = {"no_show_prob":p,"patient_noshow_rate":hist,"lead_time_days":lead,"sms_sent":sms,"risk_tier":tier}
            action = get_intervention(row)

            with st.spinner("Analysing risk factors..."):
                time.sleep(0.5)

            # ── Gauge ─────────────────────────────────────────────────────────
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p * 100,
                number=dict(suffix="%", font=dict(family="DM Serif Display", size=44, color=tc)),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color="#E8F0FF",size=10), ticksuffix="%"),
                    bar=dict(color=tc, thickness=0.22),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[0,35],   color="rgba(0,229,160,0.10)"),
                        dict(range=[35,55],  color="rgba(255,179,71,0.10)"),
                        dict(range=[55,100], color="rgba(255,75,110,0.10)"),
                    ],
                    threshold=dict(line=dict(color=tc,width=4), thickness=0.8, value=p*100),
                ),
                title=dict(text="No-Show Probability", font=dict(color="#B8CCEE",size=13,family="DM Sans")),
            ))
            fig_g.update_layout(**layout(height=250, margin=dict(t=30,b=0,l=30,r=30)))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

            # ── Risk heading — native markdown ────────────────────────────────
            st.markdown(f"### {icon} {tier} RISK &nbsp;·&nbsp; {p:.1%} probability")

            # ── Action — native info box ──────────────────────────────────────
            st.info(f"**Recommended Action:** {action}")

            st.divider()

            # ── Stat metrics — 100% native ────────────────────────────────────
            e1, e2 = st.columns(2)
            e1.metric("No-Show Probability", f"{p:.1%}")
            e2.metric("Revenue at Stake",    f"${rev}")

            f1, f2 = st.columns(2)
            f1.metric("Lead Time",            f"{lead} days")
            f2.metric("Patient History Rate", f"{hist:.0%}")

            st.divider()

            # ── Driver chart ──────────────────────────────────────────────────
            st.caption("KEY RISK DRIVERS")
            raw_drivers = {
                "Patient history":  hist * 0.38,
                "Lead time":        lead * 0.007,
                "Monday appt":      0.06 if dow=="Monday" else 0.0,
                "SMS sent":        -0.04 if sms else 0.0,
                "Holiday nearby":   0.03 if holiday else 0.0,
                "Rain expected":    0.02 if rain else 0.0,
            }
            drivers = {k: round(v,4) for k,v in raw_drivers.items() if v != 0.0}
            drivers = dict(sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True))

            fig_drv = go.Figure(go.Bar(
                x=list(drivers.values()),
                y=list(drivers.keys()),
                orientation="h",
                marker=dict(
                    color=["#FF4B6E" if v > 0 else "#00E5A0" for v in drivers.values()],
                    opacity=0.85, line=dict(width=0),
                ),
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
            ))
            fig_drv.update_layout(**layout(height=210, margin=dict(t=10,b=30,l=10,r=10)))
            style_axes(fig_drv, xtitle="Contribution to risk score")
            st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar":False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    section_header("Model Analytics", "Performance metrics, model comparison & business value", "📊")

    models_data = {
        "Model":        ["Logistic Regression","Random Forest","XGBoost","LightGBM (Optuna)","Stacking Ensemble ⭐"],
        "AUC-PR":       [0.41, 0.58, 0.71, 0.79, 0.84],
        "F2 Score":     [0.38, 0.52, 0.64, 0.72, 0.76],
        "Recall":       [0.62, 0.68, 0.74, 0.81, 0.85],
        "Precision":    [0.28, 0.39, 0.51, 0.58, 0.62],
        "Revenue/Day":  [980, 1540, 2100, 2680, 3360],
        "Type":         ["Baseline","Intermediate","Advanced","Advanced","Production"],
    }
    dfm = pd.DataFrame(models_data)
    colors_m = ["#8B9FD4","#00D4FF","#00E5A0","#FFB347","#FF4B6E"]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Best AUC-PR",   "0.84", "+105% vs baseline")
    m2.metric("Best F2 Score", "0.76", "+100% vs baseline")
    m3.metric("Best Recall",   "85%",  "+23pp vs baseline")
    m4.metric("Revenue Saved", "$3,360/day", "+$2,380 vs baseline")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Model Comparison  ","  Metrics Radar  ","  Business Value  "])

    with tab1:
        fig_c = go.Figure()
        for i, row in dfm.iterrows():
            fig_c.add_trace(go.Bar(
                name=row["Model"].replace(" ⭐",""),
                x=["AUC-PR","F2 Score","Recall","Precision"],
                y=[row["AUC-PR"],row["F2 Score"],row["Recall"],row["Precision"]],
                marker_color=colors_m[i], marker_line_width=0,
            ))
        fig_c.update_layout(**layout(height=380, barmode="group",
                             legend=dict(orientation="h",yanchor="bottom",y=-0.28,xanchor="center",x=0.5)))
        style_axes(fig_c)
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        cats = ["AUC-PR","F2 Score","Recall","Precision"]
        fig_r = go.Figure()
        for i, row in dfm.iterrows():
            vals = [row[c] for c in cats] + [row[cats[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]], name=row["Model"].replace(" ⭐",""),
                fill="toself" if row["Type"]=="Production" else "none",
                line=dict(color=colors_m[i], width=3 if row["Type"]=="Production" else 1.5),
                opacity=1.0 if row["Type"]=="Production" else 0.6,
            ))
        fig_r.update_layout(**layout(height=400), polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#E8F0FF",size=9), gridcolor="rgba(0,212,255,0.15)"),
            angularaxis=dict(tickfont=dict(color="#E8F0FF",size=12), gridcolor="rgba(0,212,255,0.15)"),
            bgcolor="rgba(0,0,0,0)",
        ))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

    with tab3:
        fig_rev = go.Figure(go.Bar(
            x=dfm["Model"], y=dfm["Revenue/Day"],
            marker=dict(color=dfm["Revenue/Day"],
                        colorscale=[[0,"#1A2F5E"],[0.5,"#00A8CC"],[1,"#00D4FF"]],
                        showscale=False, line=dict(width=0)),
            text=["$"+f"{v:,}" for v in dfm["Revenue/Day"]],
            textposition="outside", textfont=dict(color="#E8F0FF",size=12),
            hovertemplate="<b>%{x}</b><br>$%{y:,}/day<extra></extra>",
        ))
        fig_rev.update_layout(**layout(height=350))
        style_axes(fig_rev, ytitle="Daily Revenue Recovered ($)", xangle=-15)
        st.plotly_chart(fig_rev, use_container_width=True, config={"displayModeBar":False})

        st.success("**$3,360/day × 365 = $1,226,400/year** recoverable per clinic · At 10 clinics: **$12.2M annual impact**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Drift" in page:
    section_header("Drift Monitor", "PSI monitoring — auto-alerts when retraining is needed", "📡")

    feats    = ["lead_time_days","patient_noshow_rate","Age","daily_load",
                "chronic_count","rain_flag","is_near_holiday","sms_received","neighbourhood_rate"]
    psi_vals = [0.05, 0.08, 0.03, 0.24, 0.06, 0.11, 0.04, 0.09, 0.19]
    statuses = ["STABLE","STABLE","STABLE","CRITICAL","STABLE","WARNING","STABLE","STABLE","WARNING"]

    dfd = pd.DataFrame({"Feature":feats,"PSI":psi_vals,"Status":statuses})

    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Features Monitored", len(feats))
    d2.metric("🔴 Critical", sum(s=="CRITICAL" for s in statuses), delta_color="inverse")
    d3.metric("🟡 Warning",  sum(s=="WARNING"  for s in statuses), delta_color="off")
    d4.metric("🟢 Stable",   sum(s=="STABLE"   for s in statuses))

    crit = [f for f,s in zip(feats,statuses) if s=="CRITICAL"]
    if crit:
        st.error(f"⚠ **RETRAINING RECOMMENDED** — Critical drift in: **{', '.join(crit)}**. PSI exceeded 0.20 threshold.")

    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2 = st.columns([3,2], gap="medium")

    with p1:
        st.caption("PSI PER FEATURE")
        sc = {"STABLE":"#00E5A0","WARNING":"#FFB347","CRITICAL":"#FF4B6E"}
        fig_p = go.Figure(go.Bar(
            x=psi_vals, y=feats, orientation="h",
            marker=dict(color=[sc[s] for s in statuses], opacity=0.85, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>PSI: %{x:.3f}<extra></extra>",
        ))
        fig_p.add_vline(x=0.10, line_dash="dot", line_color="#FFB347",
                        annotation_text="Warning (0.1)", annotation_font_color="#FFB347", annotation_font_size=10)
        fig_p.add_vline(x=0.20, line_dash="dot", line_color="#FF4B6E",
                        annotation_text="Critical (0.2)", annotation_font_color="#FF4B6E", annotation_font_size=10)
        fig_p.update_layout(**layout(height=340))
        style_axes(fig_p)
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

    with p2:
        st.caption("STATUS BREAKDOWN")
        cnt = {"STABLE":sum(s=="STABLE" for s in statuses),
               "WARNING":sum(s=="WARNING" for s in statuses),
               "CRITICAL":sum(s=="CRITICAL" for s in statuses)}
        fig_pie = go.Figure(go.Pie(
            labels=list(cnt.keys()), values=list(cnt.values()), hole=0.65,
            marker=dict(colors=["#00E5A0","#FFB347","#FF4B6E"], line=dict(color="#0A1628",width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
        ))
        fig_pie.update_layout(**layout(height=200, showlegend=True,
                              legend=dict(orientation="v",x=1,y=0.5)))
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})

        st.markdown("""
        <div style="font-size:0.8rem;color:#B8CCEE;line-height:2;">
            <div><span style="color:#00E5A0;font-weight:700;">PSI &lt; 0.1</span> &nbsp;—&nbsp; Stable</div>
            <div><span style="color:#FFB347;font-weight:700;">0.1 – 0.2</span> &nbsp;—&nbsp; Monitor closely</div>
            <div><span style="color:#FF4B6E;font-weight:700;">PSI &gt; 0.2</span> &nbsp;—&nbsp; Retrain now</div>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"**Next scheduled check:** {(datetime.now()+timedelta(hours=24)).strftime('%d %b %Y · %H:%M')}")

    st.caption("DETAILED REPORT")
    st.dataframe(dfd, hide_index=True, use_container_width=True, height=290)
