Skip to content
Pranit1117
Clinic-No-show-Predictor
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Files
Go to file
t
.github
deployment/streamlit
streamlit_app.py
notebooks
src
tests
.env.example
.gitignore
Makefile
README.md
config.yaml
requirements(main).txt
requirements.txt
Clinic-No-show-Predictor/deployment/streamlit
/
streamlit_app.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing streamlit_app.py file contents
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
"""
MediPredict — Clinic No-Show Intelligence Platform
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

# ── CACHE BUSTER — change this string whenever simulate_appointments logic changes ──
DATA_VERSION = "v4"

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

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif !important; color: #F0F6FF !important; }
.stApp { background: linear-gradient(135deg, #040D1E 0%, #081530 40%, #0A1A3A 100%); }
.stApp::before {
    content: ''; position: fixed; top:0; left:0; right:0; bottom:0;
    background-image: linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px; pointer-events: none; z-index: 0;
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
