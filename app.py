# ============================================================
# app.py
# Sistem Pembelajaran Adaptif Matematik Tingkatan 4 dan 5
# Versi 35: aras semasa dashboard ikut aras akhir/adaptif terkini
# ============================================================

import os
import time
import math
import random
import sqlite3
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from database import (
    init_db, get_or_create_user, save_profile, create_quiz_session,
    save_answer, save_chat, complete_session, fetch_sessions,
    fetch_answers, fetch_latest_profile, get_fsrs_topic_state,
    save_fsrs_review, fetch_fsrs_progress
)

APP_TITLE = "Matematik Pintar SPM"
DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "hybrid_rf_dnn_bundle.pkl"
QUESTION_PATH = DATA_DIR / "Bank Soalan.xlsx"
LEVEL_TEXT = {0: "Rendah", 1: "Sederhana", 2: "Tinggi"}
LEVEL_COLOR = {0: "#38bdf8", 1: "#2563eb", 2: "#14b8a6"}
LEVEL_LIGHT = {0: "#ecfeff", 1: "#eff6ff", 2: "#ecfdf5"}

st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------
# CSS GAYA PREMIUM: Biru moden dan hijau petunjuk 
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root{
    --biru-gelap:#082f49;
    --biru:#2563eb;
    --biru-muda:#38bdf8;
    --hijau:#14b8a6;
    --hijau-muda:#ccfbf1;
    --latar:#f4fbff;
    --kad:#ffffff;
    --teks:#0f172a;
    --teks-lembut:#475569;
    --garis:#dbeafe;
    --bayang:0 22px 60px rgba(15, 23, 42, .10);
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif !important;
    color:var(--teks) !important;
}

.stApp{
    background:
        radial-gradient(circle at 8% 10%, rgba(56,189,248,.26), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(20,184,166,.20), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(37,99,235,.12), transparent 45%),
        linear-gradient(135deg,#f8fdff 0%,#eef7ff 42%,#f1fffb 100%) !important;
    background-attachment: fixed !important;
}

.block-container{
    max-width:1210px !important;
    padding-top:1.1rem !important;
    padding-bottom:4rem !important;
}

#MainMenu, footer, header {visibility:hidden;}

/* Bar sistem */
.v12-topbar{
    position: sticky;
    top: .75rem;
    z-index: 10;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    background:rgba(255,255,255,.86);
    border:1px solid rgba(147,197,253,.55);
    border-radius:26px;
    padding:16px 20px;
    box-shadow:0 18px 45px rgba(15,23,42,.08);
    backdrop-filter:blur(18px);
    margin-bottom:16px;
}
.v12-brand{display:flex;align-items:center;gap:12px;}
.v12-logo{
    width:52px;height:52px;border-radius:18px;
    display:flex;align-items:center;justify-content:center;
    color:#fff;font-size:27px;
    background:linear-gradient(135deg,#0f172a,#2563eb 55%,#14b8a6);
    box-shadow:0 14px 32px rgba(37,99,235,.28);
}
.v12-brand-title{
    font-family:'Space Grotesk', sans-serif !important;
    font-size:1.22rem;
    font-weight:800;
    letter-spacing:-.03em;
    color:#082f49 !important;
    background:none !important;
    -webkit-text-fill-color:initial !important;
}
.v12-brand-sub{font-size:.79rem;color:#64748b;font-weight:700;margin-top:1px;}
.v12-section-pill,.v12-user-pill,.v12-chip,.mp-badge{
    display:inline-flex;align-items:center;gap:6px;
    padding:7px 13px;border-radius:999px;
    font-size:.82rem;font-weight:850;
    border:1px solid rgba(37,99,235,.16);
    background:#eff6ff;color:#1d4ed8;
    white-space:nowrap;
}
.v12-section-pill{background:linear-gradient(135deg,#0f172a,#2563eb);color:#fff;border:0;}
.v12-user-pill{background:#ffffff;color:#0f172a;box-shadow:0 8px 20px rgba(15,23,42,.08);}

/* Butang sistem: semua butang diselaraskan dengan tema biru dan hijau */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button,
button[kind="primary"],
.stDownloadButton button{
    border:0 !important;
    border-radius:16px !important;
    min-height:44px !important;
    color:white !important;
    font-weight:850 !important;
    background:linear-gradient(135deg,#1d4ed8 0%,#0f766e 100%) !important;
    box-shadow:0 12px 24px rgba(37,99,235,.20) !important;
    transition:all .2s ease !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
button[kind="primary"]:hover,
.stDownloadButton button:hover{
    transform:translateY(-2px) scale(1.01);
    box-shadow:0 18px 34px rgba(20,184,166,.25) !important;
    filter:saturate(1.1);
}

/* Kad umum */
.v12-card,.v12-question-card,.v12-tutor-card,.v12-result-card{
    background:rgba(255,255,255,.90);
    border:1px solid rgba(191,219,254,.75);
    border-radius:28px;
    padding:24px;
    box-shadow:var(--bayang);
    backdrop-filter:blur(18px);
}
.v12-card-title{
    font-family:'Space Grotesk', sans-serif !important;
    font-size:1.45rem;
    font-weight:800;
    color:#0f172a !important;
    background:none !important;
    -webkit-text-fill-color:initial !important;
    margin-bottom:6px;
}
.v12-card-sub,.mp-small{color:#475569;font-size:.94rem;line-height:1.65;}

/* Hero biru moden */
.v12-hero-premium{
    min-height:520px;
    position:relative;
    overflow:hidden;
    border-radius:34px;
    padding:34px;
    background:
        radial-gradient(circle at 14% 10%, rgba(125,211,252,.35), transparent 24%),
        radial-gradient(circle at 88% 18%, rgba(45,212,191,.25), transparent 26%),
        linear-gradient(135deg,#071f36 0%,#0f4c81 45%,#2563eb 76%,#14b8a6 100%) !important;
    box-shadow:0 30px 70px rgba(2,132,199,.25);
    color:white;
}
.v12-hero-premium:before{
    content:"";
    position:absolute;
    width:330px;height:330px;border-radius:50%;
    right:-105px;top:-120px;
    background:rgba(255,255,255,.13);
}
.v12-hero-premium:after{
    content:"∑  π  √  x²";
    position:absolute;
    right:24px;bottom:18px;
    font-family:'Space Grotesk',sans-serif;
    font-weight:800;
    font-size:3rem;
    color:rgba(255,255,255,.12);
    letter-spacing:.08em;
}
.v12-hero-content{position:relative;z-index:1;}
.v12-kicker{
    display:inline-flex;align-items:center;gap:8px;
    padding:8px 14px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.24);
    font-weight:850;
    font-size:.86rem;
    margin-bottom:22px;
}
.v12-hero-title{
    font-family:'Space Grotesk',sans-serif !important;
    font-size:3.15rem;
    line-height:1.02;
    letter-spacing:-.055em;
    margin:0 0 16px 0;
    color:#ffffff !important;
    background:none !important;
    -webkit-text-fill-color:#ffffff !important;
    text-shadow:0 2px 14px rgba(0,0,0,.16);
}
.v12-hero-desc{
    max-width:96%;
    font-size:1.04rem;
    line-height:1.75;
    color:#e0f2fe;
    font-weight:700;
    margin-bottom:20px;
}
.v12-feature-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
    margin-top:22px;
}
.v12-feature-card{
    background:rgba(255,255,255,.13);
    border:1px solid rgba(255,255,255,.22);
    border-radius:20px;
    padding:14px 15px;
    min-height:86px;
}
.v12-feature-card b{display:block;font-size:.96rem;color:#fff;font-weight:850;margin-bottom:5px;}
.v12-feature-card span{font-size:.82rem;line-height:1.35;color:#dbeafe;font-weight:700;}
.v12-feature-chip{
    display:inline-flex;
    align-items:center;
    margin:6px 6px 0 0;
    padding:8px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.22);
    color:#fff;
    font-size:.82rem;
    font-weight:850;
}

/* Borang dan kawalan input */
[data-testid="stForm"]{
    background:rgba(255,255,255,.82) !important;
    border:1px solid rgba(147,197,253,.72) !important;
    border-radius:26px !important;
    padding:20px !important;
    box-shadow:0 16px 42px rgba(37,99,235,.08) !important;
}
label{font-weight:850 !important;color:#1e3a8a !important;}

/* Input, select dan number input ikut tema biru-hijau */
.stTextInput input, .stNumberInput input, textarea{
    border-radius:16px !important;
    border:1px solid #bfdbfe !important;
    background:#f8fbff !important;
    color:#0f172a !important;
}
.stTextInput input:focus, .stNumberInput input:focus, textarea:focus{
    border-color:#14b8a6 !important;
    box-shadow:0 0 0 3px rgba(20,184,166,.18) !important;
}
[data-baseweb="select"] > div{
    border-radius:16px !important;
    border-color:#bfdbfe !important;
    background:#f8fbff !important;
    box-shadow:none !important;
}
[data-baseweb="select"]:focus-within > div{
    border-color:#14b8a6 !important;
    box-shadow:0 0 0 3px rgba(20,184,166,.18) !important;
}
[data-baseweb="popover"] [role="listbox"]{
    border:1px solid #bfdbfe !important;
    border-radius:16px !important;
    box-shadow:0 18px 40px rgba(15,23,42,.16) !important;
}
[data-testid="stNumberInput"] button{
    background:linear-gradient(135deg,#1d4ed8,#0f766e) !important;
    color:white !important;
    border:none !important;
}
[data-testid="stNumberInput"] button:hover{
    background:linear-gradient(135deg,#0f766e,#14b8a6) !important;
}

/* Slider ikut tema sistem */
[data-testid="stSlider"] [role="slider"]{
    background:#0f766e !important;
    border:3px solid #e0f2fe !important;
    box-shadow:0 4px 14px rgba(20,184,166,.35) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div{
    color:#1d4ed8 !important;
}

/* Kad keputusan klasifikasi */
.v12-smart-strip{
    margin-top:18px;
    background:linear-gradient(135deg,#ffffff 0%,#eff6ff 52%,#ecfdf5 100%);
    border:1px solid rgba(147,197,253,.75);
    border-radius:28px;
    padding:24px;
    display:grid;
    grid-template-columns:1fr 250px;
    gap:20px;
    align-items:center;
    box-shadow:0 20px 48px rgba(37,99,235,.10);
}
.v12-smart-title{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:800;color:#0f172a;margin-bottom:7px;}
.v12-level-badge{
    display:inline-flex;align-items:center;justify-content:center;
    padding:10px 18px;
    border-radius:999px;
    color:#fff;
    font-size:1rem;
    font-weight:850;
    box-shadow:0 12px 24px rgba(37,99,235,.20);
    margin-bottom:10px;
}
.v12-confidence-track,.v12-progress-track{
    width:100%;height:13px;border-radius:999px;background:#dbeafe;overflow:hidden;
}
.v12-confidence-fill,.v12-progress-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#2563eb,#14b8a6);}

/* Papan pemuka */
.v12-dashboard-hero,.mp-dashboard-top{
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:28px;
    color:#fff;
    background:linear-gradient(135deg,#082f49,#1d4ed8 56%,#14b8a6);
    box-shadow:0 25px 60px rgba(37,99,235,.18);
    margin-bottom:20px;
}
.v12-dashboard-hero h2{color:#fff !important;-webkit-text-fill-color:#fff !important;background:none !important;margin:0 0 8px 0;}
.v12-dashboard-hero p{color:#e0f2fe;font-weight:700;line-height:1.6;}
.v12-status-item,.v12-result-stat{
    background:#fff;
    border:1px solid rgba(191,219,254,.8);
    border-radius:20px;
    padding:16px;
    box-shadow:0 14px 32px rgba(37,99,235,.08);
}

/* Kuiz */
.v12-quiz-head{
    display:flex;justify-content:space-between;align-items:center;
    border-radius:24px;padding:16px 20px;margin-bottom:14px;
    border:1px solid rgba(191,219,254,.78);
    background:rgba(255,255,255,.9);
    box-shadow:0 14px 32px rgba(37,99,235,.08);
}
.v12-progress-wrap{margin:0 0 18px 0;}
.v12-question-meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;}
.stRadio label{
    background:#ffffff !important;
    border:1px solid #bfdbfe !important;
    border-radius:16px !important;
    padding:12px 14px !important;
    margin-bottom:8px !important;
    box-shadow:0 8px 18px rgba(37,99,235,.04);
}
.stRadio label:hover{border-color:#14b8a6 !important;background:#f0fdfa !important;}
.v12-tutor-header{display:flex;justify-content:space-between;align-items:center;font-weight:850;color:#0f172a;margin-bottom:12px;}
.v12-ai-pill{font-size:.72rem;padding:5px 10px;border-radius:999px;background:#ecfdf5;color:#0f766e;border:1px solid #99f6e4;}
.mp-status-ok,.mp-status-warn{display:inline-flex;padding:7px 12px;border-radius:999px;font-weight:850;font-size:.82rem;margin-bottom:12px;}
.mp-status-ok{background:#ecfdf5;color:#047857;}
.mp-status-warn{background:#fff7ed;color:#c2410c;}

/* Keputusan */
.v12-result-card{text-align:center;max-width:820px;margin:0 auto;}
.v12-success-icon{font-size:3rem;margin-bottom:8px;}
.v12-result-card h2{color:#0f172a !important;-webkit-text-fill-color:#0f172a !important;background:none !important;}
.v12-score-panel{
    margin:22px auto;
    max-width:320px;
    border-radius:28px;
    padding:24px;
    color:#fff;
    background:linear-gradient(135deg,#1d4ed8,#14b8a6);
    box-shadow:0 20px 46px rgba(37,99,235,.18);
}
.v12-score-panel .score{font-family:'Space Grotesk';font-weight:800;font-size:3.6rem;line-height:1;}
.v12-result-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px;}
.v12-result-stat b{display:block;color:#0f172a;font-size:1.3rem;font-weight:850;}
.v12-result-stat span{font-size:.85rem;color:#475569;font-weight:800;}

/* Jadual */
[data-testid="stDataFrame"]{border-radius:22px;overflow:hidden;}

/* Buang kotak kosong / pembungkus kosong Streamlit */
[data-testid="stMarkdownContainer"]:empty,
.element-container:has([data-testid="stMarkdownContainer"]:empty),
.element-container:has(iframe[height="0"]),
div:has(> div[data-testid="stMarkdownContainer"]:empty){
    display:none !important;
}
.element-container{margin-bottom:.35rem !important;}

/* Kad laporan dan visual lebih selari dengan tema */
.js-plotly-plot, .stPlotlyChart, [data-testid="stImage"]{
    border-radius:22px !important;
}

/* Panel pilihan latihan yang lebih hidup */
.v18-training-panel{
    background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(239,246,255,.88));
    border:1px solid rgba(147,197,253,.8);
    border-radius:30px;
    padding:26px;
    box-shadow:0 24px 60px rgba(37,99,235,.12);
    position:relative;
    overflow:hidden;
}
.v18-training-panel:after{
    content:"";
    position:absolute;
    width:220px;height:220px;border-radius:50%;
    right:-70px;top:-80px;
    background:radial-gradient(circle,rgba(20,184,166,.18),transparent 70%);
}
.v18-panel-title{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.35rem;
    font-weight:850;
    color:#082f49;
    margin-bottom:6px;
}
.v18-note{
    background:#ecfdf5;
    border:1px solid #99f6e4;
    color:#0f766e;
    padding:13px 15px;
    border-radius:18px;
    font-weight:750;
    line-height:1.55;
    margin-top:14px;
}
.v18-count-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    margin:16px 0 18px;
}
.v18-count-card{
    background:#ffffff;
    border:1px solid #dbeafe;
    border-radius:20px;
    padding:14px;
    box-shadow:0 12px 28px rgba(37,99,235,.07);
}
.v18-count-card span{font-size:.78rem;color:#64748b;font-weight:800;}
.v18-count-card b{display:block;font-size:1.35rem;color:#082f49;margin-top:4px;}
.v18-system-card{
    background:linear-gradient(135deg,#082f49 0%,#1d4ed8 58%,#0f766e 100%);
    color:white;
    border-radius:30px;
    padding:26px;
    box-shadow:0 24px 58px rgba(37,99,235,.18);
    min-height:100%;
}
.v18-system-card h3{color:white !important;-webkit-text-fill-color:white !important;background:none !important;margin-top:0;}
.v18-system-card p{color:#dbeafe;line-height:1.7;font-weight:700;}
.v18-mini-feature{
    background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.18);
    border-radius:18px;
    padding:12px;
    margin:10px 0;
    color:#eff6ff;
    font-weight:750;
}
.v18-timer-card{
    font-family:Inter,Arial;
    background:linear-gradient(135deg,#ffffff,#eff6ff);
    border:1px solid #93c5fd;
    border-radius:18px;
    padding:12px 14px;
    text-align:center;
    min-width:140px;
    box-shadow:0 14px 28px rgba(37,99,235,.12);
}
.v18-timer-label{font-size:11px;color:#1d4ed8;font-weight:900;letter-spacing:.03em;}
.v18-timer-time{font-size:24px;font-weight:950;color:#082f49;line-height:1.15;}
.v18-timer-sub{font-size:11px;color:#0f766e;font-weight:850;}


/* V19: Papan pemuka lebih bersih dan tidak serabut */
.v19-section-title{
    display:flex;align-items:center;gap:12px;margin:28px 0 14px 0;
}
.v19-section-title h2{
    font-family:'Space Grotesk',sans-serif !important;
    margin:0 !important;
    font-size:1.85rem !important;
    color:#0f172a !important;
    background:none !important;
    -webkit-text-fill-color:#0f172a !important;
}
.v19-section-title span{
    background:#ecfdf5;
    color:#0f766e;
    border:1px solid #99f6e4;
    border-radius:999px;
    padding:7px 12px;
    font-size:.78rem;
    font-weight:900;
}
.v19-clean-card{
    background:rgba(255,255,255,.92);
    border:1px solid rgba(191,219,254,.85);
    border-radius:30px;
    padding:24px;
    box-shadow:0 22px 55px rgba(37,99,235,.10);
    backdrop-filter:blur(18px);
    margin-bottom:16px;
}
.v19-card-head{
    display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:16px;
}
.v19-card-head h3{
    font-family:'Space Grotesk',sans-serif !important;
    font-size:1.35rem !important;
    color:#082f49 !important;
    background:none !important;
    -webkit-text-fill-color:#082f49 !important;
    margin:0 !important;
}
.v19-card-head p{margin:6px 0 0 0;color:#64748b;font-weight:700;line-height:1.55;}
.v19-soft-badge{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:999px;padding:8px 12px;font-weight:900;font-size:.78rem;white-space:nowrap;}
.v19-stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0 18px 0;}
.v19-stat{
    background:linear-gradient(135deg,#ffffff 0%,#eff6ff 100%);
    border:1px solid #bfdbfe;
    border-radius:22px;
    padding:16px;
    box-shadow:0 12px 28px rgba(37,99,235,.06);
}
.v19-stat small{display:block;color:#64748b;font-weight:850;font-size:.78rem;margin-bottom:6px;}
.v19-stat strong{display:block;color:#082f49;font-size:1.15rem;font-weight:950;line-height:1.25;}
.v19-green-note{
    background:linear-gradient(135deg,#ecfdf5 0%,#f0fdfa 100%);
    border:1px solid #99f6e4;
    color:#0f766e;
    padding:14px 16px;
    border-radius:20px;
    font-weight:780;
    line-height:1.6;
    margin:14px 0 16px 0;
}
.v19-side-card{
    background:linear-gradient(135deg,#082f49 0%,#1d4ed8 55%,#0f766e 100%);
    border-radius:30px;
    color:white;
    padding:26px;
    box-shadow:0 24px 58px rgba(37,99,235,.20);
    margin-bottom:14px;
    min-height:250px;
}
.v19-side-card h3{color:white !important;-webkit-text-fill-color:white !important;background:none !important;margin:0 0 10px 0 !important;font-family:'Space Grotesk',sans-serif !important;font-size:1.45rem !important;}
.v19-side-card p{color:#dbeafe;font-weight:750;line-height:1.65;margin-bottom:16px;}
.v19-side-item{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:12px 14px;margin:10px 0;color:#eff6ff;font-weight:780;}
.v19-model-pill{background:#e0f2fe;border:1px solid #93c5fd;border-radius:18px;padding:13px 15px;color:#075985;font-weight:850;margin-bottom:14px;}
.v19-divider{height:1px;background:linear-gradient(90deg,transparent,#bfdbfe,transparent);margin:18px 0;}

/* Buang elemen kosong yang terjadi daripada HTML wrapper Streamlit */
.v18-training-panel:empty,
.v12-card:empty,
.v12-question-card:empty,
.v12-tutor-card:empty,
.v12-result-card:empty{
    display:none !important;
    padding:0 !important;
    margin:0 !important;
    border:0 !important;
    box-shadow:none !important;
}

/* Pemasa v19 */
.v19-timer-shell{
    background:linear-gradient(135deg,#ffffff 0%,#eff6ff 55%,#ecfdf5 100%);
    border:1px solid #93c5fd;
    border-radius:22px;
    padding:14px 18px;
    box-shadow:0 16px 34px rgba(37,99,235,.12);
    text-align:center;
    min-height:72px;
}
.v19-timer-label{font-size:.76rem;color:#1d4ed8;font-weight:950;letter-spacing:.05em;text-transform:uppercase;}
.v19-timer-time{font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:900;color:#082f49;line-height:1;margin-top:5px;}
.v19-timer-sub{font-size:.78rem;color:#0f766e;font-weight:850;margin-top:4px;}

@media (max-width:900px){.v19-stat-grid{grid-template-columns:1fr;}}


@media (max-width: 900px){
    .v12-smart-strip{grid-template-columns:1fr;}
    .v12-feature-grid{grid-template-columns:1fr;}
    .v12-hero-title{font-size:2.35rem;}
    .v12-topbar{position:relative;top:0;flex-direction:column;align-items:flex-start;}
    .v12-result-stats{grid-template-columns:1fr;}
    .v18-count-grid{grid-template-columns:1fr;}
}


/* ============================================================
   V20 REFINEMENT: dashboard premium, input color consistency,
   timer card and removal of empty pill/border artifacts
   ============================================================ */
.v19-section-title{display:none !important;}
.element-container:has(.v20-hide-empty){display:none !important;}
[data-testid="stDecoration"]{display:none !important;}

/* Override form widget accents to blue/teal, including plus-minus and dropdown borders */
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
.stNumberInput input,
.stTextInput input{
    border:1.5px solid rgba(37,99,235,.25) !important;
    background:#f8fbff !important;
    border-radius:15px !important;
}
[data-baseweb="select"]:focus-within > div,
[data-testid="stNumberInput"]:focus-within input,
.stTextInput input:focus{
    border-color:#14b8a6 !important;
    box-shadow:0 0 0 4px rgba(20,184,166,.14) !important;
}
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button:hover{
    background:linear-gradient(135deg,#2563eb,#0f766e) !important;
    color:#ffffff !important;
    border:0 !important;
}
[data-testid="stSlider"] [role="slider"]{
    background:#14b8a6 !important;
    border:4px solid #dffdf6 !important;
    box-shadow:0 8px 20px rgba(20,184,166,.35) !important;
}

.v20-dashboard-wrap{
    display:grid;
    grid-template-columns:1fr 330px;
    gap:20px;
    align-items:stretch;
    margin-top:18px;
}
.v20-learning-card{
    background:rgba(255,255,255,.94);
    border:1px solid rgba(147,197,253,.70);
    border-radius:30px;
    padding:24px 26px;
    box-shadow:0 24px 58px rgba(37,99,235,.12);
    position:relative;
    overflow:hidden;
}
.v20-learning-card:before{
    content:"";
    position:absolute;
    right:-90px;top:-120px;
    width:260px;height:260px;border-radius:50%;
    background:radial-gradient(circle,rgba(20,184,166,.20),transparent 65%);
}
.v20-card-kicker{
    display:inline-flex;align-items:center;gap:7px;
    padding:7px 12px;
    border-radius:999px;
    background:#ecfdf5;
    border:1px solid #99f6e4;
    color:#0f766e;
    font-size:.78rem;
    font-weight:900;
    margin-bottom:12px;
}
.v20-card-title{
    font-family:'Space Grotesk',sans-serif;
    color:#082f49;
    font-size:1.65rem;
    line-height:1.1;
    font-weight:900;
    margin:0 0 8px 0;
    letter-spacing:-.035em;
}
.v20-card-desc{color:#475569;font-weight:700;line-height:1.62;margin-bottom:18px;}
.v20-flow{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    margin-top:14px;
}
.v20-flow-step{
    background:linear-gradient(135deg,#f8fbff,#eff6ff);
    border:1px solid #bfdbfe;
    border-radius:20px;
    padding:14px;
    min-height:100px;
}
.v20-flow-step b{display:block;color:#082f49;font-size:.95rem;margin-bottom:5px;}
.v20-flow-step span{display:block;color:#64748b;font-size:.82rem;font-weight:750;line-height:1.4;}
.v20-hero-mini{
    background:linear-gradient(135deg,#082f49,#1d4ed8 55%,#0f766e);
    color:white;
    border-radius:30px;
    padding:24px;
    box-shadow:0 24px 60px rgba(37,99,235,.20);
    min-height:100%;
    position:relative;overflow:hidden;
}
.v20-hero-mini:after{
    content:"x²  ∑  π";
    position:absolute;right:18px;bottom:12px;
    color:rgba(255,255,255,.10);
    font-family:'Space Grotesk';font-size:2.2rem;font-weight:900;
}
.v20-hero-mini h3{color:white !important;-webkit-text-fill-color:white !important;background:none !important;margin:0 0 10px 0 !important;font-size:1.6rem !important;}
.v20-hero-mini p{color:#dbeafe;font-weight:700;line-height:1.62;}
.v20-side-metric{
    background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.20);
    border-radius:18px;
    padding:13px 14px;
    margin-top:10px;
    color:#eff6ff;
    font-weight:800;
}
.v20-side-metric small{display:block;color:#bfdbfe;font-weight:800;margin-bottom:3px;}
.v20-side-metric b{font-size:1.15rem;color:white;}
.v20-settings-title{
    font-family:'Space Grotesk',sans-serif;
    color:#082f49;
    font-size:1.3rem;
    font-weight:900;
    margin:0 0 4px 0;
}
.v20-settings-sub{color:#64748b;font-weight:700;line-height:1.55;margin-bottom:14px;}
.v20-summary-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    margin:14px 0;
}
.v20-summary-card{
    background:#ffffff;
    border:1px solid #dbeafe;
    border-radius:19px;
    padding:13px 14px;
    box-shadow:0 10px 26px rgba(37,99,235,.06);
}
.v20-summary-card small{display:block;color:#64748b;font-size:.76rem;font-weight:850;margin-bottom:4px;}
.v20-summary-card b{display:block;color:#082f49;font-size:1.06rem;line-height:1.2;font-weight:950;}
.v20-adapt-note{
    margin:12px 0 16px 0;
    padding:13px 15px;
    background:linear-gradient(135deg,#ecfdf5,#f0fdfa);
    border:1px solid #99f6e4;
    border-radius:18px;
    color:#0f766e;
    font-weight:800;
    line-height:1.55;
}
.v20-actions{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;}

.v20-timer-card{
    width:100%;
    min-height:96px;
    box-sizing:border-box;
    border-radius:24px;
    padding:16px 18px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px;
    background:linear-gradient(135deg,#082f49 0%,#1d4ed8 58%,#14b8a6 100%);
    color:#ffffff;
    box-shadow:0 20px 42px rgba(37,99,235,.23);
    border:1px solid rgba(255,255,255,.22);
}
.v20-timer-left{display:flex;align-items:center;gap:12px;}
.v20-timer-icon{width:46px;height:46px;border-radius:16px;background:rgba(255,255,255,.16);display:flex;align-items:center;justify-content:center;font-size:24px;}
.v20-timer-label{font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:#dbeafe;}
.v20-timer-sub{font-size:12px;font-weight:800;color:#ccfbf1;margin-top:3px;}
.v20-timer-time{font-family:'Space Grotesk',Arial,sans-serif;font-size:2.25rem;font-weight:900;letter-spacing:-.04em;line-height:1;}
.v20-timer-warning .v20-timer-time{color:#fecaca;}


/* Penambahbaikan v21: bar maklumat kuiz sebaris dan butang jawapan lebih premium */
.v21-question-surface{
    background:rgba(255,255,255,.92);
    border:1px solid rgba(191,219,254,.8);
    border-radius:30px;
    padding:22px 24px;
    box-shadow:0 22px 55px rgba(37,99,235,.13);
}
.v21-answer-action{
    margin-top:18px;
    padding-top:18px;
    border-top:1px solid rgba(191,219,254,.78);
}
.v21-answer-action + div button,
.v21-long-submit button{
    min-height:52px !important;
    border-radius:20px !important;
    font-size:.96rem !important;
    letter-spacing:.01em !important;
    background:linear-gradient(135deg,#1d4ed8 0%,#0f766e 100%) !important;
    box-shadow:0 16px 35px rgba(37,99,235,.25) !important;
}

@media(max-width:900px){
    .v20-dashboard-wrap{grid-template-columns:1fr;}
    .v20-flow,.v20-summary-grid,.v20-actions{grid-template-columns:1fr;}
}



/* ============================================================
   V24 CLEAN QUIZ + ADVANCED PERFORMANCE UI
   - removes blank bordered wrappers
   - cleaner prototype-style quiz layout
   - richer performance dashboard
   ============================================================ */
.v24-quiz-hero{
    background:linear-gradient(135deg,#ffffff 0%,#eff6ff 55%,#ecfdf5 100%);
    border:1px solid rgba(147,197,253,.85);
    border-radius:26px;
    padding:18px 20px;
    box-shadow:0 18px 42px rgba(37,99,235,.10);
    margin-bottom:18px;
}
.v24-quiz-hero-row{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;}
.v24-quiz-user{font-weight:950;color:#082f49;font-size:1.02rem;}
.v24-quiz-sub{font-size:.84rem;color:#475569;font-weight:750;margin-top:3px;}
.v24-progress-label{display:flex;justify-content:space-between;margin:12px 0 7px 0;font-size:.78rem;font-weight:950;color:#1d4ed8;}
.v24-progress-track{height:11px;background:#dbeafe;border-radius:999px;overflow:hidden;}
.v24-progress-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#2563eb,#14b8a6);}
.v24-question-box{
    background:rgba(255,255,255,.92);
    border:1px solid rgba(191,219,254,.85);
    border-radius:28px;
    padding:22px 24px;
    box-shadow:0 20px 46px rgba(37,99,235,.10);
    margin-bottom:16px;
}
.v24-question-title{font-size:1.22rem;font-weight:900;color:#0f172a;line-height:1.55;margin:16px 0 2px 0;}
.v24-answer-label{font-weight:900;color:#1e3a8a;font-size:.88rem;margin:14px 0 8px 0;}
.v24-tutor-box{
    background:rgba(255,255,255,.92);
    border:1px solid rgba(191,219,254,.85);
    border-radius:28px;
    padding:20px;
    box-shadow:0 20px 46px rgba(37,99,235,.10);
    margin-bottom:16px;
}
.v24-tutor-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;}
.v24-tutor-title{font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:950;color:#082f49;}
.v24-tutor-note{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;border-radius:18px;padding:13px 14px;font-weight:750;line-height:1.55;font-size:.86rem;margin:12px 0;}
.v24-tutor-divider{height:1px;background:linear-gradient(90deg,transparent,#bfdbfe,transparent);margin:16px 0;}
.v24-feedback-card{border-radius:22px;padding:14px 16px;margin:18px 0 4px 0;font-weight:780;line-height:1.55;}
.v24-feedback-ok{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;}
.v24-feedback-warn{background:#fff7ed;border:1px solid #fed7aa;color:#c2410c;}
.v24-results-wrap{
    background:linear-gradient(135deg,#ffffff,#eff6ff 55%,#ecfdf5);
    border:1px solid rgba(147,197,253,.8);
    border-radius:32px;
    padding:28px;
    box-shadow:0 24px 60px rgba(37,99,235,.12);
    text-align:center;
}
.v24-performance-hero{
    background:linear-gradient(135deg,#082f49 0%,#1d4ed8 55%,#14b8a6 100%);
    color:white;
    border-radius:32px;
    padding:28px;
    box-shadow:0 26px 64px rgba(37,99,235,.22);
    margin-bottom:20px;
    position:relative;
    overflow:hidden;
}
.v24-performance-hero:after{content:'∑  π  x²';position:absolute;right:24px;bottom:10px;color:rgba(255,255,255,.10);font-family:'Space Grotesk';font-size:3rem;font-weight:950;}
.v24-performance-hero h2{color:#fff !important;-webkit-text-fill-color:#fff !important;background:none !important;margin:0 0 8px 0 !important;font-family:'Space Grotesk',sans-serif !important;font-size:2rem !important;}
.v24-performance-hero p{color:#dbeafe;font-weight:750;line-height:1.62;max-width:760px;margin:0;}
.v24-metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 20px 0;}
.v24-metric-card{background:rgba(255,255,255,.94);border:1px solid #bfdbfe;border-radius:24px;padding:18px;box-shadow:0 16px 36px rgba(37,99,235,.08);}
.v24-metric-card small{display:block;color:#64748b;font-weight:900;font-size:.76rem;margin-bottom:6px;}
.v24-metric-card strong{display:block;color:#082f49;font-size:1.55rem;font-weight:950;line-height:1.15;}
.v24-metric-card span{display:block;color:#0f766e;font-size:.78rem;font-weight:850;margin-top:6px;}
.v24-section-card{background:rgba(255,255,255,.92);border:1px solid rgba(191,219,254,.85);border-radius:28px;padding:20px;box-shadow:0 20px 46px rgba(37,99,235,.10);margin-bottom:16px;}
.v24-section-card h3{font-family:'Space Grotesk',sans-serif !important;color:#082f49 !important;-webkit-text-fill-color:#082f49 !important;background:none !important;margin:0 0 6px 0 !important;font-size:1.25rem !important;}
.v24-section-card p{color:#64748b;font-weight:720;line-height:1.55;margin:0 0 12px 0;}
.v24-soft-pill{display:inline-flex;align-items:center;gap:6px;background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;border-radius:999px;padding:7px 11px;font-weight:900;font-size:.78rem;margin:4px 6px 4px 0;}
@media(max-width:900px){.v24-metric-grid{grid-template-columns:1fr 1fr;}.v24-performance-hero h2{font-size:1.55rem !important;}}
@media(max-width:600px){.v24-metric-grid{grid-template-columns:1fr;}}



/* ============================================================
   V25 CLEANER QUIZ UI + FIXED CHAT PANEL
   ============================================================ */
.v25-compact-head{background:rgba(255,255,255,.92);border:1px solid rgba(191,219,254,.8);border-radius:24px;padding:16px 18px;box-shadow:0 18px 42px rgba(37,99,235,.08);margin-bottom:14px;}
.v25-head-row{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;}
.v25-head-title{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:950;color:#082f49;}
.v25-head-sub{font-size:.82rem;color:#64748b;font-weight:780;margin-top:2px;}
.v25-progress-mini{height:9px;background:#dbeafe;border-radius:999px;overflow:hidden;margin-top:12px;}
.v25-progress-mini-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#2563eb,#14b8a6);}
.v25-question-panel{background:rgba(255,255,255,.94);border:1px solid rgba(191,219,254,.86);border-radius:26px;padding:22px 24px;box-shadow:0 22px 52px rgba(37,99,235,.10);margin-top:12px;}
.v25-question-title{color:#0f172a;font-size:1.18rem;font-weight:950;line-height:1.55;margin:0 0 16px 0;}
.v25-question-line{height:1px;background:linear-gradient(90deg,#dbeafe,transparent);margin:16px 0;}
.v25-answer-note{font-size:.84rem;color:#1d4ed8;font-weight:850;margin-bottom:10px;}
.v25-submit-space{margin-top:16px;}
.v25-tutor-panel{background:rgba(255,255,255,.95);border:1px solid rgba(191,219,254,.9);border-radius:26px;padding:18px;box-shadow:0 22px 52px rgba(37,99,235,.10);position:sticky;top:96px;}
.v25-tutor-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;}
.v25-tutor-title{font-family:'Space Grotesk',sans-serif;font-size:1.16rem;font-weight:950;color:#082f49;}
.v25-tutor-desc{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;border-radius:16px;padding:12px 13px;font-weight:800;line-height:1.45;font-size:.84rem;margin-bottom:12px;}
.v25-chat-box{height:285px;overflow-y:auto;background:linear-gradient(135deg,#f8fbff,#f0fdfa);border:1px solid #dbeafe;border-radius:20px;padding:12px;margin:12px 0;}
.v25-chat-box::-webkit-scrollbar{width:7px;}
.v25-chat-box::-webkit-scrollbar-thumb{background:#bfdbfe;border-radius:99px;}
.v25-chat-msg{display:flex;gap:8px;margin-bottom:10px;align-items:flex-start;}
.v25-chat-msg.user{justify-content:flex-end;}
.v25-avatar{width:30px;height:30px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:15px;flex:0 0 30px;}
.v25-avatar.bot{background:#f59e0b;color:white;}
.v25-avatar.user{background:#2563eb;color:white;order:2;}
.v25-bubble{max-width:86%;padding:10px 12px;border-radius:16px;font-size:.86rem;line-height:1.55;font-weight:650;}
.v25-bubble.bot{background:#ffffff;border:1px solid #bfdbfe;color:#0f172a;border-top-left-radius:6px;}
.v25-bubble.user{background:linear-gradient(135deg,#2563eb,#0f766e);color:white;border-top-right-radius:6px;}
.v25-tutor-actions{display:grid;grid-template-columns:1fr;gap:10px;margin:10px 0 12px 0;}
.v25-feedback{border-radius:18px;padding:13px 14px;margin-top:14px;font-weight:760;line-height:1.5;}
.v25-feedback.ok{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;}
.v25-feedback.warn{background:#fff7ed;border:1px solid #fed7aa;color:#c2410c;}
@media(max-width:900px){.v25-tutor-panel{position:relative;top:0;}.v25-chat-box{height:240px;}}


/* ============================================================
   V26 REFINEMENT: quiz lebih kemas, pilihan jawapan penuh,
   tutor chat fixed box, dan popup petunjuk.
   ============================================================ */
.v26-page-note{display:none !important;}
.v25-compact-head{
    background:rgba(255,255,255,.94) !important;
    border:1px solid rgba(147,197,253,.70) !important;
    border-radius:28px !important;
    padding:18px 20px !important;
    box-shadow:0 18px 44px rgba(37,99,235,.09) !important;
    margin-bottom:18px !important;
}
.v25-question-panel{
    background:#ffffff !important;
    border:1px solid rgba(147,197,253,.70) !important;
    border-radius:26px !important;
    padding:24px 26px !important;
    box-shadow:0 18px 44px rgba(37,99,235,.08) !important;
    margin-top:16px !important;
}
.v25-question-title{font-size:1.12rem !important;line-height:1.6 !important;}
.v25-answer-note{background:#eff6ff;border:1px solid #bfdbfe;border-radius:16px;padding:11px 13px;color:#1d4ed8;font-weight:900;margin-top:8px;}

/* Jadikan pilihan jawapan lebih kemas dan penuh seperti prototaip */
div[data-testid="stRadio"] > div{gap:10px !important;}
div[data-testid="stRadio"] label{
    width:100% !important;
    min-height:54px !important;
    border-radius:16px !important;
    border:1px solid #bfdbfe !important;
    background:rgba(255,255,255,.96) !important;
    padding:12px 16px !important;
    box-shadow:0 8px 20px rgba(37,99,235,.055) !important;
}
div[data-testid="stRadio"] label:hover{
    border-color:#14b8a6 !important;
    background:#f0fdfa !important;
}
div[data-testid="stRadio"] label p{
    font-size:.94rem !important;
    line-height:1.45 !important;
    color:#0f172a !important;
    font-weight:650 !important;
}

.v26-tutor-card{
    background:#ffffff;
    border:1px solid rgba(147,197,253,.74);
    border-radius:28px;
    padding:18px;
    box-shadow:0 22px 56px rgba(37,99,235,.11);
    position:sticky;
    top:96px;
}
.v26-tutor-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;}
.v26-tutor-title{font-family:'Space Grotesk',sans-serif;font-size:1.22rem;font-weight:950;color:#082f49;}
.v26-tutor-desc{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;border-radius:16px;padding:12px 13px;font-weight:850;line-height:1.48;font-size:.86rem;margin-bottom:12px;}
.v26-api-pill{display:inline-flex;align-items:center;padding:7px 12px;border-radius:999px;background:#ecfdf5;color:#047857;border:1px solid #99f6e4;font-size:.78rem;font-weight:900;margin-bottom:12px;}
.v26-api-pill.warn{background:#fff7ed;color:#c2410c;border-color:#fed7aa;}
.v26-chat-title{font-size:.82rem;font-weight:950;color:#1e40af;margin:8px 0 8px 0;}
.v26-form-card{background:rgba(255,255,255,.90);border:1px solid #bfdbfe;border-radius:22px;padding:14px 15px;box-shadow:0 14px 32px rgba(37,99,235,.07);margin-top:14px;}
.v26-hint-note{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:16px;padding:11px 13px;font-weight:800;font-size:.84rem;line-height:1.45;margin-top:10px;}
@media(max-width:900px){.v26-tutor-card{position:relative;top:0;}}


/* V33 FINAL: Tooltip kelabu + dropdown penuh + elak option jadi S.. / K.. */

/* Tooltip/help icon (?) kecil kelabu macam Streamlit asal */
[data-testid="stTooltipHoverTarget"] {
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    width:18px !important;
    height:18px !important;
    min-width:18px !important;
    min-height:18px !important;
    max-width:18px !important;
    max-height:18px !important;
    margin-left:6px !important;
    padding:0 !important;
    border-radius:50% !important;
    background:transparent !important;
    border:0 !important;
    box-shadow:none !important;
}
[data-testid="stTooltipHoverTarget"] button,
[data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stTooltipHoverTarget"] button:focus,
[data-testid="stTooltipHoverTarget"] button:active {
    width:18px !important;
    height:18px !important;
    min-width:18px !important;
    min-height:18px !important;
    max-width:18px !important;
    max-height:18px !important;
    padding:0 !important;
    margin:0 !important;
    border-radius:50% !important;
    background:transparent !important;
    background-image:none !important;
    border:0 !important;
    box-shadow:none !important;
    outline:none !important;
    color:#8b97a3 !important;
    transform:none !important;
}
[data-testid="stTooltipHoverTarget"] svg,
[data-testid="stTooltipHoverTarget"] button svg {
    width:17px !important;
    height:17px !important;
    color:#8b97a3 !important;
    stroke:#8b97a3 !important;
    fill:none !important;
}

/* Style plus/minus hanya untuk number input, bukan tooltip */
[data-testid="stNumberInput"] button {
    background:linear-gradient(135deg,#2563eb,#0f766e) !important;
    color:#ffffff !important;
    border:0 !important;
    box-shadow:none !important;
}

/* Fix dropdown list yang jadi S.. / K.. */
div[data-baseweb="popover"] {
    min-width:260px !important;
    width:auto !important;
    max-width:520px !important;
}
div[data-baseweb="popover"] [role="listbox"] {
    min-width:260px !important;
    width:auto !important;
    max-width:520px !important;
    overflow-x:hidden !important;
}
div[data-baseweb="popover"] [role="option"] {
    min-height:38px !important;
    padding:9px 14px !important;
    white-space:normal !important;
    overflow:visible !important;
    text-overflow:clip !important;
    line-height:1.35 !important;
}
div[data-baseweb="popover"] [role="option"] * {
    white-space:normal !important;
    overflow:visible !important;
    text-overflow:clip !important;
    max-width:none !important;
    width:auto !important;
    line-height:1.35 !important;
}

/* Pastikan teks pilihan yang dipilih dalam kotak select tidak terpotong pelik */
[data-baseweb="select"] div,
[data-baseweb="select"] span {
    white-space:nowrap !important;
    text-overflow:ellipsis !important;
}



/* V34 FINAL FIX: tooltip/help icon (?) inside number input must stay grey circle, not blue square */
[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"],
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"],
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] {
    display:inline-flex !important;
    align-items:center !important;
    justify-content:center !important;
    width:18px !important;
    height:18px !important;
    min-width:18px !important;
    min-height:18px !important;
    max-width:18px !important;
    max-height:18px !important;
    padding:0 !important;
    margin-left:6px !important;
    border-radius:50% !important;
    background:transparent !important;
    border:0 !important;
    box-shadow:none !important;
}

[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] button,
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"] button,
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] button,
[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] button:focus,
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"] button:focus,
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] button:focus {
    width:18px !important;
    height:18px !important;
    min-width:18px !important;
    min-height:18px !important;
    max-width:18px !important;
    max-height:18px !important;
    padding:0 !important;
    margin:0 !important;
    border-radius:50% !important;
    background:transparent !important;
    background-image:none !important;
    border:0 !important;
    box-shadow:none !important;
    outline:none !important;
    color:#8b97a3 !important;
    transform:none !important;
}

[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] svg,
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"] svg,
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] svg,
[data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] button svg,
[data-testid="stSelectbox"] [data-testid="stTooltipHoverTarget"] button svg,
[data-testid="stTextInput"] [data-testid="stTooltipHoverTarget"] button svg {
    width:17px !important;
    height:17px !important;
    color:#8b97a3 !important;
    stroke:#8b97a3 !important;
    fill:none !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Pemuatan Model dan Data 
# ------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuatkan model bundle: {e}")
    return None

@st.cache_data
def load_questions():
    if not QUESTION_PATH.exists():
        st.error(f"Fail '{QUESTION_PATH}' tidak ditemui dalam folder 'data'.")
        return pd.DataFrame()

    required = ["QuestionID", "Topic", "Difficulty", "QuestionText", "OptionA", "OptionB", "OptionC", "OptionD", "CorrectAnswer"]

    try:
        raw = pd.read_excel(QUESTION_PATH, sheet_name=0, header=None)
        header_row = None
        for i in range(min(15, len(raw))):
            values = [str(v).strip() for v in raw.iloc[i].tolist() if pd.notna(v)]
            if all(col in values for col in ["QuestionID", "Topic", "Difficulty", "QuestionText"]):
                header_row = i
                break

        if header_row is not None:
            columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
            q = raw.iloc[header_row + 1:].copy()
            q.columns = columns
        else:
            q = pd.read_excel(QUESTION_PATH, sheet_name=0)
            q.columns = [str(c).strip() for c in q.columns]

        q = q.loc[:, ~pd.Index(q.columns).astype(str).str.contains("^Unnamed", na=False)]
        q.columns = [str(c).strip() for c in q.columns]

        missing = [c for c in required if c not in q.columns]
        if missing:
            st.error("Struktur Bank Soalan tidak lengkap. Lajur yang hilang: " + ", ".join(missing))
            return pd.DataFrame()

        if "Hint" not in q.columns:
            q["Hint"] = ""
        if "MaxTime" not in q.columns:
            q["MaxTime"] = 60

        q = q.dropna(subset=["QuestionText", "OptionA", "OptionB", "OptionC", "OptionD", "CorrectAnswer"]).copy()
        q["Difficulty"] = q["Difficulty"].astype(str).str.strip().str.lower().map({
            "rendah": "Rendah", "sederhana": "Sederhana", "tinggi": "Tinggi"
        })
        q = q.dropna(subset=["Difficulty"]).copy()
        q["CorrectAnswer"] = q["CorrectAnswer"].astype(str).str.strip().str.upper()
        q["QuestionID"] = q["QuestionID"].astype(str).str.strip()
        q["Topic"] = q["Topic"].astype(str).str.strip()
        q["MaxTime"] = pd.to_numeric(q["MaxTime"], errors="coerce").fillna(60).astype(int)
        return q.reset_index(drop=True)
    except Exception as e:
        st.error(f"Gagal memuatkan Bank Soalan: {e}")
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    import json
    summary_path = Path("models") / "metrics_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        final_model = summary.get("final_model", "Hybrid RF-DNN")
        rows = summary.get("results", [])
        for row in rows:
            if row.get("Model") == final_model and row.get("Dataset") == "Test":
                return {"test": {"accuracy": float(row.get("Accuracy", 0))}, "summary": summary}
        return {"test": {"accuracy": 0}, "summary": summary}
    old_path = Path("models") / "metrics.json"
    if old_path.exists():
        with open(old_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ------------------------------------------------------------
# Ramalan Pembelajaran Mesin
# ------------------------------------------------------------


def predict_competency(profile_dict):
    bundle = load_model_bundle()
    feature_cols = [
        "StudyHours", "Attendance", "Resources", "Extracurricular", "Motivation",
        "Internet", "Gender", "Age", "LearningStyle", "OnlineCourses",
        "Discussions", "AssignmentCompletion", "ExamScore", "EduTech", "StressLevel"
    ]

    def sandaran_prediction():
        score = float(profile_dict.get("ExamScore", 50))
        if score < 40:
            return 0, "Rendah", 0.60, [0.60, 0.30, 0.10]
        elif score < 75:
            return 1, "Sederhana", 0.60, [0.20, 0.60, 0.20]
        else:
            return 2, "Tinggi", 0.60, [0.10, 0.20, 0.70]

    if not bundle:
        return sandaran_prediction()

    try:
        # Model dilatih menggunakan dataset proksi yang umur minimumnya bermula sekitar 18 tahun.
        # Oleh itu, umur pelajar F4/F5 yang bawah 18 tahun distandardkan kepada 18 untuk ramalan model
        # supaya input tidak terlalu jauh daripada julat data latihan. Umur asal masih kekal dalam rekod profil sistem.
        model_profile = dict(profile_dict)
        try:
            if float(model_profile.get("Age", 18)) < 18:
                model_profile["Age"] = 18
        except Exception:
            model_profile["Age"] = 18

        df_in = pd.DataFrame([model_profile])

        # Latest local deployment bundle: preprocessor + RF probabilities + hybrid MLP/DNN.
        if isinstance(bundle, dict) and "preprocessor" in bundle and "rf_model" in bundle and "hybrid_model" in bundle:
            X_proc = bundle["preprocessor"].transform(df_in)
            rf_probs = bundle["rf_model"].predict_proba(X_proc)
            try:
                from scipy.sparse import hstack, csr_matrix
                X_hybrid = hstack([csr_matrix(X_proc), csr_matrix(rf_probs)])
                if hasattr(X_hybrid, "toarray"):
                    X_hybrid = X_hybrid.toarray()
            except Exception:
                X_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)
                X_hybrid = np.hstack([X_dense, rf_probs])
            pred_prob = bundle["hybrid_model"].predict_proba(X_hybrid)[0]
            pred_class = int(np.argmax(pred_prob))
            conf = float(pred_prob[pred_class])
            label_map = bundle.get("level_decoding", LEVEL_TEXT)
            return pred_class, LEVEL_TEXT[pred_class], conf, list(pred_prob)

        # Older local bundle support: imputer + scaler + RF + DNN model.
        if isinstance(bundle, dict) and "imputer" in bundle and "scaler" in bundle:
            input_row = [[model_profile[c] for c in feature_cols]]
            X_imp = bundle["imputer"].transform(input_row)
            X_sc = bundle["scaler"].transform(X_imp)
            rf_key = "rf_model" if "rf_model" in bundle else "rf"
            dnn_key = "dnn_model" if "dnn_model" in bundle else "dnn"
            rf_probs = bundle[rf_key].predict_proba(X_sc)
            if dnn_key in bundle:
                # Some older bundles train DNN directly on RF probabilities only.
                try:
                    pred_prob = bundle[dnn_key].predict_proba(rf_probs)[0]
                except Exception:
                    X_hybrid = np.hstack([X_sc, rf_probs])
                    pred_prob = bundle[dnn_key].predict_proba(X_hybrid)[0]
            else:
                pred_prob = rf_probs[0]
            pred_class = int(np.argmax(pred_prob))
            conf = float(pred_prob[pred_class])
            return pred_class, LEVEL_TEXT[pred_class], conf, list(pred_prob)

        return sandaran_prediction()

    except Exception as e:
        st.warning(f"Ralat ramalan model: {e}. Sistem menggunakan logik sandaran sementara.")
        return sandaran_prediction()

# ------------------------------------------------------------
# Logik Enjin Adaptif FSRS

# ------------------------------------------------------------
def clamp(val, mn, mx):
    return max(mn, min(mx, val))

def apply_fsrs_and_get_next_level(current_level, is_correct, time_taken, max_time, used_hint, topic_name, user_id):
    state = get_fsrs_topic_state(user_id, topic_name)
    d_prev = float(state.get("fsrs_difficulty", state.get("difficulty", 5.0)))
    s_prev = float(state.get("fsrs_stability", state.get("stability", 1.0)))
    ratio = float(time_taken) / max(1.0, float(max_time))

    if not is_correct:
        grade = 1
        new_level = max(0, int(current_level) - 1)
        reason = "Jawapan belum tepat, jadi sistem akan beri soalan yang lebih mudah untuk bantu kukuhkan semula asas."
    elif ratio > 1.00 or used_hint:
        grade = 2
        new_level = int(current_level)
        reason = "Jawapan betul, tetapi sistem kekalkan aras dahulu kerana masa menjawab agak lama atau bantuan telah digunakan."
    elif ratio <= 0.50 and not used_hint:
        grade = 4
        new_level = min(2, int(current_level) + 1)
        reason = "Jawapan betul dan pantas, jadi sistem akan naikkan aras soalan seterusnya."
    else:
        grade = 3
        new_level = min(2, int(current_level) + 1)
        reason = "Jawapan betul dalam masa yang sesuai, jadi sistem akan naikkan aras secara terkawal."

    difficulty_change = {1: 1.20, 2: 0.55, 3: -0.25, 4: -0.70}[grade]
    if ratio > 1:
        difficulty_change += min(1.0, (ratio - 1) * 0.5)
    if used_hint:
        difficulty_change += 0.35

    d_new = clamp(d_prev + difficulty_change, 1.0, 10.0)
    stability_multiplier = {1: 0.55, 2: 1.05, 3: 1.65, 4: 2.25}[grade]
    if used_hint:
        stability_multiplier *= 0.85
    if ratio > 1:
        stability_multiplier *= 0.80

    s_new = clamp(s_prev * stability_multiplier + 0.25, 0.50, 30.0)
    retrievability = clamp(math.exp(-ratio / max(0.50, s_new)), 0.0, 1.0)

    grade_text_map = {1: "Ulang", 2: "Sukar", 3: "Baik", 4: "Mudah"}
    return {
        "new_level": new_level,
        "reason": reason,
        "grade": grade,
        "grade_text": grade_text_map[grade],
        "response_ratio": ratio,
        "difficulty": round(d_new, 3),
        "stability": round(s_new, 3),
        "retrievability": round(retrievability, 3),
        "fsrs": {
            "grade": grade,
            "grade_text": grade_text_map[grade],
            "difficulty": f"{d_new:.2f}",
            "stability": f"{s_new:.2f} hari",
            "retrievability": f"{retrievability:.2%}"
        }
    }

# ------------------------------------------------------------
# Lapisan Tutor Pintar Sokratik

# ------------------------------------------------------------
def openai_is_configured():
    try:
        k = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        return bool(k)
    except Exception:
        return False

SYSTEM_PROMPT_SOCRATIC = """
Anda ialah Tutor Pintar Sokratik bagi Sistem Pembelajaran Adaptif Matematik untuk Tingkatan 4 dan Tingkatan 5 di Malaysia. Tahap penguasaan semasa pelajar ialah {current_level_text}.

Gaya perbualan:
- Gunakan Bahasa Melayu sepenuhnya. Jangan campur Bahasa Inggeris kecuali istilah teknikal yang tiada padanan sesuai.
- Berbual secara mesra, ringkas dan menyokong pembelajaran.
- Jika pelajar memberi salam atau bertanya perkara umum, jawab secara semula jadi.
- Jika pelajar berkongsi langkah kerja, pengiraan atau formula, semak langkah tersebut tanpa mendedahkan jawapan akhir.

Gaya pengajaran:
- Bimbing pelajar menggunakan soalan berpandu dan petunjuk kecil.
- Berikan konsep, formula atau langkah seterusnya yang sesuai.
- Laraskan penerangan mengikut tahap penguasaan pelajar.

Peraturan penting:
1. Jangan dedahkan jawapan akhir bagi soalan kuiz semasa.
2. Jangan nyatakan pilihan A, B, C atau D yang betul.
3. Jangan berikan jalan kerja penuh sehingga terus membawa kepada jawapan akhir.
4. Jika pelajar meminta jawapan akhir atau pilihan yang betul, tolak secara sopan dan beri langkah bantuan seterusnya.
5. Jika pelajar mencadangkan jawapan, jangan sahkan pilihan akhir. Boleh bantu semak arah pemikiran dan perkara yang perlu disemak.
6. Pastikan jawapan pendek, jelas dan membantu.
"""

def is_simple_greeting(student_message: str) -> bool:
    text = (student_message or "").strip().lower()
    greetings = {
        "hi", "hello", "hey", "hai", "hye", "helo", "salam", "assalamualaikum", 
        "assalamualaikum hi", "good morning", "good afternoon", "good evening"
    }
    return text in greetings or (len(text.split()) <= 3 and any(g in text for g in ["hi", "hello", "hai", "hey", "salam"]))

def is_direct_answer_request(student_message: str) -> bool:
    text = (student_message or "").strip().lower()
    phrases = [
        "give me answer", "give answer", "what is the answer", "jawapan apa", "bagi jawapan", 
        "beri jawapan", "nak jawapan", "nak answer", "which pilihan", "pilihan apa", 
        "is it a", "is it b", "is it c", "is it d", "answer is a", "answer is b", 
        "answer is c", "answer is d", "jawapan a", "jawapan b", "jawapan c", "jawapan d"
    ]
    return any(p in text for p in phrases)

def should_count_ai_help(student_message: str) -> bool:
    if is_simple_greeting(student_message):
        return False
    return True

def sandaran_socratic_reply(question, student_message):
    msg = (student_message or "").strip().lower()
    if is_simple_greeting(msg):
        return "Hai! Saya Tutor Pintar anda. Ada bahagian dalam soalan matematik ini yang boleh saya bimbing?"
    if is_direct_answer_request(msg):
        return "Saya tidak boleh memberikan jawapan terus atau pilihan jawapan yang betul. Mari kita lihat formula asal atau cuba terangkan langkah pertama anda dahulu."
    return f"Petunjuk untuk soalan '{question[:20]}...': Cuba semak formula asas yang berkaitan atau senaraikan maklumat yang diberi dalam soalan."

def get_socratic_reply(question_row, student_message, current_level_text):
    """Tutor Pintar Sokratik dengan panggilan API yang lebih pantas.

    Penambahbaikan utama:
    1. Salam dan permintaan jawapan terus dijawab secara tempatan tanpa API.
    2. Prompt dipendekkan supaya API lebih cepat.
    3. Sejarah sembang dikurangkan kepada 2 mesej terakhir sahaja.
    4. Timeout dan max_retries ditetapkan supaya sistem tidak terlalu lama menunggu.
    5. Output dipendekkan supaya sesuai sebagai tutor kuiz.
    """
    question = str(question_row["QuestionText"])
    topic = str(question_row["Topic"])
    petunjuk = str(question_row.get("Hint", ""))
    options = {
        "A": str(question_row["OptionA"]),
        "B": str(question_row["OptionB"]),
        "C": str(question_row["OptionC"]),
        "D": str(question_row["OptionD"]),
    }

    # Balasan segera tanpa API untuk jimat masa dan kos.
    if is_simple_greeting(student_message):
        return "Hai! Saya Tutor Pintar anda. Beritahu bahagian mana dalam soalan ini yang anda mahu saya bantu."

    if is_direct_answer_request(student_message):
        return "Saya tidak boleh memberikan jawapan akhir atau pilihan yang betul. Namun, saya boleh bantu anda kenal pasti formula, maklumat diberi, dan langkah pertama yang sesuai."

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return "⚠️ Kunci API OpenAI tidak dikesan dalam folder sistem ini, jadi Tutor Pintar menggunakan mod sandaran tempatan.\n\n" + sandaran_socratic_reply(question, student_message)

    try:
        from openai import OpenAI

        # Timeout dan retry rendah supaya aplikasi tidak terasa tergantung.
        client = OpenAI(api_key=api_key, timeout=10.0, max_retries=1)
        model_name = "gpt-4o-mini"
        
        user_prompt = f"""
Topik: {topic}
Aras pelajar: {current_level_text}
Soalan: {question}
Pilihan: A) {options['A']} | B) {options['B']} | C) {options['C']} | D) {options['D']}
Petunjuk bank soalan: {petunjuk}
Mesej pelajar: {student_message}

Berikan bimbingan ringkas dalam Bahasa Melayu. Jangan dedahkan jawapan akhir atau pilihan betul.
"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT_SOCRATIC.format(current_level_text=current_level_text)}]

        # Hanya ambil 2 mesej terakhir untuk kurangkan token dan tingkatkan kelajuan.
        for msg in st.session_state.chat_messages[-2:]:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return sandaran_socratic_reply(question, student_message) + f"\n\n[Mod sandaran digunakan kerana ralat API OpenAI: {e}]"

# ------------------------------------------------------------
# State helpers
# ------------------------------------------------------------
def init_session_state():
    defaults = {
        "logged_in": False,
        "username": "",
        "user_id": None,
        "profile_ready": False,
        "predicted_level": None,
        "predicted_level_text": None,
        "confidence": None,
        "quiz_started": False,
        "quiz_questions": None,
        "quiz_index": 0,
        "current_level": None,
        "session_id": None,
        "score": 0.0,
        "correct_count": 0,
        "hints_used": 0,
        "used_hint_current": False,
        "hint_count_current": 0,
        "question_start_time": None,
        "last_feedback": None,
        "quiz_finished": False,
        "chat_messages": [],
        "nav_page": "Profil",
        "question_pool": None,
        "question_target_count": 0,
        "selected_topic": "",
        "show_hint_popup": False,
        "current_hint_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_quiz():
    st.session_state.quiz_started = False
    st.session_state.quiz_questions = None
    st.session_state.quiz_index = 0
    st.session_state.session_id = None
    st.session_state.score = 0.0
    st.session_state.correct_count = 0
    st.session_state.hints_used = 0
    st.session_state.used_hint_current = False
    st.session_state.hint_count_current = 0
    st.session_state.question_start_time = None
    st.session_state.last_feedback = None
    st.session_state.quiz_finished = False
    st.session_state.chat_messages = []
    st.session_state.question_pool = None
    st.session_state.question_target_count = 0
    st.session_state.selected_topic = ""

def get_adaptive_start_level():
    """Dapatkan aras semasa untuk sesi kuiz baharu.
    Keutamaan: aras adaptif dalam session_state -> aras akhir kuiz terakhir -> ramalan profil awal.
    Ini memastikan papan pemuka tidak kembali kepada aras permulaan lama selepas kuiz selesai.
    """
    level = st.session_state.get("current_level", None)
    if level is not None:
        try:
            return int(level), "Aras semasa"
        except Exception:
            pass

    try:
        if st.session_state.get("user_id") is not None:
            sessions = fetch_sessions(st.session_state.user_id)
            if sessions:
                latest_final = sessions[0].get("final_level")
                if latest_final is not None:
                    st.session_state.current_level = int(latest_final)
                    return int(latest_final), "Aras semasa"
    except Exception:
        pass

    fallback = st.session_state.get("predicted_level", 1)
    if fallback is None:
        fallback = 1
    return int(fallback), "Aras permulaan"

def go_to(page_name):
    st.session_state.nav_page = page_name

def get_current_question():
    if st.session_state.quiz_questions is not None and st.session_state.quiz_index < len(st.session_state.quiz_questions):
        return st.session_state.quiz_questions[st.session_state.quiz_index]
    return None

def get_question_id(question):
    return str(question.get("QuestionID", "")).strip()

def pilih_soalan_adaptif(pool_records, target_level, used_ids):
    """Pilih soalan seterusnya berdasarkan aras semasa pelajar.
    Keutamaan diberi kepada aras yang sepadan dengan tahap adaptif semasa,
    kemudian sistem akan menggunakan soalan baki jika aras itu sudah habis.
    """
    if not pool_records:
        return None
    used_ids = set(str(x).strip() for x in used_ids)
    remaining = [q for q in pool_records if get_question_id(q) not in used_ids]
    if not remaining:
        return None

    target_text = LEVEL_TEXT.get(int(target_level), "Sederhana")
    matched = [q for q in remaining if str(q.get("Difficulty", "")).strip() == target_text]
    pilihan = matched if matched else remaining
    return random.choice(pilihan)

# ------------------------------------------------------------
# Page components
# ------------------------------------------------------------
def app_topbar(section_title):
    user_badge = f"<span class='v12-user-pill'>👤 {st.session_state.username}</span>" if st.session_state.logged_in else ""
    st.markdown(f"""
    <div class="v12-topbar">
        <div class="v12-brand">
            <div class="v12-logo">📘</div>
            <div>
                <div class="v12-brand-title">Matematik Pintar SPM</div>
                <div class="v12-brand-sub">Pembelajaran adaptif Matematik Tingkatan 4 dan 5</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="v12-section-pill">{section_title}</span>
            {user_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav1, nav2, nav3, nav4, nav5 = st.columns(5)
    section = section_title.replace(" ", "_").lower()
    with nav1:
        if st.button("👤 Profil", key=f"top_prof_{section}", use_container_width=True): go_to("Profil")
    with nav2:
        if st.button("🏠 Papan Pemuka", key=f"top_dash_{section}", use_container_width=True): go_to("Papan Pemuka")
    with nav3:
        if st.button("🎯 Kuiz Adaptif", key=f"top_quiz_{section}", use_container_width=True): go_to("Kuiz Adaptif")
    with nav4:
        if st.button("📊 Prestasi", key=f"top_perf_{section}", use_container_width=True): go_to("Laporan Prestasi")
    with nav5:
        if st.button("🗄️ Pangkalan Data", key=f"top_db_{section}", use_container_width=True): go_to("Pangkalan Data")

def live_timer_component(start_time, max_time, key_suffix=""):
    start_ms = int(float(start_time) * 1000)
    max_s = int(max_time)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800;900&family=Space+Grotesk:wght@700;800;900&display=swap');
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
        * {{ box-sizing:border-box; }}
        .timer {{
            width:100%;
            min-height:96px;
            border-radius:24px;
            padding:16px 18px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:16px;
            background:linear-gradient(135deg,#082f49 0%,#1d4ed8 58%,#14b8a6 100%);
            color:#ffffff;
            box-shadow:0 20px 42px rgba(37,99,235,.23);
            border:1px solid rgba(255,255,255,.22);
            font-family:Inter,Arial,sans-serif;
        }}
        .left {{ display:flex;align-items:center;gap:12px; }}
        .icon {{ width:46px;height:46px;border-radius:16px;background:rgba(255,255,255,.16);display:flex;align-items:center;justify-content:center;font-size:24px; }}
        .label {{ font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:#dbeafe; }}
        .sub {{ font-size:12px;font-weight:800;color:#ccfbf1;margin-top:3px; }}
        .time {{ font-family:'Space Grotesk',Arial,sans-serif;font-size:2.25rem;font-weight:900;letter-spacing:-.04em;line-height:1; }}
        .warn .time {{ color:#fecaca; }}
        .warn .sub {{ color:#fee2e2; }}
    </style>
    </head>
    <body>
        <div id="timerBox{key_suffix}" class="timer">
            <div class="left">
                <div class="icon">⏱️</div>
                <div>
                    <div class="label">Pemasa Kuiz</div>
                    <div id="timerSub{key_suffix}" class="sub">Had masa: {max_s}s</div>
                </div>
            </div>
            <div id="timerText{key_suffix}" class="time">--:--</div>
        </div>
        <script>
            const start{key_suffix} = {start_ms};
            const max{key_suffix} = {max_s};
            function updateTimer{key_suffix}() {{
                const elapsed = Math.floor((Date.now() - start{key_suffix}) / 1000);
                const remaining = Math.max(0, max{key_suffix} - elapsed);
                const m = Math.floor(remaining / 60).toString().padStart(2,'0');
                const s = (remaining % 60).toString().padStart(2,'0');
                const text = document.getElementById('timerText{key_suffix}');
                const sub = document.getElementById('timerSub{key_suffix}');
                const box = document.getElementById('timerBox{key_suffix}');
                text.innerHTML = m + ':' + s;
                if (remaining <= 10 && remaining > 0) {{
                    box.classList.add('warn');
                    sub.innerHTML = 'Hampir tamat';
                }}
                if (remaining <= 0) {{
                    box.classList.add('warn');
                    sub.innerHTML = 'Masa tamat';
                }} else {{
                    setTimeout(updateTimer{key_suffix}, 1000);
                }}
            }}
            updateTimer{key_suffix}();
        </script>
    </body>
    </html>
    """
    components.html(html, height=118, scrolling=False)

def quiz_meta_timer_component(topic, difficulty, start_time, max_time, key_suffix=""):
    """Paparan topik, aras dan pemasa kecil dalam satu bar yang kemas."""
    start_ms = int(float(start_time) * 1000)
    max_s = int(max_time)
    topic_safe = html.escape(str(topic))
    difficulty_safe = html.escape(str(difficulty))
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800;900&family=Space+Grotesk:wght@700;800;900&display=swap');
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; font-family:Inter,Arial,sans-serif; }}
        * {{ box-sizing:border-box; }}
        .bar {{
            display:flex; align-items:center; gap:10px; flex-wrap:wrap;
            padding:0; width:100%; min-height:42px;
        }}
        .chip {{
            display:inline-flex; align-items:center; gap:7px;
            height:42px; padding:0 15px; border-radius:999px;
            border:1px solid #bfdbfe; background:rgba(255,255,255,.86);
            color:#1e40af; font-size:13px; font-weight:900;
            box-shadow:0 10px 22px rgba(37,99,235,.07);
            white-space:nowrap;
        }}
        .level {{ color:#0f766e; border-color:#99f6e4; background:#ecfdf5; }}
        .timer {{
            display:inline-flex; align-items:center; gap:8px;
            height:42px; padding:0 14px; border-radius:999px;
            background:linear-gradient(135deg,#082f49 0%,#1d4ed8 55%,#14b8a6 100%);
            color:#ffffff; font-weight:900;
            box-shadow:0 12px 26px rgba(37,99,235,.18);
            white-space:nowrap;
        }}
        .timer small {{ font-size:11px; opacity:.9; font-weight:900; letter-spacing:.02em; }}
        .timer b {{ font-family:'Space Grotesk',Arial,sans-serif; font-size:17px; letter-spacing:-.02em; }}
        .warn {{ background:linear-gradient(135deg,#7f1d1d 0%,#dc2626 60%,#f97316 100%); }}
    </style>
    </head>
    <body>
        <div class="bar">
            <div class="chip">📋 Topik: {topic_safe}</div>
            <div class="chip level">📊 Aras Soalan: {difficulty_safe}</div>
            <div id="timerBox{key_suffix}" class="timer"><span>⏱️</span><small>Pemasa</small><b id="timerText{key_suffix}">--:--</b></div>
        </div>
        <script>
            const start{key_suffix} = {start_ms};
            const max{key_suffix} = {max_s};
            function updateTimer{key_suffix}() {{
                const elapsed = Math.floor((Date.now() - start{key_suffix}) / 1000);
                const remaining = Math.max(0, max{key_suffix} - elapsed);
                const m = Math.floor(remaining / 60).toString().padStart(2,'0');
                const s = (remaining % 60).toString().padStart(2,'0');
                const text = document.getElementById('timerText{key_suffix}');
                const box = document.getElementById('timerBox{key_suffix}');
                text.innerHTML = m + ':' + s;
                if (remaining <= 10) box.classList.add('warn');
                if (remaining > 0) setTimeout(updateTimer{key_suffix}, 1000);
            }}
            updateTimer{key_suffix}();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=48, scrolling=False)

# ------------------------------------------------------------
# Pages functions
# ------------------------------------------------------------
def page_login_profile():
    app_topbar("Profil & Kemasukan")
    left, right = st.columns([0.9, 1.1], gap="medium")
    with left:
        st.markdown("""
        <div class="v12-hero-premium">
            <div class="v12-hero-content">
                <div class="v12-kicker">📘 Sistem pembelajaran adaptif untuk calon SPM</div>
                <h1 class="v12-hero-title">Matematik Pintar SPM</h1>
                <p class="v12-hero-desc">Belajar Matematik dengan laluan latihan yang berubah mengikut tahap penguasaan anda. Sistem ini menggabungkan ramalan Pembelajaran Mesin, kuiz adaptif, petunjuk pintar, pemasa masa nyata dan laporan prestasi dalam satu ruang pembelajaran yang moden.</p>
                <div>
                    <span class="v12-feature-chip">🧠 Ramalan Hibrid RF-DNN</span>
                    <span class="v12-feature-chip">🎯 Kuiz adaptif</span>
                    <span class="v12-feature-chip">💡 Petunjuk Pintar</span>
                    <span class="v12-feature-chip">📊 Laporan prestasi</span>
                </div>
                <div class="v12-feature-grid">
                    <div class="v12-feature-card"><b>Aras Pembelajaran</b><span>Sistem menentukan aras awal pelajar kepada Rendah, Sederhana atau Tinggi.</span></div>
                    <div class="v12-feature-card"><b>Latihan Fleksibel</b><span>Pilih topik tertentu atau campuran semua topik dan tetapkan bilangan soalan.</span></div>
                    <div class="v12-feature-card"><b>Tutor Pintar</b><span>Tutor memberi bimbingan tanpa mendedahkan jawapan akhir kuiz.</span></div>
                    <div class="v12-feature-card"><b>Rekod Kemajuan</b><span>Jawapan, masa, bantuan dan prestasi disimpan untuk laporan pembelajaran.</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with right:
        st.markdown('<div class="v12-card-title">Profil Pembelajaran Pelajar</div><div class="v12-card-sub">Masukkan nama dan butiran profil untuk memulakan klasifikasi awal Pembelajaran Mesin.</div>', unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            with st.form("login_form"):
                username = st.text_input("Nama Pelajar", placeholder="Contoh: Marsya Aneesa")
                submitted = st.form_submit_button("Masuk Sistem", type="primary")
                if submitted:
                    if not username.strip():
                        st.error("Sila masukkan nama pelajar.")
                    else:
                        init_db()
                        st.session_state.username = username.strip()
                        st.session_state.user_id = get_or_create_user(username.strip())
                        st.session_state.logged_in = True
                        st.success("Log masuk berjaya.")
                        st.rerun()
                return

        latest_prof = fetch_latest_profile(st.session_state.user_id)
        
        with st.form("profile_form"):
            cc1, cc2 = st.columns(2)
            with cc1:
                Gender = st.selectbox("Jantina", [0, 1], index=0, format_func=lambda x: "Perempuan" if x==0 else "Lelaki")
                Age = st.number_input("Umur (Tahun)", min_value=15, max_value=18, value=16, help="Umur sebenar pelajar. Untuk ramalan model, umur bawah 18 distandardkan kepada 18 kerana dataset latihan bermula pada umur 18 tahun.")
                StudyHours = st.number_input("Jam Belajar Seminggu", min_value=1, max_value=30, value=6)
                attendance_label = st.selectbox(
                    "Kekerapan Kehadiran Sekolah",
                    [
                        "Sangat jarang (1 hari seminggu)",
                        "Jarang (2 hari seminggu)",
                        "Sederhana (3 hari seminggu)",
                        "Kerap (4 hari seminggu)",
                        "Sangat kerap (5 hari seminggu)",
                    ],
                    index=3,
                    help="Dipaparkan sebagai kekerapan supaya lebih mudah difahami pelajar. Sistem akan menukarnya kepada anggaran peratus untuk model."
                )
                attendance_map = {
                    "Sangat jarang (1 hari seminggu)": 20,
                    "Jarang (2 hari seminggu)": 40,
                    "Sederhana (3 hari seminggu)": 60,
                    "Kerap (4 hari seminggu)": 80,
                    "Sangat kerap (5 hari seminggu)": 100,
                }
                Attendance = attendance_map[attendance_label]
                Resources = st.selectbox("Akses Sumber Pembelajaran", [0, 1, 2], index=1, format_func=lambda x: {0:"Rendah", 1:"Sederhana", 2:"Tinggi"}[x])
                Extracurricular = st.selectbox("Aktif Kokurikulum", [0, 1], index=1, format_func=lambda x: "Ya" if x==1 else "Tidak")
                Motivation = st.selectbox("Tahap Motivasi Diri", [0, 1, 2], index=2, format_func=lambda x: {0:"Rendah", 1:"Sederhana", 2:"Tinggi"}[x])
                Internet = st.selectbox("Akses Internet Rumah", [0, 1], index=1, format_func=lambda x: "Ya" if x==1 else "Tidak")
            with cc2:
                LearningStyle = st.selectbox("Gaya Pembelajaran Terpilih", [0, 1, 2, 3], index=0, format_func=lambda x: {0:"Visual", 1:"Auditori", 2:"Kinestetik", 3:"Membaca/Menulis"}[x])
                OnlineCourses = st.number_input("Bilangan Kursus Dalam Talian", min_value=0, max_value=20, value=5)
                Discussions = st.selectbox("Kerap Sertai Perbincangan Kumpulan", [0, 1], index=1, format_func=lambda x: "Ya" if x==1 else "Tidak")
                assignment_label = st.selectbox(
                    "Kekerapan Menyiapkan Tugasan",
                    [
                        "Jarang siap",
                        "Kerap siap",
                        "Sentiasa siap",
                    ],
                    index=1,
                    help="Dipaparkan sebagai kekerapan supaya lebih mudah dijawab oleh pelajar. Sistem akan menukarnya kepada anggaran peratus untuk model."
                )
                assignment_map = {
                    "Jarang siap": 50,
                    "Kerap siap": 75,
                    "Sentiasa siap": 100,
                }
                AssignmentCompletion = assignment_map[assignment_label]
                ExamScore = st.number_input("Markah Peperiksaan Matematik Terakhir", min_value=0, max_value=100, value=65)
                EduTech = st.selectbox("Kerap Guna Aplikasi EduTech", [0, 1], index=1, format_func=lambda x: "Ya" if x==1 else "Tidak")
                StressLevel = st.selectbox("Tahap Stres Pembelajaran", [0, 1, 2], index=1, format_func=lambda x: {0:"Rendah", 1:"Sederhana", 2:"Tinggi"}[x])
            
            if latest_prof:
                st.caption("Profil terdahulu telah disimpan. Ubah mana-mana nilai di atas dan tekan butang di bawah untuk kemas kini ramalan.")
                
            save_prof_btn = st.form_submit_button("Kemaskini Profil & Ramal Aras", type="primary")
            
        if save_prof_btn:
            feature_cols = [
                "StudyHours", "Attendance", "Resources", "Extracurricular", "Motivation",
                "Internet", "Gender", "Age", "LearningStyle", "OnlineCourses",
                "Discussions", "AssignmentCompletion", "ExamScore", "EduTech", "StressLevel"
            ]
            profile = {
                "StudyHours": StudyHours, "Attendance": Attendance, "Resources": Resources,
                "Extracurricular": Extracurricular, "Motivation": Motivation, "Internet": Internet,
                "Gender": Gender, "Age": Age, "LearningStyle": LearningStyle, "OnlineCourses": OnlineCourses,
                "Discussions": Discussions, "AssignmentCompletion": AssignmentCompletion,
                "ExamScore": ExamScore, "EduTech": EduTech, "StressLevel": StressLevel,
            }
            profile = {col: profile[col] for col in feature_cols}
            pred, text, conf, probs = predict_competency(profile)
            
            st.session_state.profile_ready = True
            st.session_state.predicted_level = pred
            st.session_state.predicted_level_text = text
            st.session_state.confidence = conf
            st.session_state.prediction_probs = probs
            
            save_profile(st.session_state.user_id, profile, pred, text, conf)
            reset_quiz()
            # Apabila profil dikemas kini, aras semasa dimulakan semula mengikut ramalan baharu.
            st.session_state.current_level = int(pred)
            st.success(f"Tahap penguasaan: {text} | Keyakinan: {conf:.2%}")
            st.rerun()

        if st.session_state.profile_ready:
            pred = int(st.session_state.predicted_level)
            insight = {
                0: "Mulakan dengan soalan berpandu dan pengukuhan asas sebelum bergerak ke topik yang lebih mencabar.",
                1: "Mulakan dengan soalan sederhana dan naikkan kesukaran apabila jawapan betul serta pantas.",
                2: "Mulakan dengan soalan aras tinggi dan kekalkan cabaran melalui perkembangan adaptif.",
            }.get(pred, "Laluan pembelajaran adaptif akan dilaraskan semasa kuiz.")

            probs = st.session_state.get("prediction_probs", None)
            if probs is None or len(probs) < 3:
                probs = [0.0, 0.0, 0.0]
                probs[pred] = float(st.session_state.confidence)

            prob_html = ""
            for i in [0, 1, 2]:
                percentage = float(probs[i]) * 100
                selected = i == pred
                selected_text = "Dipilih" if selected else ""
                border_style = f"2px solid {LEVEL_COLOR[i]}" if selected else "1px solid #bfdbfe"
                bg_style = "linear-gradient(135deg, #eff6ff, #ecfeff)" if selected else "rgba(255,255,255,0.70)"
                shadow_style = "0 8px 18px rgba(37,99,235,0.14)" if selected else "none"
                prob_html += (
                    f'<div style="padding:12px 14px; border-radius:18px; margin-bottom:10px; border:{border_style}; background:{bg_style}; box-shadow:{shadow_style};">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:7px;">'
                    f'<div style="display:flex; align-items:center; gap:8px;">'
                    f'<b style="color:#0f172a;">{LEVEL_TEXT[i]}</b>'
                    f'<span style="font-size:11px; font-weight:900; color:#1d4ed8;">{selected_text}</span>'
                    f'</div>'
                    f'<span style="font-weight:900; color:{LEVEL_COLOR[i]};">{percentage:.2f}%</span>'
                    f'</div>'
                    f'<div class="v12-confidence-track">'
                    f'<div class="v12-confidence-fill" style="width:{percentage:.1f}%; background:{LEVEL_COLOR[i]};"></div>'
                    f'</div>'
                    f'</div>'
                )
            
            st.markdown(f'''
            <div class="v12-smart-strip">
                <div>
                    <div class="v12-smart-title">Keputusan Klasifikasi Awal</div>
                    <div class="v12-card-sub">{insight}</div>
                </div>
                <div style="min-width:270px;">
                    <div style="margin-bottom:12px;">
                        <span class="v12-level-badge" style="background:{LEVEL_COLOR[pred]};">{LEVEL_TEXT[pred]}</span>
                    </div>
                    <div class="mp-small" style="margin-bottom:10px; color:#1d4ed8; font-weight:900;">Pecahan keyakinan model</div>
                    {prob_html}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("Terus ke Papan Pemuka", type="primary", use_container_width=True):
                go_to("Papan Pemuka")

def page_dashboard():
    app_topbar("Papan Pemuka")
    if not st.session_state.logged_in:
        st.warning("Sila log masuk di halaman Profil dahulu.")
        return
    if not st.session_state.profile_ready:
        st.warning("Sila simpan maklumat profil anda untuk memulakan ramalan pembelajaran mesin.")
        return

    questions_df = load_questions()
    if questions_df.empty:
        st.error("Bank Soalan kosong atau tidak ditemui.")
        return

    topics = sorted(questions_df["Topic"].dropna().astype(str).unique().tolist())
    topic_options = ["Campuran Semua Topik"] + topics
    current_level_num, current_level_label = get_adaptive_start_level()
    current_level_text = LEVEL_TEXT[int(current_level_num)]

    total_bank = len(questions_df)
    total_current_level = len(questions_df[questions_df["Difficulty"].astype(str) == current_level_text])
    metrics = load_metrics()
    metric_text = "Belum tersedia"
    if metrics:
        metric_text = f"{metrics.get('test', {}).get('accuracy', 0):.2%}"

    st.markdown(f"""
    <div class="v12-dashboard-hero" style="margin-bottom:18px;">
        <h2>Selamat Kembali, {st.session_state.username}! 👋</h2>
        <p>{current_level_label} anda ialah <b>{current_level_text}</b>. Jika sebelum ini anda sudah menjawab kuiz, aras ini akan ikut aras akhir/adaptif terkini, bukan semata-mata ramalan awal profil.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="v20-dashboard-wrap">
        <div class="v20-learning-card">
            <div class="v20-card-kicker">🎯 Laluan latihan adaptif</div>
            <h2 class="v20-card-title">Pilih topik, sistem tentukan aras</h2>
            <div class="v20-card-desc">Anda hanya perlu memilih topik dan bilangan soalan. Sistem akan mula daripada aras semasa pelajar, kemudian berubah secara adaptif semasa kuiz dijalankan.</div>
            <div class="v20-flow">
                <div class="v20-flow-step"><b>1. Profil</b><span>Model meramal aras awal berdasarkan data pelajar.</span></div>
                <div class="v20-flow-step"><b>2. Kuiz</b><span>Soalan pertama mengikut aras semasa pelajar.</span></div>
                <div class="v20-flow-step"><b>3. Adaptasi</b><span>Aras berubah ikut jawapan, masa dan bantuan.</span></div>
            </div>
        </div>
        <div class="v20-hero-mini">
            <h3>Pusat Sistem</h3>
            <p>Rekod pembelajaran disimpan untuk menghasilkan laporan prestasi dan sejarah latihan pelajar.</p>
            <div class="v20-side-metric"><small>{current_level_label}</small><b>{current_level_text}</b></div>
            <div class="v20-side-metric"><small>Jumlah bank soalan</small><b>{total_bank} soalan</b></div>
            <div class="v20-side-metric"><small>Ketepatan model</small><b>{metric_text}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="v20-settings-title">Tetapan Sesi Kuiz</div><div class="v20-settings-sub">Pilih jenis latihan yang ingin dijawab. Sistem akan memaparkan jumlah soalan yang tersedia mengikut aras semasa pelajar.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, .7], gap="large")
    with col_a:
        selected_topic = st.selectbox("Pilihan latihan", topic_options, key="training_topic_select")
        if selected_topic == "Campuran Semua Topik":
            topic_qs = questions_df.copy()
            nama_pilihan = "Semua topik"
        else:
            topic_qs = questions_df[questions_df["Topic"].astype(str) == str(selected_topic)].copy()
            nama_pilihan = selected_topic

        level_qs = topic_qs[topic_qs["Difficulty"].astype(str) == current_level_text].copy()
        total_topic = len(topic_qs)
        total_level = len(level_qs)

        st.markdown(f"""
        <div class="v20-summary-grid">
            <div class="v20-summary-card"><small>Latihan dipilih</small><b>{nama_pilihan}</b></div>
            <div class="v20-summary-card"><small>{current_level_label}</small><b>{current_level_text}</b></div>
            <div class="v20-summary-card"><small>Soalan pada aras ini</small><b>{total_level}</b></div>
        </div>
        """, unsafe_allow_html=True)

        if total_level <= 0:
            question_count = 0
            st.error(f"Tiada soalan aras {current_level_text} untuk pilihan latihan ini. Sila pilih topik lain atau gunakan Campuran Semua Topik.")
        elif total_level == 1:
            question_count = 1
            st.info(f"Pilihan ini hanya mempunyai 1 soalan pada aras {current_level_text}. Kuiz akan bermula dengan 1 soalan.")
        else:
            max_slider = min(20, total_level)
            default_n = min(5, max_slider)
            question_count = st.slider("Bilangan soalan", min_value=1, max_value=max_slider, value=default_n, key="question_count_slider")

        st.markdown(f"""
        <div class="v20-adapt-note">
            📌 Jumlah soalan dalam pilihan ini: <b>{total_topic}</b><br>
            🎯 Soalan yang sepadan dengan aras semasa <b>{current_level_text}</b>: <b>{total_level}</b><br>
            🔁 Selepas pelajar menjawab, sistem akan memilih soalan seterusnya berdasarkan aras adaptif terkini.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Mulakan Kuiz Adaptif", type="primary", use_container_width=True, disabled=(question_count == 0)):
            init_level = current_level_num
            base_pool = topic_qs.copy().reset_index(drop=True)
            pool_records = base_pool.to_dict(orient="records")
            first_question = pilih_soalan_adaptif(pool_records, init_level, [])

            if first_question is None:
                st.error("Tiada soalan tersedia untuk pilihan ini.")
                st.stop()

            st.session_state.quiz_questions = [first_question]
            st.session_state.question_pool = pool_records
            st.session_state.question_target_count = int(question_count)
            st.session_state.selected_topic = selected_topic
            st.session_state.quiz_index = 0
            st.session_state.current_level = init_level
            st.session_state.quiz_started = True
            st.session_state.quiz_finished = False
            st.session_state.score = 0.0
            st.session_state.correct_count = 0
            st.session_state.hints_used = 0
            st.session_state.last_feedback = None
            st.session_state.chat_messages = []
            st.session_state.session_id = create_quiz_session(st.session_state.user_id, init_level)
            go_to("Kuiz Adaptif")
            st.rerun()

    with col_b:
        st.markdown(f"""
        <div class="v20-learning-card" style="padding:22px;">
            <div class="v20-card-kicker">🧠 Status pembelajaran</div>
            <h2 class="v20-card-title" style="font-size:1.35rem;">Ringkasan Pelajar</h2>
            <div class="v20-summary-card" style="margin-bottom:10px;"><small>Nama pelajar</small><b>{st.session_state.username}</b></div>
            <div class="v20-summary-card" style="margin-bottom:10px;"><small>Keyakinan model</small><b>{float(st.session_state.confidence):.2%}</b></div>
            <div class="v20-summary-card"><small>Soalan aras semasa dalam bank</small><b>{total_current_level}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="v20-actions">', unsafe_allow_html=True)
        if st.button("📊 Laporan Prestasi", key="dashboard_go_report", use_container_width=True):
            go_to("Laporan Prestasi")
        if st.button("🗄️ Log Pangkalan Data", key="dashboard_go_database", use_container_width=True):
            go_to("Pangkalan Data")
        st.markdown('</div>', unsafe_allow_html=True)

def page_quiz():
    app_topbar("Kuiz Adaptif")

    # Kemasan khusus halaman kuiz v27: tutor dalam satu panel kecil dan tidak melebihi baris butang hantar jawapan.
    st.markdown("""
    <style>
    .v27-quiz-top{
        background:rgba(255,255,255,.92);
        border:1px solid rgba(191,219,254,.80);
        border-radius:24px;
        padding:15px 18px;
        box-shadow:0 16px 36px rgba(37,99,235,.08);
        margin-bottom:14px;
    }
    .v27-quiz-top-row{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;}
    .v27-quiz-title{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:950;color:#082f49;}
    .v27-quiz-sub{font-size:.82rem;color:#64748b;font-weight:750;margin-top:2px;}
    .v27-progress{height:8px;background:#dbeafe;border-radius:999px;overflow:hidden;margin-top:12px;}
    .v27-progress-fill{height:100%;background:linear-gradient(90deg,#2563eb,#14b8a6);border-radius:999px;}
    .v27-question-card{
        background:rgba(255,255,255,.95);
        border:1px solid rgba(191,219,254,.86);
        border-radius:26px;
        padding:22px 24px;
        box-shadow:0 20px 48px rgba(37,99,235,.10);
        margin:12px 0 14px 0;
    }
    .v27-question-title{font-size:1.15rem;font-weight:950;line-height:1.55;color:#0f172a;margin-bottom:16px;}
    .v27-answer-note{border-top:1px solid #dbeafe;padding-top:13px;color:#1d4ed8;font-weight:900;font-size:.85rem;}
    .v27-meta-wrap{margin-bottom:10px;}
    div[data-testid="stRadio"] > div{width:100%;}
    div[data-testid="stRadio"] label{
        width:100% !important;
        min-height:52px !important;
        border-radius:16px !important;
        border:1.4px solid #bfdbfe !important;
        background:#ffffff !important;
        padding:12px 15px !important;
        margin-bottom:8px !important;
        box-shadow:0 8px 18px rgba(37,99,235,.04) !important;
    }
    div[data-testid="stRadio"] label:hover{border-color:#14b8a6 !important;background:#f0fdfa !important;}
    .v27-submit-gap{height:10px;}

    /* Tutor panel: satu kotak yang kemas, tinggi terkawal, dan tidak memanjang ke bawah. */
    .v27-tutor-shell{
        background:rgba(255,255,255,.95);
        border:1px solid rgba(191,219,254,.90);
        border-radius:26px;
        padding:18px 18px 16px 18px;
        box-shadow:0 22px 52px rgba(37,99,235,.11);
        margin-top:0;
    }
    .v27-tutor-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;}
    .v27-tutor-title{font-family:'Space Grotesk',sans-serif;font-size:1.18rem;font-weight:950;color:#082f49;}
    .v27-status{display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;background:#ecfdf5;color:#047857;border:1px solid #99f6e4;font-size:.72rem;font-weight:900;white-space:nowrap;}
    .v27-status.warn{background:#fff7ed;color:#c2410c;border-color:#fed7aa;}
    .v27-tutor-desc{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;border-radius:16px;padding:11px 13px;font-weight:850;line-height:1.45;font-size:.84rem;margin-bottom:10px;}
    .v27-chat-window{
        height:175px;
        max-height:175px;
        overflow-y:auto;
        background:linear-gradient(135deg,#f8fbff,#f0fdfa);
        border:1px solid #dbeafe;
        border-radius:18px;
        padding:11px;
        box-sizing:border-box;
        margin:10px 0 12px 0;
        scrollbar-width:thin;
    }
    .v27-chat-window::-webkit-scrollbar{width:7px;}
    .v27-chat-window::-webkit-scrollbar-thumb{background:#93c5fd;border-radius:999px;}
    .v27-msg{display:flex;gap:8px;margin-bottom:9px;align-items:flex-start;}
    .v27-msg.user{justify-content:flex-end;}
    .v27-avatar{width:28px;height:28px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:14px;flex:0 0 28px;}
    .v27-avatar.bot{background:#f59e0b;color:#fff;}
    .v27-avatar.user{background:#2563eb;color:#fff;order:2;}
    .v27-bubble{max-width:82%;padding:9px 11px;border-radius:15px;font-size:.82rem;line-height:1.5;font-weight:650;}
    .v27-bubble.bot{background:#fff;border:1px solid #bfdbfe;color:#0f172a;border-top-left-radius:5px;}
    .v27-bubble.user{background:linear-gradient(135deg,#2563eb,#0f766e);color:#fff;border-top-right-radius:5px;}
    .v27-form-title{font-size:.80rem;font-weight:900;color:#1e40af;margin:10px 0 7px 0;}
    .v27-tutor-shell textarea{min-height:62px !important;}
    .v27-feedback{border-radius:18px;padding:13px 14px;margin-top:14px;font-weight:760;line-height:1.5;}
    .v27-feedback.ok{background:#ecfdf5;border:1px solid #99f6e4;color:#0f766e;}
    .v27-feedback.warn{background:#fff7ed;border:1px solid #fed7aa;color:#c2410c;}
    @media(max-width:900px){.v27-chat-window{height:160px;max-height:160px;}.v27-question-card{padding:18px;}}
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Sila log masuk dahulu.")
        return
    if not st.session_state.profile_ready:
        st.warning("Sila lengkapkan profil dahulu.")
        return
    if not st.session_state.quiz_started:
        st.info("Kuiz belum dimulakan. Pergi ke Papan Pemuka dan tekan Mula Kuiz.")
        if st.button("Pergi ke Papan Pemuka", type="primary"):
            go_to("Papan Pemuka")
        return
    if st.session_state.quiz_finished:
        show_results()
        return

    q = get_current_question()
    idx = int(st.session_state.quiz_index)
    total = int(st.session_state.question_target_count or len(st.session_state.quiz_questions))
    level_before = int(st.session_state.current_level)
    level_text = LEVEL_TEXT[level_before]

    if st.session_state.question_start_time is None:
        st.session_state.question_start_time = time.time()

    max_time = int(q["MaxTime"])
    progress_pct = ((idx + 1) / max(1, total)) * 100

    # Popup petunjuk: tidak dimasukkan ke dalam chat supaya ruang Tutor kekal kemas.
    if st.session_state.get("show_hint_popup", False):
        hint_text = st.session_state.get("current_hint_text", "Tiada petunjuk bertulis bagi soalan ini.")
        if hasattr(st, "dialog"):
            @st.dialog("💡 Petunjuk Soalan")
            def _hint_dialog():
                st.markdown("**Gunakan petunjuk ini sebagai bimbingan awal.**")
                st.info(hint_text)
                st.caption("Petunjuk ini dikira sebagai bantuan dan boleh memberi kesan kepada skor soalan semasa.")
                if st.button("Tutup", type="primary", use_container_width=True):
                    st.session_state.show_hint_popup = False
                    st.rerun()
            _hint_dialog()
        else:
            st.info("Petunjuk Soalan: " + str(hint_text))
            if st.button("Tutup Petunjuk", use_container_width=True):
                st.session_state.show_hint_popup = False
                st.rerun()

    st.markdown(f"""
    <div class="v27-quiz-top">
        <div class="v27-quiz-top-row">
            <div>
                <div class="v27-quiz-title">Kuiz Adaptif Matematik</div>
                <div class="v27-quiz-sub">Pelajar: <b>{html.escape(str(st.session_state.username))}</b> &nbsp;•&nbsp; Aras semasa: <b>{level_text}</b></div>
            </div>
            <span class="v12-section-pill">Soalan {idx + 1} / {total}</span>
        </div>
        <div class="v27-progress"><div class="v27-progress-fill" style="width:{progress_pct:.1f}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.55, 0.85], gap="large")

    with left:
        st.markdown('<div class="v27-meta-wrap">', unsafe_allow_html=True)
        quiz_meta_timer_component(q["Topic"], q["Difficulty"], st.session_state.question_start_time, max_time, key_suffix=str(idx))
        st.markdown('</div>', unsafe_allow_html=True)

        question_text = html.escape(str(q["QuestionText"]))
        st.markdown(f"""
        <div class="v27-question-card">
            <div class="v27-question-title">{question_text}</div>
            <div class="v27-answer-note">Pilih satu jawapan yang paling tepat.</div>
        </div>
        """, unsafe_allow_html=True)

        opts = ["A", "B", "C", "D"]
        option_texts = {
            "A": str(q.get("OptionA", "")),
            "B": str(q.get("OptionB", "")),
            "C": str(q.get("OptionC", "")),
            "D": str(q.get("OptionD", "")),
        }

        user_choice = st.radio(
            "Pilihan jawapan:",
            opts,
            index=None,
            key=f"q_radio_{idx}",
            format_func=lambda x: f"{x}. {option_texts.get(x, '')}",
            label_visibility="collapsed"
        )

        st.markdown('<div class="v27-submit-gap"></div>', unsafe_allow_html=True)
        submit_btn = st.button("Hantar Jawapan ✔️", type="primary", use_container_width=True, key=f"submit_answer_{idx}")

        if submit_btn:
            if user_choice is None:
                st.warning("Sila pilih satu jawapan dahulu sebelum hantar.")
                return

            time_taken = time.time() - st.session_state.question_start_time
            is_correct = (user_choice == q["CorrectAnswer"])

            fsrs_result = apply_fsrs_and_get_next_level(
                st.session_state.current_level, is_correct, time_taken, max_time,
                st.session_state.used_hint_current, q["Topic"], st.session_state.user_id
            )

            weight = 1.0 if q["Difficulty"] == "Rendah" else (1.5 if q["Difficulty"] == "Sederhana" else 2.0)
            q_score = weight * (1.0 if is_correct else 0.0)
            if st.session_state.used_hint_current:
                q_score *= 0.5

            st.session_state.score += q_score
            if is_correct:
                st.session_state.correct_count += 1

            save_answer(st.session_state.session_id, {
                "question_id": q["QuestionID"],
                "topic": q["Topic"],
                "difficulty": q["Difficulty"],
                "selected_answer": user_choice,
                "correct_answer": q["CorrectAnswer"],
                "is_correct": is_correct,
                "time_taken": time_taken,
                "max_time": max_time,
                "used_hint": st.session_state.used_hint_current,
                "level_before": level_before,
                "level_after": fsrs_result["new_level"],
            })
            save_fsrs_review(
                st.session_state.user_id,
                st.session_state.session_id,
                q["QuestionID"],
                str(q["Topic"]),
                fsrs_result
            )

            st.session_state.last_feedback = {
                "is_correct": is_correct,
                "time_taken": time_taken,
                "reason": fsrs_result["reason"],
                "fsrs": fsrs_result["fsrs"]
            }

            st.session_state.current_level = fsrs_result["new_level"]
            st.session_state.quiz_index += 1
            st.session_state.question_start_time = None
            st.session_state.used_hint_current = False
            st.session_state.hint_count_current = 0
            st.session_state.chat_messages = []
            st.session_state.show_hint_popup = False

            if st.session_state.quiz_index >= total:
                complete_session(
                    st.session_state.session_id, st.session_state.current_level,
                    st.session_state.score, st.session_state.correct_count, len(st.session_state.quiz_questions),
                    st.session_state.hints_used
                )
                st.session_state.quiz_finished = True
            else:
                used_ids = [get_question_id(item) for item in st.session_state.quiz_questions]
                next_question = pilih_soalan_adaptif(
                    st.session_state.question_pool,
                    st.session_state.current_level,
                    used_ids
                )
                if next_question is None:
                    complete_session(
                        st.session_state.session_id, st.session_state.current_level,
                        st.session_state.score, st.session_state.correct_count, len(st.session_state.quiz_questions),
                        st.session_state.hints_used
                    )
                    st.session_state.quiz_finished = True
                else:
                    st.session_state.quiz_questions.append(next_question)
            st.rerun()

        if st.session_state.last_feedback:
            fb = st.session_state.last_feedback
            if fb["is_correct"]:
                st.markdown(f"""
                <div class="v27-feedback ok">
                    <b>Jawapan sebelumnya betul.</b><br>
                    Masa menjawab: {fb['time_taken']:.1f} saat.<br>
                    {fb['reason']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="v27-feedback warn">
                    <b>Jawapan sebelumnya belum tepat.</b><br>
                    Masa menjawab: {fb['time_taken']:.1f} saat.<br>
                    {fb['reason']}
                </div>
                """, unsafe_allow_html=True)

    with right:
        api_status = '<span class="v27-status">API Bersambung</span>' if openai_is_configured() else '<span class="v27-status warn">Mod Sandaran</span>'
        st.markdown(f"""
        <div class="v27-tutor-shell">
            <div class="v27-tutor-head">
                <div class="v27-tutor-title">🤖 Tutor Pintar</div>
                {api_status}
            </div>
            <div class="v27-tutor-desc">
                Saya boleh bantu beri petunjuk dan langkah permulaan tanpa mendedahkan jawapan akhir.
            </div>
        """, unsafe_allow_html=True)

        chat_items = list(st.session_state.chat_messages)
        if not chat_items:
            chat_items = [{"role": "assistant", "content": "Tanya bahagian yang anda kurang faham. Saya akan bantu secara ringkas."}]

        chat_rows = ""
        for msg in chat_items:
            role = msg.get("role", "assistant")
            content = html.escape(str(msg.get("content", ""))).replace("\n", "<br>")
            if role == "user":
                chat_rows += f'<div class="v27-msg user"><div class="v27-bubble user">{content}</div><div class="v27-avatar user">👤</div></div>'
            else:
                chat_rows += f'<div class="v27-msg bot"><div class="v27-avatar bot">🤖</div><div class="v27-bubble bot">{content}</div></div>'

        st.markdown(f'<div class="v27-chat-window">{chat_rows}</div>', unsafe_allow_html=True)

        if st.button("💡 Petunjuk Soalan", use_container_width=True, key=f"hint_btn_{idx}"):
            st.session_state.used_hint_current = True
            st.session_state.hint_count_current += 1
            st.session_state.hints_used += 1
            st.session_state.current_hint_text = str(q.get('Hint', 'Tiada petunjuk bertulis bagi soalan ini.'))
            st.session_state.show_hint_popup = True
            st.rerun()

        st.markdown('<div class="v27-form-title">Tanya Tutor Pintar</div>', unsafe_allow_html=True)
        with st.form(f"borang_tutor_{idx}", clear_on_submit=True):
            student_msg = st.text_area(
                "Tanya Tutor Pintar",
                placeholder="Contoh: Saya tak faham formula yang perlu digunakan.",
                height=62,
                label_visibility="collapsed"
            )
            ask_btn = st.form_submit_button("Hantar kepada Tutor Pintar", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if ask_btn and student_msg.strip():
            st.session_state.chat_messages.append({"role": "user", "content": student_msg.strip()})
            if should_count_ai_help(student_msg):
                st.session_state.used_hint_current = True
                st.session_state.hint_count_current += 1
                st.session_state.hints_used += 1

            reply = get_socratic_reply(q, student_msg.strip(), level_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            save_chat(st.session_state.session_id, q["QuestionID"], student_msg.strip(), reply)
            st.rerun()

def show_results():
    total_q = len(st.session_state.quiz_questions)
    accuracy = (st.session_state.correct_count / max(1, total_q)) * 100
    st.markdown(f"""
    <div class="v24-results-wrap">
        <div style="font-size:3rem;margin-bottom:8px;">🎉</div>
        <h2 style="margin:0;color:#082f49;font-family:Space Grotesk, sans-serif;">Sesi Kuiz Selesai</h2>
        <p class="mp-small" style="margin-top:8px;">Tahniah, anda telah melengkapkan sesi kuiz adaptif.</p>
        <div class="v12-score-panel">
            <div class="score">{float(st.session_state.score):.1f}</div>
            <div style="font-size:0.85rem; letter-spacing:1px; color:#fdf2f8; margin-top:4px; font-weight:700;">JUMLAH SKOR</div>
        </div>
        <div class="v12-result-stats">
            <div class="v12-result-stat"><b>{st.session_state.correct_count} / {total_q}</b><span>Jawapan Betul</span></div>
            <div class="v12-result-stat"><b>{accuracy:.0f}%</b><span>Ketepatan</span></div>
            <div class="v12-result-stat"><b>{LEVEL_TEXT[st.session_state.current_level]}</b><span>Aras Akhir</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Mula Sesi Baharu 🔄", type="primary", use_container_width=True):
        reset_quiz()
        go_to("Papan Pemuka")
        st.rerun()


def page_performance():
    app_topbar("Laporan Prestasi")
    if not st.session_state.logged_in:
        st.warning("Sila log masuk dahulu.")
        return

    sessions = fetch_sessions(st.session_state.user_id)
    if not sessions:
        st.info("Belum ada rekod kuiz selesai. Sila lengkapkan satu kuiz adaptif di Papan Pemuka dahulu.")
        if st.button("Pergi ke Papan Pemuka", type="primary"):
            go_to("Papan Pemuka")
        return

    df = pd.DataFrame(sessions)
    df["completed_at"] = pd.to_datetime(df["completed_at"])
    df = df.sort_values("completed_at")
    df["percent"] = (df["correct_count"] / df["total_questions"].replace(0, np.nan) * 100).fillna(0)
    df["final_level_text"] = df["final_level"].map(LEVEL_TEXT)
    df["initial_level_text"] = df["initial_level"].map(LEVEL_TEXT)

    all_answers = []
    for sid in df["id"].tolist():
        ans = fetch_answers(int(sid))
        for a in ans:
            a["session_id"] = int(sid)
            all_answers.append(a)

    ans_df = pd.DataFrame(all_answers) if all_answers else pd.DataFrame()
    total_questions_answered = len(ans_df)
    avg_percent = float(df["percent"].mean()) if len(df) else 0.0
    latest_percent = float(df.iloc[-1]["percent"]) if len(df) else 0.0
    total_hints = int(df["hints_used"].sum()) if "hints_used" in df else 0
    total_time = float(ans_df["time_taken"].sum()) if not ans_df.empty and "time_taken" in ans_df else 0.0
    avg_time = float(ans_df["time_taken"].mean()) if not ans_df.empty and "time_taken" in ans_df else 0.0

    st.markdown(f"""
    <div class="v24-performance-hero">
        <h2>Laporan Prestasi Pembelajaran</h2>
        <p>Bahagian ini memaparkan ringkasan pencapaian kuiz, penggunaan bantuan Tutor Pintar, masa menjawab dan perkembangan aras adaptif pelajar.</p>
        <div style="margin-top:14px;">
            <span class="v24-soft-pill" style="background:rgba(255,255,255,.14);border-color:rgba(255,255,255,.22);color:#fff;">Sesi selesai: {len(df)}</span>
            <span class="v24-soft-pill" style="background:rgba(255,255,255,.14);border-color:rgba(255,255,255,.22);color:#fff;">Aras terkini: {html.escape(str(df.iloc[-1]['final_level_text']))}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="v24-metric-grid">
        <div class="v24-metric-card"><small>Jumlah Soalan Dijawab</small><strong>{total_questions_answered}</strong><span>Semua sesi kuiz</span></div>
        <div class="v24-metric-card"><small>Purata Ketepatan</small><strong>{avg_percent:.0f}%</strong><span>Berdasarkan jawapan betul</span></div>
        <div class="v24-metric-card"><small>Ketepatan Sesi Terkini</small><strong>{latest_percent:.0f}%</strong><span>Sesi terakhir</span></div>
        <div class="v24-metric-card"><small>Bantuan Tutor Digunakan</small><strong>{total_hints}x</strong><span>Petunjuk dan chat</span></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown("""
        <div class="v24-section-card">
            <h3>Trend Ketepatan Mengikut Sesi</h3>
            <p>Graf ini menunjukkan perubahan prestasi pelajar daripada satu sesi latihan ke sesi seterusnya.</p>
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x_values = range(1, len(df) + 1)
        ax.plot(x_values, df["percent"], marker="o", linewidth=2.8, label="Ketepatan (%)")
        ax.axhline(avg_percent, linestyle="--", alpha=0.65, label="Purata")
        ax.set_xlabel("Sesi Latihan")
        ax.set_ylabel("Ketepatan (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.18)
        ax.legend()
        st.pyplot(fig)

        if not ans_df.empty:
            st.markdown("""
            <div class="v24-section-card">
                <h3>Prestasi Mengikut Topik</h3>
                <p>Ringkasan ini membantu melihat topik yang lebih kuat dan topik yang masih memerlukan latihan.</p>
            </div>
            """, unsafe_allow_html=True)
            topic_summary = ans_df.groupby("topic").agg(
                Bil_Soalan=("question_id", "count"),
                Betul=("is_correct", "sum"),
                Purata_Masa=("time_taken", "mean"),
            ).reset_index()
            topic_summary["Ketepatan (%)"] = (topic_summary["Betul"] / topic_summary["Bil_Soalan"] * 100).round(1)
            topic_summary["Purata_Masa"] = topic_summary["Purata_Masa"].round(1)
            topic_summary = topic_summary.rename(columns={"topic": "Topik", "Purata_Masa": "Purata Masa (s)"})
            st.dataframe(topic_summary, use_container_width=True, hide_index=True)

    with right:
        st.markdown(f"""
        <div class="v24-section-card">
            <h3>Ringkasan Pembelajaran</h3>
            <p>Maklumat ringkas sesi terkini dan corak penggunaan sistem.</p>
            <span class="v24-soft-pill">⏱️ Jumlah masa: {total_time/60:.1f} min</span>
            <span class="v24-soft-pill">⚡ Purata masa: {avg_time:.1f}s</span>
            <span class="v24-soft-pill">🎯 Aras akhir: {html.escape(str(df.iloc[-1]['final_level_text']))}</span>
        </div>
        """, unsafe_allow_html=True)

        display_df = df[["completed_at", "initial_level_text", "final_level_text", "correct_count", "total_questions", "hints_used", "percent"]].copy()
        display_df["completed_at"] = display_df["completed_at"].dt.strftime("%d/%m/%Y %H:%M")
        display_df["percent"] = display_df["percent"].round(1)
        display_df.columns = ["Tarikh", "Aras Awal", "Aras Akhir", "Betul", "Jumlah", "Bantuan", "Ketepatan (%)"]
        st.markdown("### Senarai Sesi")
        st.dataframe(display_df.sort_values("Tarikh", ascending=False), use_container_width=True, hide_index=True)

        fsrs_progress = fetch_fsrs_progress(st.session_state.user_id)
        if fsrs_progress:
            fsrs_df = pd.DataFrame(fsrs_progress)
            if not fsrs_df.empty:
                fsrs_df["current_level"] = fsrs_df["current_level"].map(LEVEL_TEXT)
                fsrs_show = fsrs_df[["topic", "current_level", "review_count", "latest_grade"]].copy()
                fsrs_show.columns = ["Topik", "Aras Semasa", "Bil. Ulang Kaji", "Gred Terkini"]
                st.markdown("### Kemajuan Adaptif Topik")
                st.dataframe(fsrs_show, use_container_width=True, hide_index=True)

def page_database():
    app_topbar("Pangkalan Data")
    if not st.session_state.logged_in:
        st.warning("Sila log masuk dahulu.")
        return
        
    st.markdown("### Semakan Log SQLite Sistem Pembelajaran Adaptif")
    st.markdown('<p class="mp-small">Semua data profil pelajar, transaksi jawapan kuiz, perbualan Tutor Pintar, dan rekod adaptasi topik disimpan secara selamat dalam SQLite.</p>', unsafe_allow_html=True)
    
    def tukar_label_melayu(df_in):
        peta = {
            "id": "ID", "user_id": "ID Pengguna", "username": "Nama Pengguna", "created_at": "Tarikh Cipta",
            "StudyHours": "Jam Belajar", "Attendance": "Kehadiran", "Resources": "Sumber Pembelajaran",
            "Extracurricular": "Kokurikulum", "Motivation": "Motivasi", "Internet": "Internet",
            "Gender": "Jantina", "Age": "Umur", "LearningStyle": "Gaya Pembelajaran",
            "OnlineCourses": "Kursus Dalam Talian", "Discussions": "Perbincangan", "AssignmentCompletion": "Tugasan",
            "ExamScore": "Markah Peperiksaan", "EduTech": "Aplikasi EduTech", "StressLevel": "Tahap Stres",
            "predicted_level": "Aras Ramalan", "predicted_level_text": "Tahap Ramalan", "confidence": "Keyakinan",
            "initial_level": "Aras Awal", "final_level": "Aras Akhir", "score": "Skor",
            "correct_count": "Bilangan Betul", "total_questions": "Jumlah Soalan", "hints_used": "Bilangan Bantuan",
            "started_at": "Masa Mula", "completed_at": "Masa Tamat", "is_completed": "Selesai",
            "session_id": "ID Sesi", "question_id": "ID Soalan", "difficulty": "Kesukaran Soalan",
            "user_answer": "Jawapan Pelajar", "correct_answer": "Jawapan Betul", "is_correct": "Betul",
            "time_taken": "Masa Diambil (s)", "hint_count": "Jumlah Petunjuk", "student_message": "Mesej Pelajar",
            "ai_reply": "Respon AI", "timestamp": "Masa Transaksi", "topic": "Topik",
            "difficulty_fsrs": "Kesukaran Adaptasi", "stability_fsrs": "Kestabilan Adaptasi", "retrievability_fsrs": "Kebolehingatan Adaptasi"
        }
        df_out = df_in.copy()
        df_out.columns = [peta.get(c, c) for c in df_out.columns]
        return df_out

    try:
        conn = sqlite3.connect("adaptive_math.db")
        
        st.markdown("#### 1. Jadual Pengguna (`users`)")
        u = pd.read_sql_query("SELECT * FROM users", conn)
        st.dataframe(tukar_label_melayu(u), use_container_width=True)
        
        st.markdown("#### 2. Jadual Profil Pelajar (`profiles`)")
        sp = pd.read_sql_query("SELECT * FROM profiles", conn)
        st.dataframe(tukar_label_melayu(sp), use_container_width=True)
        
        st.markdown("#### 3. Jadual Sesi Kuiz (`quiz_sessions`)")
        qs = pd.read_sql_query("SELECT * FROM quiz_sessions", conn)
        st.dataframe(tukar_label_melayu(qs), use_container_width=True)
        
        st.markdown("#### 4. Jadual Log Jawapan Pelajar (`quiz_answers`)")
        sa = pd.read_sql_query("SELECT * FROM quiz_answers", conn)
        st.dataframe(tukar_label_melayu(sa), use_container_width=True)
        
        st.markdown("#### 5. Jadual Transaksi Sembang Tutor Pintar (`chat_logs`)")
        cl = pd.read_sql_query("SELECT * FROM chat_logs", conn)
        st.dataframe(tukar_label_melayu(cl), use_container_width=True)
        
        st.markdown("#### 6. Jadual Kemajuan Adaptasi Topik (`fsrs_progress`)")
        fs = pd.read_sql_query("SELECT * FROM fsrs_progress", conn)
        st.dataframe(tukar_label_melayu(fs), use_container_width=True)
        
        conn.close()
    except Exception as e:
        st.error(f"Gagal membaca log pangkalan data SQLite: {e}")

# ------------------------------------------------------------
# Titik Mula Pelaksanaan
# ------------------------------------------------------------
def main():
    init_session_state()
    
    page = st.session_state.nav_page
    if page == "Profil":
        page_login_profile()
    elif page == "Papan Pemuka":
        page_dashboard()
    elif page == "Kuiz Adaptif":
        page_quiz()
    elif page == "Laporan Prestasi":
        page_performance()
    elif page == "Pangkalan Data":
        page_database()

if __name__ == "__main__":
    main()
