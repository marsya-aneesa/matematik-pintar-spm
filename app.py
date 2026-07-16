# ============================================================
# app.py
# Sistem Pembelajaran Adaptif Matematik Tingkatan 4 dan 5
# ============================================================

import os
import time
import math
import random
import sqlite3
import html
import json
import textwrap
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
QUESTION_PATH = DATA_DIR / "Bank Soalan.xlsx"
LEVEL_TEXT = {0: "Rendah", 1: "Sederhana", 2: "Tinggi"}
LEVEL_COLOR = {0: "#7FA8F0", 1: "#1A56DB", 2: "#F2994A"}
LEVEL_LIGHT = {0: "#EAF1FF", 1: "#D6E4FF", 2: "#FFE3C2"}
LEVEL_TEXT_ON_COLOR = {0: "#001B4D", 1: "#FFFFFF", 2: "#7A3B00"}
TUTOR_MARK_SVG = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="12,2.5 21,7.5 21,16.5 12,21.5 3,16.5 3,7.5" stroke="currentColor" stroke-width="1.7"/>'
    '<circle cx="12" cy="12" r="2.3" fill="currentColor"/>'
    '</svg>'
)
LOGO_MARK_SVG = (
    '<svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M4 16 L9 21 L15 6 H27" stroke="#FFFFFF" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

ASSETS_DIR = Path("assets")
_LOGO_CANDIDATES = [ASSETS_DIR / f"logo.{ext}" for ext in ("svg", "png", "jpg", "jpeg", "webp")]
LOGO_PATH = next((p for p in _LOGO_CANDIDATES if p.exists()), None)

st.set_page_config(page_title=APP_TITLE, page_icon=None, layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root{
    --primary:#1A56DB;
    --primary-hover:#1447B8;
    --primary-deep:#0B2E6B;
    --primary-deep-soft:#123B82;
    --on-primary:#FFFFFF;
    --primary-container:#D6E4FF;
    --on-primary-container:#001B4D;

    --accent:#F2994A;
    --accent-hover:#C97A2E;
    --accent-container:#FFE3C2;
    --on-accent:#7A3B00;

    --surface:#E6ECFA;
    --surface-card:#FFFFFF;
    --surface-alt:#E2EAFB;
    --on-surface:#10193A;
    --on-surface-variant:#4A5578;
    --outline:#C6D2EE;
    --outline-strong:#98A9D6;

    --success-bg:#DCF3E0;--on-success:#0C5323;
    --warning-bg:#FBE7D0;--on-warning:#7A4A12;

    --shadow-sm:0 1px 2px rgba(11,17,45,.06),0 1px 1px rgba(11,17,45,.04);
    --shadow-md:0 8px 24px rgba(11,17,45,.10);
    --shadow-lg:0 20px 48px rgba(11,17,45,.16);
    --radius-lg:20px;--radius-md:14px;--radius-sm:10px;
    --sans:'Manrope',system-ui,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,monospace;

    /* internal aliases so the rest of this stylesheet reads consistently */
    --ink:var(--primary-deep);
    --ink-soft:var(--primary-deep-soft);

    --blueprint:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cpath d='M240 0H0V240' fill='none' stroke='%23BFD6FF' stroke-opacity='0.35' stroke-width='1'/%3E%3Ccircle cx='36' cy='196' r='30' fill='none' stroke='%23BFD6FF' stroke-opacity='0.45' stroke-width='1.2'/%3E%3Cpath d='M160 26 A48 48 0 0 1 208 74' fill='none' stroke='%23BFD6FF' stroke-opacity='0.5' stroke-width='1.2'/%3E%3Cline x1='140' y1='140' x2='198' y2='140' stroke='%23BFD6FF' stroke-opacity='0.4' stroke-width='1'/%3E%3Cline x1='169' y1='111' x2='169' y2='169' stroke='%23BFD6FF' stroke-opacity='0.4' stroke-width='1'/%3E%3C/svg%3E");
    --page-pattern:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cpath d='M140 0H0V140' fill='none' stroke='%231A56DB' stroke-opacity='0.08' stroke-width='1'/%3E%3Ccircle cx='20' cy='120' r='16' fill='none' stroke='%231A56DB' stroke-opacity='0.10' stroke-width='1'/%3E%3Cpath d='M92 18 A22 22 0 0 1 114 40' fill='none' stroke='%231A56DB' stroke-opacity='0.10' stroke-width='1'/%3E%3C/svg%3E");
}

/* ------------------------------------------------------------
   Base
------------------------------------------------------------ */
html, body, [class*="css"]{
    font-family:var(--sans) !important;
    color:var(--on-surface) !important;
}
.stApp{
    background-color:var(--surface) !important;
    background-image:var(--page-pattern) !important;
    background-repeat:repeat !important;
    background-attachment:fixed !important;
}
.block-container{
    max-width:1180px !important;
    padding-top:1.1rem !important;
    padding-bottom:4rem !important;
}
#MainMenu, footer, header {visibility:hidden;}
[data-testid="stDecoration"]{display:none !important;}

[data-testid="stMarkdownContainer"]:empty,
.element-container:has([data-testid="stMarkdownContainer"]:empty),
.element-container:has(iframe[height="0"]),
div:has(> div[data-testid="stMarkdownContainer"]:empty){
    display:none !important;
}
.element-container{margin-bottom:.35rem !important;}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5{
    font-family:var(--sans) !important;
    color:var(--on-surface) !important;
    font-weight:800 !important;
    letter-spacing:-.015em !important;
}
.stMarkdown h5{
    font-size:1.02rem !important;
    font-weight:800 !important;
    color:var(--on-surface) !important;
    margin:22px 0 12px 0 !important;
    padding-bottom:9px !important;
    border-bottom:2px solid var(--primary) !important;
    display:inline-block !important;
}

/* ------------------------------------------------------------
   Topbar + navigation (glass)
------------------------------------------------------------ */
.v12-topbar{
    position:sticky;top:.75rem;z-index:30;
    display:flex;align-items:center;justify-content:space-between;gap:1rem;
    padding:13px 20px;border-radius:var(--radius-lg);margin-bottom:18px;
    background:rgba(255,255,255,.62);
    border:1px solid rgba(255,255,255,.80);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.9), var(--shadow-md);
    backdrop-filter:blur(14px) saturate(1.6);
    -webkit-backdrop-filter:blur(14px) saturate(1.6);
}
.v12-brand{display:flex;align-items:center;gap:12px;}
.v12-logo{
    width:46px;height:46px;border-radius:13px;flex:0 0 46px;
    display:flex;align-items:center;justify-content:center;
    background:var(--primary-deep);color:var(--accent);
    border:1px solid var(--accent);overflow:hidden;
}
.v12-logo svg{width:22px;height:22px;}
.v12-logo-img{background:var(--surface-card);}
.v12-logo-img img{width:100%;height:100%;object-fit:contain;padding:4px;box-sizing:border-box;}
.v12-brand-title{
    font-family:var(--sans);font-size:1.14rem;font-weight:800;letter-spacing:-.01em;
    color:var(--on-surface) !important;background:none !important;-webkit-text-fill-color:initial !important;
}
.v12-brand-sub{font-size:.78rem;color:var(--on-surface-variant);font-weight:600;margin-top:1px;}
.v12-section-pill,.v12-user-pill{
    display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:999px;
    font-size:.80rem;font-weight:700;white-space:nowrap;
}
.v12-section-pill{background:var(--primary);color:#fff;box-shadow:0 6px 16px rgba(26,86,219,.28);}
.v12-user-pill{background:rgba(255,255,255,.85);color:var(--on-surface);border:1px solid rgba(255,255,255,.9);}

@media (max-width:900px){
    .v12-topbar{position:relative;top:0;flex-direction:column;align-items:flex-start;}
}

/* ------------------------------------------------------------
   Buttons
------------------------------------------------------------ */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button,
.stDownloadButton button{
    border-radius:999px !important;
    min-height:44px !important;
    padding:0 22px !important;
    font-weight:800 !important;
    letter-spacing:.01em !important;
    transition:background-color .15s ease, border-color .15s ease, color .15s ease, transform .15s ease !important;
}
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
div[data-testid="stFormSubmitButton"] button[kind="primary"]{
    border:1px solid var(--primary) !important;
    background:var(--primary) !important;
    color:var(--on-primary) !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.20), 0 8px 20px rgba(26,86,219,.28) !important;
}
div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover{
    background:var(--primary-hover) !important;
    border-color:var(--primary-hover) !important;
    transform:translateY(-1px);
}
div.stButton > button[kind="secondary"],
div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"],
div[data-testid="stFormSubmitButton"] button[kind="secondary"],
.stDownloadButton button{
    background:var(--surface-card) !important;
    border:1.5px solid var(--outline-strong) !important;
    color:var(--on-surface) !important;
}
div.stButton > button[kind="secondary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"]:hover,
.stDownloadButton button:hover{
    border-color:var(--primary) !important;
    color:var(--primary) !important;
    background:var(--primary-container) !important;
}
div.stButton > button:focus-visible,
div[data-testid="stFormSubmitButton"] button:focus-visible{
    outline:2px solid var(--primary) !important;
    outline-offset:2px !important;
}
div.stButton > button:disabled{opacity:.5 !important;}

/* Petunjuk (hint) button: distinct accent-orange, full-width bar under the composer row */
.st-key-mp_hint_btn_container{margin-top:8px;}
.st-key-mp_hint_btn_container div[data-testid="stFormSubmitButton"] button{
    background:var(--accent) !important;
    border:1px solid var(--accent) !important;
    color:var(--on-accent) !important;
    min-height:44px !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.25), 0 6px 16px rgba(242,153,74,.30) !important;
}
.st-key-mp_hint_btn_container div[data-testid="stFormSubmitButton"] button:hover{
    background:var(--accent-hover) !important;
    border-color:var(--accent-hover) !important;
    color:#fff !important;
}

/* Send button: rounded rectangle (not the global pill) sized to align with the
   textarea beside it — a pill at this width/height ratio would collapse into a circle */
.st-key-mp_send_btn_container div[data-testid="stFormSubmitButton"] button{
    border-radius:var(--radius-md) !important;
    min-height:60px !important;
    padding:0 4px !important;
    font-size:.80rem !important;
    white-space:nowrap !important;
}

/* ------------------------------------------------------------
   Forms and inputs
------------------------------------------------------------ */
[data-testid="stForm"]{
    background:var(--surface-card) !important;
    border:1px solid var(--outline) !important;
    border-radius:var(--radius-lg) !important;
    padding:20px !important;
    box-shadow:var(--shadow-sm) !important;
}
label{font-weight:700 !important;color:var(--on-surface-variant) !important;}

.stTextInput input, .stNumberInput input, textarea{
    border-radius:var(--radius-sm) !important;
    border:1.5px solid var(--outline-strong) !important;
    background:var(--surface-card) !important;
    color:var(--on-surface) !important;
}
.stTextInput input:focus, .stNumberInput input:focus, textarea:focus{
    border-color:var(--primary) !important;
    box-shadow:0 0 0 3px var(--primary-container) !important;
}
[data-baseweb="select"] > div{
    border-radius:var(--radius-sm) !important;
    border:1.5px solid var(--outline-strong) !important;
    background:var(--surface-card) !important;
    box-shadow:none !important;
}
[data-baseweb="select"]:focus-within > div{
    border-color:var(--primary) !important;
    box-shadow:0 0 0 3px var(--primary-container) !important;
}
[data-baseweb="popover"] [role="listbox"]{
    border:1px solid var(--outline) !important;
    border-radius:var(--radius-sm) !important;
    box-shadow:var(--shadow-md) !important;
}
[data-testid="stNumberInput"] button{
    background:var(--primary-deep) !important;
    color:#fff !important;
    border:0 !important;
}
[data-testid="stNumberInput"] button:hover{background:var(--primary) !important;}

[data-testid="stSlider"] [role="slider"]{
    background:var(--primary) !important;
    border:3px solid #ffffff !important;
    box-shadow:0 0 0 1px var(--outline-strong) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div{
    color:var(--primary-deep) !important;
    font-weight:700 !important;
}

.stRadio label,
div[data-testid="stRadio"] label{
    width:100% !important;
    background:var(--surface-card) !important;
    border:1.5px solid var(--outline) !important;
    border-radius:var(--radius-sm) !important;
    padding:13px 16px !important;margin-bottom:8px !important;min-height:52px !important;
    box-shadow:none !important;
    transition:border-color .15s ease, background .15s ease;
}
div[data-testid="stRadio"] > div{gap:8px !important;}
.stRadio label:hover,
div[data-testid="stRadio"] label:hover{
    border-color:var(--primary) !important;
    background:var(--primary-container) !important;
}
div[data-testid="stRadio"] label p{
    font-size:.94rem !important;line-height:1.45 !important;
    color:var(--on-surface) !important;font-weight:550 !important;
}

[data-testid="stTooltipHoverTarget"]{
    display:inline-flex !important;align-items:center !important;justify-content:center !important;
    width:18px !important;height:18px !important;min-width:18px !important;min-height:18px !important;
    max-width:18px !important;max-height:18px !important;margin-left:6px !important;padding:0 !important;
    border-radius:50% !important;background:transparent !important;border:0 !important;box-shadow:none !important;
}
[data-testid="stTooltipHoverTarget"] button,
[data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stTooltipHoverTarget"] button:focus,
[data-testid="stTooltipHoverTarget"] button:active{
    width:18px !important;height:18px !important;min-width:18px !important;min-height:18px !important;
    max-width:18px !important;max-height:18px !important;padding:0 !important;margin:0 !important;
    border-radius:50% !important;background:transparent !important;background-image:none !important;
    border:0 !important;box-shadow:none !important;outline:none !important;
    color:var(--on-surface-variant) !important;transform:none !important;
}
[data-testid="stTooltipHoverTarget"] svg,
[data-testid="stTooltipHoverTarget"] button svg{
    width:17px !important;height:17px !important;
    color:var(--on-surface-variant) !important;stroke:var(--on-surface-variant) !important;fill:none !important;
}

div[data-baseweb="popover"]{min-width:260px !important;width:auto !important;max-width:560px !important;}
div[data-baseweb="popover"] [role="listbox"]{min-width:260px !important;width:auto !important;max-width:560px !important;overflow-x:hidden !important;}
div[data-baseweb="popover"] [role="option"]{
    min-height:38px !important;padding:9px 14px !important;
    white-space:normal !important;overflow:visible !important;text-overflow:clip !important;line-height:1.35 !important;
}
div[data-baseweb="popover"] [role="option"] *{
    white-space:normal !important;overflow:visible !important;text-overflow:clip !important;max-width:none !important;width:auto !important;line-height:1.35 !important;
}
[data-baseweb="select"] div, [data-baseweb="select"] span{white-space:nowrap !important;text-overflow:ellipsis !important;}

/* ------------------------------------------------------------
   Generic cards, kickers, glass pill
------------------------------------------------------------ */
.v12-card-title{
    font-family:var(--sans);font-size:1.30rem;font-weight:800;color:var(--on-surface) !important;
    background:none !important;-webkit-text-fill-color:initial !important;margin-bottom:6px;
}
.v12-card-sub{color:var(--on-surface-variant);font-size:.92rem;line-height:1.6;}

.v40-panel{
    position:relative;overflow:hidden;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -20px bottom -30px;
    color:#fff;border-radius:var(--radius-lg);padding:26px 28px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.10), var(--shadow-lg);
}
.v40-section-kicker{
    display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border-radius:999px;
    background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);color:#fff;
    font-size:.72rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase;
}
.v40-panel-title{
    font-family:var(--sans);font-size:1.45rem;font-weight:800;color:#fff !important;
    background:none !important;-webkit-text-fill-color:#fff !important;
}
.v40-panel-sub{color:rgba(255,255,255,.80);font-weight:550;line-height:1.6;margin-top:4px;max-width:760px;}

.v40-footer{
    margin:34px 0 8px 0;padding-top:16px;border-top:1px solid var(--outline);
    color:var(--on-surface-variant);font-size:.80rem;font-weight:600;text-align:center;
}

.mp-glass-pill{
    display:inline-flex;flex-wrap:wrap;align-items:center;justify-content:center;
    gap:4px 12px;max-width:100%;box-sizing:border-box;
    padding:10px 18px;border-radius:999px;font-size:.82rem;font-weight:700;color:var(--on-surface);
    background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.75);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.8), 0 6px 16px rgba(11,17,45,.08);
    backdrop-filter:blur(10px) saturate(1.6);-webkit-backdrop-filter:blur(10px) saturate(1.6);
}
.mp-glass-pill b{color:var(--primary-deep);}
.mp-glass-pill .mp-glass-item{white-space:nowrap;}

/* ------------------------------------------------------------
   Profile page
------------------------------------------------------------ */
.v12-hero-premium{
    position:relative;overflow:hidden;border-radius:var(--radius-lg);padding:30px;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -20px top -20px;
    color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.10), var(--shadow-lg);
}
.v12-hero-content{position:relative;z-index:1;}
.v12-kicker{
    display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border-radius:999px;
    background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);color:#fff;
    font-weight:800;font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;margin-bottom:18px;
}
.v12-hero-title{
    font-family:var(--sans);font-weight:800;font-size:2.15rem;line-height:1.12;letter-spacing:-.02em;margin:0 0 12px 0;
    color:#fff !important;background:none !important;-webkit-text-fill-color:#fff !important;
}
.v12-hero-desc{max-width:96%;font-size:.98rem;line-height:1.7;color:#D7E3FF;font-weight:500;margin-bottom:18px;}
.v12-feature-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px;}
.v12-feature-card{
    background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.20);
    border-radius:var(--radius-md);padding:14px 15px;min-height:82px;
}
.v12-feature-card b{display:block;font-size:.92rem;color:#fff;font-weight:750;margin-bottom:5px;}
.v12-feature-card span{font-size:.80rem;line-height:1.4;color:#B9CBF2;font-weight:500;}
.v12-feature-chip{
    display:inline-flex;align-items:center;margin:6px 6px 0 0;padding:7px 12px;border-radius:999px;
    background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.20);
    color:#fff;font-size:.78rem;font-weight:650;
}

.v36-mini-result{
    margin-top:14px;background:var(--surface-card);border:1px solid var(--outline);border-radius:var(--radius-md);
    padding:14px 16px;box-shadow:var(--shadow-sm);
    display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
}
.v36-mini-left{display:flex;align-items:center;gap:12px;min-width:0;}
.v36-mini-icon{width:8px;height:8px;border-radius:2px;background:var(--primary);flex:0 0 8px;}
.v36-mini-title{color:var(--on-surface);font-size:1rem;font-weight:750;margin-bottom:2px;}
.v36-mini-sub{color:var(--on-surface-variant);font-size:.82rem;font-weight:550;line-height:1.35;}
.v36-mini-badge{
    display:inline-flex;align-items:center;justify-content:center;
    padding:8px 14px;border-radius:8px;color:#fff;font-size:.82rem;font-weight:800;white-space:nowrap;
    font-family:var(--mono);
}

.v36-modal-card{background:var(--surface-card);border:1px solid var(--outline);border-radius:var(--radius-lg);padding:20px;}
.v36-modal-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px;}
.v36-modal-head{display:flex;align-items:center;gap:11px;min-width:0;}
.v36-modal-icon{width:8px;height:8px;border-radius:2px;background:var(--primary);flex:0 0 8px;}
.v36-modal-title{font-family:var(--sans);font-size:1.12rem;font-weight:800;color:var(--on-surface);letter-spacing:-.01em;line-height:1.2;margin-bottom:2px;}
.v36-modal-desc{color:var(--on-surface-variant);font-size:.80rem;font-weight:550;line-height:1.4;}
.v36-level-pill{
    display:inline-flex;flex-direction:column;align-items:center;justify-content:center;
    min-width:100px;padding:10px 12px;border-radius:var(--radius-sm);color:#fff;font-family:var(--mono);
}
.v36-level-pill small{font-size:.62rem;letter-spacing:.06em;text-transform:uppercase;font-weight:700;opacity:.9;line-height:1;font-family:var(--sans);}
.v36-level-pill b{font-size:1.12rem;font-weight:800;margin:4px 0 2px;line-height:1;}
.v36-quick-note{
    background:var(--primary-container);border:1px solid var(--outline);color:var(--on-primary-container);
    border-radius:var(--radius-sm);padding:11px 13px;font-weight:600;line-height:1.5;font-size:.85rem;margin-bottom:13px;
}
.v36-prob-title{color:var(--on-surface-variant);font-weight:750;font-size:.82rem;margin:2px 0 9px;}
.v36-prob-card{padding:10px 12px;border-radius:var(--radius-sm);margin-bottom:8px;border:1px solid var(--outline);background:var(--surface-card);}
.v36-prob-card.selected{border-width:1.6px;background:var(--primary-container);}
.v36-prob-top{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:7px;}
.v36-prob-label{display:flex;align-items:center;gap:7px;font-weight:750;color:var(--on-surface);font-size:.82rem;}
.v36-selected-chip{font-size:9px;font-weight:800;padding:3px 7px;border-radius:999px;background:var(--primary);color:#fff;}
.v36-prob-track{width:100%;height:8px;border-radius:4px;background:var(--surface-alt);overflow:hidden;}
.v36-prob-fill{height:100%;border-radius:4px;}
.v36-modal-footer-note{margin-top:10px;color:var(--on-surface-variant);font-size:.74rem;font-weight:550;line-height:1.4;text-align:center;}
@media(max-width:700px){.v36-modal-top{align-items:flex-start;}.v36-level-pill{min-width:88px;}}

/* ------------------------------------------------------------
   Dashboard
------------------------------------------------------------ */
.v43-welcome-hero{
    position:relative;overflow:hidden;border-radius:var(--radius-lg);padding:32px 36px;margin:16px 0 22px;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -20px top -20px;
    color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.10), var(--shadow-lg);
}
.v43-hero-inner{position:relative;z-index:1;display:grid;grid-template-columns:1.25fr 1fr;gap:24px;align-items:end;}
.v43-hero-kicker{
    display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.28);color:#fff;border-radius:999px;padding:7px 13px;
    font-size:.74rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase;margin-bottom:14px;
}
.v43-hero-title{
    font-family:var(--sans);font-weight:800;font-size:2.35rem;line-height:1.08;letter-spacing:-.02em;
    color:#fff !important;background:none !important;-webkit-text-fill-color:#fff !important;margin:0 0 12px;
}
.v43-hero-copy{max-width:520px;color:#D7E3FF;font-weight:500;line-height:1.65;margin:0;font-size:1rem;}
.v43-hero-metrics{position:relative;z-index:1;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.v43-hero-metric{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.20);border-radius:var(--radius-md);padding:16px;}
.v43-hero-metric small{display:block;color:#AFC4F5;font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;}
.v43-hero-metric b{display:block;color:#fff;font-size:1.35rem;font-weight:700;font-family:var(--mono);}

.v43-step-grid{position:relative;z-index:1;display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:26px;}
.v43-step{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);border-radius:13px;padding:12px;}
.v43-step-num{
    width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;
    color:var(--on-accent);font-weight:700;background:var(--accent);font-size:.72rem;margin-bottom:8px;font-family:var(--mono);
}
.v43-step b{display:block;color:#fff;font-size:.78rem;margin-bottom:3px;}
.v43-step span{display:block;color:#B9CBF2;font-size:.70rem;font-weight:500;line-height:1.4;}

.v43-dashboard-grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:22px;align-items:start;}
.v43-setup-card{border-radius:var(--radius-lg);background:var(--surface-card);border:1px solid var(--outline);box-shadow:var(--shadow-sm);padding:26px;}
.v43-setup-intro{border-radius:var(--radius-md);background:var(--surface-alt);border:1px solid var(--outline);padding:20px;margin-bottom:20px;}
.v43-setup-kicker{
    display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:999px;
    background:var(--primary-container);color:var(--on-primary-container);
    font-size:.72rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin-bottom:12px;
}
.v43-setup-title{font-family:var(--sans);font-size:1.32rem;font-weight:800;letter-spacing:-.01em;color:var(--on-surface);margin:0 0 7px;}
.v43-setup-sub{color:var(--on-surface-variant);font-weight:550;line-height:1.6;margin:0 0 14px;max-width:780px;}
.v43-control-title{
    display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;
    background:var(--primary-container);color:var(--on-primary-container);
    font-size:.72rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin:10px 0;
}
.v43-stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0;}
.v43-stat-box{background:var(--surface);border:1px solid var(--outline);border-left:3px solid var(--primary);border-radius:var(--radius-md);padding:14px 16px;}
.v43-stat-box small{display:block;color:var(--on-surface-variant);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;margin-bottom:7px;}
.v43-stat-box b{display:block;color:var(--on-surface);font-size:1.05rem;font-weight:700;line-height:1.2;font-family:var(--mono);}
.v43-summary-note{
    margin:18px 0;padding:16px 18px;border-radius:var(--radius-md);
    background:var(--surface-alt);border:1px solid var(--outline);color:var(--on-surface-variant);font-weight:550;line-height:1.7;
}
.v43-summary-note b{color:var(--on-surface);}
.v43-student-card{border-radius:var(--radius-lg);background:var(--surface-card);border:1px solid var(--outline);box-shadow:var(--shadow-sm);overflow:hidden;}
.v43-student-head{
    position:relative;overflow:hidden;padding:22px;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -30px bottom -40px;
}
.v43-student-flex{position:relative;z-index:1;display:flex;gap:14px;align-items:center;}
.v43-avatar{
    width:46px;height:46px;border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;
    background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.26);color:#fff;font-weight:700;font-size:1rem;font-family:var(--mono);
}
.v43-student-flex h3{margin:0 !important;color:#fff !important;background:none !important;-webkit-text-fill-color:#fff !important;font-size:1.10rem !important;}
.v43-student-flex p{margin:3px 0 0;color:#B9CBF2;font-size:.78rem;font-weight:550;}
.v43-current-level{margin:18px;display:flex;justify-content:center;}
.v43-level-icon{display:none;}
.v43-info-list{padding:6px 20px 18px;display:grid;gap:9px;}
.v43-info-item{
    display:flex;justify-content:space-between;align-items:center;gap:12px;
    background:var(--surface);border:1px solid var(--outline);border-radius:var(--radius-sm);padding:12px 14px;
}
.v43-info-item span{color:var(--on-surface-variant);font-size:.74rem;font-weight:700;text-transform:uppercase;letter-spacing:.02em;}
.v43-info-item b{color:var(--on-surface);font-size:.92rem;text-align:right;font-family:var(--mono);}
.v43-side-actions{padding:0 20px 20px;display:grid;gap:10px;}
@media(max-width:940px){
    .v43-hero-inner,.v43-dashboard-grid{grid-template-columns:1fr;}
    .v43-hero-metrics,.v43-step-grid,.v43-stat-row{grid-template-columns:1fr;}
}

/* ------------------------------------------------------------
   Quiz
------------------------------------------------------------ */
.v41-quiz-shell{
    border-radius:var(--radius-lg);padding:20px 22px;background:var(--surface-card);border:1px solid var(--outline);
    box-shadow:var(--shadow-sm);margin:14px 0 18px;
}
.v41-quiz-row{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;}
.v41-quiz-title{font-family:var(--sans);font-size:1.16rem;font-weight:800;color:var(--on-surface);margin:0 0 5px;}
.v41-quiz-sub{
    display:inline-flex;flex-wrap:wrap;align-items:center;gap:4px 12px;max-width:100%;box-sizing:border-box;
    color:var(--on-surface);font-weight:650;font-size:.84rem;
    padding:9px 16px;border-radius:999px;margin-top:4px;
    background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.80);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.8), 0 4px 14px rgba(11,17,45,.08);
    backdrop-filter:blur(10px) saturate(1.6);-webkit-backdrop-filter:blur(10px) saturate(1.6);
}
.v41-quiz-sub b{color:var(--primary-deep);}
.v41-quiz-sub .mp-glass-item{white-space:nowrap;}
.v41-question-pill{
    display:inline-flex;align-items:center;justify-content:center;padding:8px 14px;border-radius:8px;
    background:var(--primary);color:#fff;font-weight:700;font-family:var(--mono);
}
.v41-progress-track{height:6px;border-radius:4px;background:var(--surface-alt);margin-top:16px;overflow:hidden;}
.v41-progress-fill{height:100%;border-radius:4px;background:var(--primary);}

.v27-meta-wrap{margin:0 0 14px 0;}
.v27-question-card{background:var(--surface-card);border:1px solid var(--outline);border-radius:var(--radius-lg);padding:22px 24px;box-shadow:var(--shadow-sm);margin:0 0 14px 0;}
.v27-question-title{font-size:1.12rem;font-weight:700;line-height:1.55;color:var(--on-surface);margin-bottom:16px;}
.v27-answer-note{border-top:1px solid var(--outline);padding-top:13px;color:var(--on-surface-variant);font-weight:650;font-size:.85rem;}
.v27-submit-gap{height:10px;}
.v27-feedback{border-radius:var(--radius-md);padding:13px 15px;margin-top:14px;font-weight:550;line-height:1.55;}
.v27-feedback.ok{background:var(--success-bg);color:var(--on-success);}
.v27-feedback.warn{background:var(--warning-bg);color:var(--on-warning);}

/* Tutor panel: a real Streamlit bordered container (not a raw div-wrap) so it
   genuinely encloses the chat window and composer as one box in the DOM. */
div[data-testid="stVerticalBlockBorderWrapper"].st-key-mp_tutor_panel{
    position:sticky !important;top:96px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"].st-key-mp_tutor_panel > div{
    background:var(--surface-card) !important;border:1px solid var(--outline) !important;
    border-top:3px solid var(--primary) !important;border-radius:var(--radius-lg) !important;
    padding:18px 18px 16px 18px !important;box-shadow:var(--shadow-md) !important;
}
.st-key-mp_tutor_panel [data-testid="stForm"]{
    background:transparent !important;border:0 !important;box-shadow:none !important;
    padding:0 !important;margin-top:10px !important;
}
.st-key-mp_tutor_panel textarea{min-height:62px !important;}
.v27-tutor-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--outline);}
.v27-tutor-title{display:inline-flex;align-items:center;gap:9px;font-family:var(--sans);font-size:1.10rem;font-weight:800;color:var(--on-surface);}
.v27-tutor-title svg{flex:0 0 auto;color:#fff;background:var(--primary);border-radius:7px;padding:4px;box-sizing:content-box;}
.v27-status{display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;background:var(--success-bg);color:var(--on-success);font-size:.72rem;font-weight:750;white-space:nowrap;}
.v27-status:before{content:"";width:6px;height:6px;border-radius:50%;background:var(--on-success);}
.v27-status.warn{background:var(--warning-bg);color:var(--on-warning);}
.v27-status.warn:before{background:var(--on-warning);}
.v27-tutor-desc{
    background:var(--primary-container);border:1px solid var(--outline);color:var(--on-primary-container);
    border-radius:var(--radius-sm);padding:11px 13px;font-weight:550;line-height:1.45;font-size:.84rem;margin-bottom:10px;
}
.v27-chat-window{
    height:175px;max-height:175px;overflow-y:auto;background:var(--surface);border:1px solid var(--outline);
    border-radius:var(--radius-md);padding:12px;box-sizing:border-box;margin:10px 0 12px 0;scrollbar-width:thin;
    box-shadow:inset 0 2px 6px rgba(11,17,45,.06);
}
.v27-chat-window::-webkit-scrollbar{width:7px;}
.v27-chat-window::-webkit-scrollbar-thumb{background:var(--outline-strong);border-radius:999px;}
.v27-msg{display:flex;flex-direction:column;gap:4px;margin-bottom:12px;}
.v27-msg.user{align-items:flex-end;}
.v27-msg.bot{align-items:flex-start;}
.v27-avatar{display:inline-flex;align-items:center;gap:5px;font-size:.64rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;}
.v27-avatar.bot{background:var(--surface-alt);color:var(--on-surface-variant);}
.v27-avatar.user{background:var(--primary-container);color:var(--on-primary-container);}
.v27-avatar svg{color:var(--accent);}
.v27-bubble{max-width:86%;padding:10px 13px;border-radius:var(--radius-sm);font-size:.83rem;line-height:1.5;font-weight:500;box-shadow:var(--shadow-sm);}
.v27-bubble.bot{background:var(--surface-card);border:1px solid var(--outline);color:var(--on-surface);border-top-left-radius:4px;}
.v27-bubble.user{background:var(--primary-deep);color:#fff;border-top-right-radius:4px;}
@media(max-width:900px){
    div[data-testid="stVerticalBlockBorderWrapper"].st-key-mp_tutor_panel{position:relative !important;top:0 !important;}
    .v27-chat-window{height:160px;max-height:160px;}
    .v27-question-card{padding:18px;}
}

/* ------------------------------------------------------------
   Results
------------------------------------------------------------ */
.v41-result-wrap{
    position:relative;overflow:hidden;border-radius:var(--radius-lg);padding:34px;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -20px top -20px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.10), var(--shadow-lg);text-align:center;margin:16px 0 22px;color:#fff;
}
.v41-result-wrap h2{position:relative;z-index:1;color:#fff !important;-webkit-text-fill-color:#fff !important;background:none !important;font-size:1.65rem !important;margin:6px 0 !important;}
.v41-result-wrap p{position:relative;z-index:1;color:#D7E3FF;font-weight:550;margin:0 0 22px;}
.v41-score-ring{
    position:relative;z-index:1;width:auto;min-width:240px;height:auto;border-radius:var(--radius-lg);
    margin:0 auto 24px;background:var(--surface-card);padding:22px 30px;
    display:inline-flex;flex-direction:column;align-items:center;box-shadow:var(--shadow-md);border:1px solid var(--outline);
}
.v41-score-inner{width:auto;height:auto;border-radius:0;background:transparent;border:0;}
.v41-score-value{font-family:var(--mono);font-weight:700;font-size:2.6rem;color:var(--primary-deep);line-height:1;}
.v41-score-label{font-size:.72rem;font-weight:700;color:var(--on-surface-variant);letter-spacing:.05em;text-transform:uppercase;margin-top:8px;}
.v41-result-grid{position:relative;z-index:1;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:900px;margin:0 auto;}
.v41-result-card{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.20);border-radius:var(--radius-md);padding:18px;}
.v41-result-card b{display:block;font-size:1.30rem;color:#fff;font-family:var(--mono);}
.v41-result-card span{display:block;color:#B9CBF2;font-weight:600;font-size:.80rem;margin-top:6px;}
@media(max-width:900px){.v41-result-grid{grid-template-columns:1fr;}}

/* ------------------------------------------------------------
   Performance report
------------------------------------------------------------ */
.v24-performance-hero{
    position:relative;overflow:hidden;
    background:var(--primary-deep);
    background-image:var(--blueprint);background-repeat:no-repeat;background-position:right -20px bottom -30px;
    color:#fff;border-radius:var(--radius-lg);padding:26px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.10), var(--shadow-lg);margin-bottom:18px;
}
.v24-performance-hero h2{position:relative;z-index:1;color:#fff !important;-webkit-text-fill-color:#fff !important;background:none !important;margin:0 0 8px 0 !important;font-size:1.45rem !important;}
.v24-performance-hero p{position:relative;z-index:1;color:#D7E3FF;font-weight:550;line-height:1.6;max-width:760px;margin:0;}
.v24-soft-pill{
    position:relative;z-index:1;display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.24);color:#fff;border-radius:8px;padding:7px 12px;font-weight:700;font-size:.78rem;margin:4px 6px 4px 0;font-family:var(--mono);
}
/* on a light card (not the dark hero) the translucent-white pill is invisible; use the tonal variant instead */
.v24-section-card .v24-soft-pill{
    background:var(--primary-container);border-color:var(--outline);color:var(--on-primary-container);
}
.v24-metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 20px 0;}
.v24-metric-card{background:var(--surface-card);border:1px solid var(--outline);border-left:3px solid var(--primary);border-radius:var(--radius-md);padding:18px;box-shadow:var(--shadow-sm);}
.v24-metric-card small{display:block;color:var(--on-surface-variant);font-weight:700;font-size:.74rem;margin-bottom:6px;}
.v24-metric-card strong{font-family:var(--mono);display:block;color:var(--on-surface);font-size:1.45rem;font-weight:700;line-height:1.15;}
.v24-metric-card span{display:block;color:var(--on-surface-variant);font-size:.76rem;font-weight:600;margin-top:6px;}
.v24-section-card{background:var(--surface-card);border:1px solid var(--outline);border-radius:var(--radius-lg);padding:20px;box-shadow:var(--shadow-sm);margin-bottom:16px;}
.v24-section-card h3{font-family:var(--sans);color:var(--on-surface) !important;-webkit-text-fill-color:initial !important;background:none !important;margin:0 0 6px 0 !important;font-size:1.10rem !important;}
.v24-section-card p{color:var(--on-surface-variant);font-weight:500;line-height:1.55;margin:0 0 12px 0;}
@media(max-width:900px){.v24-metric-grid{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.v24-metric-grid{grid-template-columns:1fr;}}

/* ------------------------------------------------------------
   Tables
------------------------------------------------------------ */
.mp-table-wrap{overflow-x:auto;border:1px solid var(--outline);border-radius:var(--radius-md);box-shadow:var(--shadow-sm);margin-bottom:16px;}
table.mp-table{width:100%;border-collapse:collapse;background:var(--surface-card);font-size:.86rem;}
table.mp-table thead th{
    background:var(--primary-deep);color:#fff;text-align:left;
    font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
    padding:11px 14px;white-space:nowrap;
}
table.mp-table tbody td{padding:10px 14px;color:var(--on-surface);border-top:1px solid var(--outline);font-family:var(--mono);white-space:nowrap;}
table.mp-table tbody tr:nth-child(even){background:var(--surface-alt);}
table.mp-table tbody tr:hover{background:var(--primary-container);}

[data-testid="stDataFrame"]{
    border:1px solid var(--outline) !important;
    border-radius:var(--radius-md) !important;
    box-shadow:var(--shadow-sm) !important;
    overflow:hidden !important;
}
.js-plotly-plot, .stPlotlyChart, [data-testid="stImage"]{
    border-radius:var(--radius-md) !important;
    border:1px solid var(--outline) !important;
}
</style>
""", unsafe_allow_html=True)

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

# ------------------------------------------------------------
# Ramalan Aras Awal (Model Hibrid RF-DNN)
# ------------------------------------------------------------
# Trained by train_model.py. ExamScore is banded into 4 coarse categories rather than used as
# a precise 0-100 value: the raw score has ~0.97 correlation with the label (see train_model.py's
# ExamScore audit), so using it directly would make the model a trivial 100%-accuracy lookup
# with the behavioral features contributing nothing. Banding keeps ExamScore as the dominant,
# honest signal while leaving two of the four bands genuinely ambiguous, so the behavioral
# features have real (if modest) influence -- and heavier MLP regularization plus sigmoid
# calibration keep that influence smooth and the reported confidence realistic.
MODEL_PATH = Path("models") / "hybrid_rf_dnn_bundle.pkl"


@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuatkan model bundle: {e}")
    return None


METRICS_PATH = Path("models") / "metrics.json"


@st.cache_data
def load_model_accuracy():
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if "test" in metrics and "accuracy" in metrics["test"]:
            return f"{float(metrics['test']['accuracy']):.2%}"
        results = metrics.get("results") or metrics.get("summary", {}).get("results", [])
        for row in results:
            if row.get("Dataset") == "Test" and row.get("Model") == "Hybrid RF-DNN":
                return f"{float(row['Accuracy']):.2%}"
    except Exception:
        return None
    return None


def exam_score_to_band(score):
    score = clamp(float(score), 0, 100)
    if score <= 40:
        return 0
    if score <= 60:
        return 1
    if score <= 80:
        return 2
    return 3


def predict_competency(profile_dict):
    bundle = load_model_bundle()

    def sandaran_prediction():
        # Deterministic, smooth fallback used only if the trained bundle can't be loaded.
        score = clamp(float(profile_dict.get("ExamScore", 50)), 0, 100)
        band_centers = {0: 27.5, 1: 70.0, 2: 92.5}
        temperature = 18.0
        raw = {lvl: -abs(score - center) / temperature for lvl, center in band_centers.items()}
        max_raw = max(raw.values())
        exp_scores = {lvl: math.exp(v - max_raw) for lvl, v in raw.items()}
        total = sum(exp_scores.values())
        probs = {lvl: v / total for lvl, v in exp_scores.items()}
        pred_class = max(probs, key=probs.get)
        return pred_class, [probs[0], probs[1], probs[2]]

    if not bundle:
        pred_class, probs = sandaran_prediction()
    else:
        try:
            model_profile = dict(profile_dict)
            try:
                if float(model_profile.get("Age", 18)) < 18:
                    model_profile["Age"] = 18
            except Exception:
                model_profile["Age"] = 18
            model_profile["ExamBand"] = exam_score_to_band(model_profile.get("ExamScore", 50))

            df_in = pd.DataFrame([model_profile])[bundle["feature_cols"]]
            X_proc = bundle["preprocessor"].transform(df_in)
            X_proc = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)
            rf_probs = bundle["rf_model"].predict_proba(X_proc)
            X_hybrid = np.hstack([X_proc, rf_probs])
            pred_prob = bundle["hybrid_model"].predict_proba(X_hybrid)[0]
            pred_class = int(np.argmax(pred_prob))
            probs = list(pred_prob)
        except Exception as e:
            st.warning(f"Ralat ramalan model: {e}. Sistem menggunakan logik sandaran sementara.")
            pred_class, probs = sandaran_prediction()

    conf = float(probs[pred_class])
    return pred_class, LEVEL_TEXT[pred_class], conf, probs

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
Anda ialah Tutor Pintar Sokratik bagi Sistem Pembelajaran Adaptif Matematik untuk Tingkatan 4 dan Tingkatan 5 di Malaysia. 
Tahap penguasaan semasa pelajar ialah {current_level_text}.

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
        return "Kunci API OpenAI tidak dikesan dalam folder sistem ini, jadi Tutor Pintar menggunakan mod sandaran tempatan.\n\n" + sandaran_socratic_reply(question, student_message)

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
        "show_prediction_popup": False,
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
def load_logo_markup():
    """Papar logo tersuai (assets/logo.*) jika wujud; jika tidak, guna tanda grafik lalai."""
    if LOGO_PATH is None:
        return f'<div class="v12-logo">{LOGO_MARK_SVG}</div>'
    import base64
    import mimetypes
    mime = mimetypes.guess_type(str(LOGO_PATH))[0] or "image/png"
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f'<div class="v12-logo v12-logo-img"><img src="data:{mime};base64,{data}" alt="Logo"/></div>'

def app_topbar(section_title):
    """Topbar + aligned navigation.
    Fungsi routing kekal sama; hanya paparan bar atas dan butang navigasi diperhalusi.
    """
    user_badge = f"<span class='v12-user-pill'>{html.escape(str(st.session_state.username))}</span>" if st.session_state.logged_in else ""
    logo_markup = load_logo_markup()
    st.markdown(f"""
    <div class="v12-topbar">
        <div class="v12-brand">
            {logo_markup}
            <div>
                <div class="v12-brand-title">Matematik Pintar SPM</div>
                <div class="v12-brand-sub">Pembelajaran Adaptif Matematik SPM</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:flex-end;">
            <span class="v12-section-pill">{section_title}</span>
            {user_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("Profil", "Profil"),
        ("Papan Pemuka", "Papan Pemuka"),
        ("Kuiz Adaptif", "Kuiz Adaptif"),
        ("Laporan Prestasi", "Laporan Prestasi"),
        ("Pangkalan Data", "Rekod Pembelajaran"),
    ]
    nav_cols = st.columns(5, gap="small")
    current_page = st.session_state.get("nav_page", "Profil")
    safe_section = section_title.replace(" ", "_").replace("&", "dan").lower()
    for col, (page_name, label) in zip(nav_cols, pages):
        is_active = (current_page == page_name)
        with col:
            if st.button(label, key=f"top_nav_{safe_section}_{page_name}", use_container_width=True,
                         type=("primary" if is_active else "secondary")):
                if current_page != page_name:
                    go_to(page_name)
                    st.rerun()


def app_footer():
    st.markdown('<div class="v40-footer">Matematik Pintar SPM · Marsya Aneesa · 2026</div>', unsafe_allow_html=True)

def live_timer_component(start_time, max_time, key_suffix=""):
    start_ms = int(float(start_time) * 1000)
    max_s = int(max_time)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=IBM+Plex+Mono:wght@600;700&display=swap');
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
        * {{ box-sizing:border-box; }}
        .timer {{
            width:100%;
            min-height:96px;
            border-radius:16px;
            padding:16px 18px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:16px;
            background:#0B2E6B;
            color:#FFFFFF;
            border:1px solid rgba(242,153,74,.35);
            font-family:Manrope,Arial,sans-serif;
        }}
        .left {{ display:flex;align-items:center;gap:12px; }}
        .label {{ font-size:12px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#AFC4F5; }}
        .sub {{ font-size:12px;font-weight:700;color:#AFC4F5;margin-top:3px; }}
        .time {{ font-family:'IBM Plex Mono',Manrope,Arial,monospace;font-size:2.1rem;font-weight:700;letter-spacing:-.01em;line-height:1;color:#FFFFFF; }}
        .warn {{ background:#7A2E1D; }}
        .warn .time {{ color:#FFE3C2; }}
        .warn .sub {{ color:#FFE3C2; }}
    </style>
    </head>
    <body>
        <div id="timerBox{key_suffix}" class="timer">
            <div class="left">
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
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=IBM+Plex+Mono:wght@600;700&display=swap');
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; font-family:Manrope,Arial,sans-serif; }}
        * {{ box-sizing:border-box; }}
        .bar {{
            display:flex; align-items:center; gap:10px; flex-wrap:wrap;
            padding:0; width:100%; min-height:42px;
        }}
        .chip {{
            display:inline-flex; align-items:center; gap:7px;
            height:42px; padding:0 15px; border-radius:10px;
            border:1px solid #C6D2EE; background:#FFFFFF;
            color:#10193A; font-size:13px; font-weight:700;
            white-space:nowrap;
        }}
        .level {{ color:#001B4D; border-color:#C6D2EE; background:#D6E4FF; }}
        .timer {{
            display:inline-flex; align-items:center; gap:8px;
            height:42px; padding:0 14px; border-radius:10px;
            background:#0B2E6B;
            color:#FFFFFF; font-weight:700;
            white-space:nowrap;
        }}
        .timer small {{ font-size:11px; opacity:.85; font-weight:700; letter-spacing:.02em; }}
        .timer b {{ font-family:'IBM Plex Mono',Manrope,Arial,monospace; font-size:17px; letter-spacing:-.01em; }}
        .warn {{ background:#7A2E1D; }}
    </style>
    </head>
    <body>
        <div class="bar">
            <div class="chip">Topik: {topic_safe}</div>
            <div class="chip level">Aras Soalan: {difficulty_safe}</div>
            <div id="timerBox{key_suffix}" class="timer"><small>Pemasa</small><b id="timerText{key_suffix}">--:--</b></div>
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
def get_prediction_insight(pred):
    return {
        0: "Sistem mencadangkan pelajar bermula dengan soalan berpandu dan pengukuhan asas sebelum bergerak ke topik yang lebih mencabar.",
        1: "Sistem mencadangkan pelajar bermula dengan soalan sederhana dan menaikkan kesukaran apabila jawapan betul serta pantas.",
        2: "Sistem mencadangkan pelajar bermula dengan soalan aras tinggi dan mengekalkan cabaran melalui perkembangan adaptif.",
    }.get(int(pred), "Laluan pembelajaran adaptif akan dilaraskan semasa kuiz berdasarkan prestasi pelajar.")

def build_prediction_result_html(pred, probs):
    pred = int(pred)
    probs = list(probs) if probs is not None else [0.0, 0.0, 0.0]
    if len(probs) < 3:
        probs = [0.0, 0.0, 0.0]
        probs[pred] = float(st.session_state.get("confidence", 0.0) or 0.0)

    confidence_pct = float(probs[pred]) * 100
    short_note = {
        0: "Mula dengan asas dan bina keyakinan secara berperingkat.",
        1: "Mula dengan soalan sederhana dan naik aras jika jawapan konsisten.",
        2: "Mula dengan soalan mencabar dan kekalkan latihan aras tinggi.",
    }.get(pred, "Sistem akan melaras aras semasa kuiz berdasarkan prestasi pelajar.")

    prob_html_parts = []
    for i in [0, 1, 2]:
        percentage = float(probs[i]) * 100
        selected = i == pred
        selected_class = " selected" if selected else ""
        selected_chip = "<span class='v36-selected-chip'>Dipilih</span>" if selected else ""
        prob_html_parts.append(
            f"<div class='v36-prob-card{selected_class}' style='border-color:{LEVEL_COLOR[i]};'>"
            f"<div class='v36-prob-top'>"
            f"<div class='v36-prob-label'>{LEVEL_TEXT[i]} {selected_chip}</div>"
            f"<div style='font-weight:950;color:var(--on-surface);font-size:.82rem;'>{percentage:.2f}%</div>"
            f"</div>"
            f"<div class='v36-prob-track'>"
            f"<div class='v36-prob-fill' style='width:{percentage:.1f}%;background:{LEVEL_COLOR[i]};'></div>"
            f"</div>"
            f"</div>"
        )
    prob_html = "".join(prob_html_parts)

    return (
        f"<div class='v36-modal-card'>"
        f"<div class='v36-modal-top'>"
        f"<div class='v36-modal-head'>"
        f"<div class='v36-modal-icon'></div>"
        f"<div>"
        f"<div class='v36-modal-title'>Keputusan Klasifikasi Awal</div>"
        f"<div class='v36-modal-desc'>Model Hibrid RF-DNN menggunakan markah peperiksaan dan profil pembelajaran anda untuk menetapkan aras permulaan kuiz adaptif.</div>"
        f"</div>"
        f"</div>"
        f"<div class='v36-level-pill' style='background:{LEVEL_COLOR[pred]};color:{LEVEL_TEXT_ON_COLOR[pred]};'>"
        f"<small>Aras Awal</small>"
        f"<b>{LEVEL_TEXT[pred]}</b>"
        f"<small>{confidence_pct:.2f}%</small>"
        f"</div>"
        f"</div>"
        f"<div class='v36-quick-note'>{html.escape(short_note)}</div>"
        f"<div class='v36-prob-title'>Pecahan keyakinan model</div>"
        f"{prob_html}"
        f"<div class='v36-modal-footer-note'>Aras ini ialah titik permulaan sahaja. Semasa kuiz, sistem akan melaras aras berdasarkan jawapan, masa dan bantuan Tutor Pintar.</div>"
        f"</div>"
    )

def show_prediction_popup_if_needed():
    if not st.session_state.get("profile_ready", False):
        return

    pred = int(st.session_state.predicted_level)
    probs = st.session_state.get("prediction_probs", None)
    if probs is None or len(probs) < 3:
        probs = [0.0, 0.0, 0.0]
        probs[pred] = float(st.session_state.get("confidence", 0.0) or 0.0)

    if st.session_state.get("show_prediction_popup", False) and hasattr(st, "dialog"):
        @st.dialog("Keputusan Klasifikasi Awal")
        def _prediction_dialog():
            st.markdown(build_prediction_result_html(pred, probs), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Tutup", key="prediction_popup_close", use_container_width=True):
                    st.session_state.show_prediction_popup = False
                    st.rerun()
            with c2:
                if st.button("Terus ke Papan Pemuka", key="prediction_popup_dashboard", type="primary", use_container_width=True):
                    st.session_state.show_prediction_popup = False
                    go_to("Papan Pemuka")
                    st.rerun()

        _prediction_dialog()

    elif st.session_state.get("show_prediction_popup", False):
        st.markdown(build_prediction_result_html(pred, probs), unsafe_allow_html=True)
        if st.button("Terus ke Papan Pemuka", key="prediction_inline_dashboard", type="primary", use_container_width=True):
            st.session_state.show_prediction_popup = False
            go_to("Papan Pemuka")
            st.rerun()


def page_login_profile():
    app_topbar("Profil & Kemasukan")
    left, right = st.columns([0.9, 1.1], gap="medium")
    with left:
        st.markdown("""
        <div class="v12-hero-premium">
            <div class="v12-hero-content">
                <h1 class="v12-hero-title">Matematik Pintar SPM</h1>
                <p class="v12-hero-desc">Maklumat profil di sebelah, termasuk markah peperiksaan matematik terakhir, digunakan oleh model Pembelajaran Mesin Hibrid RF-DNN untuk menetapkan tahap penguasaan awal pelajar — Rendah, Sederhana atau Tinggi. Tahap ini menjadi titik permulaan kuiz adaptif, yang seterusnya menyesuaikan aras soalan berdasarkan ketepatan jawapan, masa yang diambil dan penggunaan bantuan Tutor Pintar.</p>
                <div class="v12-feature-grid">
                    <div class="v12-feature-card"><b>Klasifikasi Penguasaan</b><span>Model meramal aras awal pelajar daripada markah peperiksaan dan profil pembelajaran.</span></div>
                    <div class="v12-feature-card"><b>Kuiz Adaptif</b><span>Aras soalan berubah mengikut prestasi semasa kuiz.</span></div>
                    <div class="v12-feature-card"><b>Tutor Pintar</b><span>Bimbingan berpandu tanpa mendedahkan jawapan akhir.</span></div>
                    <div class="v12-feature-card"><b>Rekod Pembelajaran</b><span>Jawapan, masa dan bantuan disimpan untuk laporan prestasi.</span></div>
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
                    help="Dipaparkan sebagai kekerapan supaya lebih mudah difahami pelajar. " \
                    "Sistem akan menukarnya kepada anggaran peratus untuk model."
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
                    help="Dipaparkan sebagai kekerapan supaya lebih mudah dijawab oleh pelajar. " \
                    "Sistem akan menukarnya kepada anggaran peratus untuk model."
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
            st.session_state.show_prediction_popup = True
            
            save_profile(st.session_state.user_id, profile, pred, text, conf)
            reset_quiz()
            # Apabila profil dikemas kini, aras semasa dimulakan semula mengikut ramalan baharu.
            st.session_state.current_level = int(pred)
            st.success(f"Tahap penguasaan: {text} | Keyakinan: {conf:.2%}")
            st.rerun()

        if st.session_state.profile_ready:
            pred = int(st.session_state.predicted_level)
            probs = st.session_state.get("prediction_probs", None)
            if probs is None or len(probs) < 3:
                probs = [0.0, 0.0, 0.0]
                probs[pred] = float(st.session_state.confidence)

            confidence_pct = float(probs[pred]) * 100
            insight = {
                0: "Aras permulaan ditetapkan kepada Rendah.",
                1: "Aras permulaan ditetapkan kepada Sederhana.",
                2: "Aras permulaan ditetapkan kepada Tinggi.",
            }.get(pred, "Aras permulaan telah ditetapkan.")

            # Paparan kecil sahaja di halaman profil supaya pengguna tidak perlu scroll panjang.
            # Keputusan penuh dipaparkan dalam popup/modal yang lebih kemas.
            st.markdown(f"""
            <div class="v36-mini-result">
                <div class="v36-mini-left">
                    <div class="v36-mini-icon"></div>
                    <div>
                        <div class="v36-mini-title">Klasifikasi awal berjaya dijana</div>
                        <div class="v36-mini-sub">{insight}</div>
                    </div>
                </div>
                <span class="v36-mini-badge" style="background:{LEVEL_COLOR[pred]};color:{LEVEL_TEXT_ON_COLOR[pred]};">
                    {LEVEL_TEXT[pred]} · {confidence_pct:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            show_prediction_popup_if_needed()

            col_result_1, col_result_2 = st.columns(2)
            with col_result_1:
                if st.button("Lihat Semula Keputusan", key="lihat_prediction_popup", use_container_width=True):
                    st.session_state.show_prediction_popup = True
                    st.rerun()
            with col_result_2:
                if st.button("Terus ke Papan Pemuka", key="profile_go_dashboard_after_prediction", type="primary", use_container_width=True):
                    st.session_state.show_prediction_popup = False
                    go_to("Papan Pemuka")
                    st.rerun()

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

    try:
        confidence_text = f"{float(st.session_state.confidence):.2%}"
    except Exception:
        confidence_text = "-"

    model_accuracy_text = load_model_accuracy() or "-"

    safe_name = html.escape(str(st.session_state.username).title())
    safe_level = html.escape(str(current_level_text))
    safe_level_label = html.escape(str(current_level_label))

    st.markdown(f"""
    <div class="v43-welcome-hero">
        <div class="v43-hero-inner">
            <div>
                <span class="v43-hero-kicker">Papan Pemuka Pembelajaran</span>
                <h1 class="v43-hero-title">Selamat kembali, {safe_name}</h1>
                <p class="v43-hero-copy">Pilih topik dan bilangan soalan. Sistem menggunakan aras semasa pelajar sebagai titik mula, kemudian menyesuaikan soalan seterusnya berdasarkan ketepatan jawapan, masa menjawab dan penggunaan Tutor Pintar.</p>
            </div>
            <div class="v43-hero-metrics">
                <div class="v43-hero-metric"><small>Bank Soalan</small><b>{total_bank}</b></div>
                <div class="v43-hero-metric"><small>Keyakinan Profil</small><b>{confidence_text}</b></div>
                <div class="v43-hero-metric"><small>Ketepatan Model</small><b>{model_accuracy_text}</b></div>
            </div>
        </div>
        <div class="v43-step-grid" style="margin-top:20px;">
            <div class="v43-step"><div class="v43-step-num">1</div><b>Profil Dianalisis</b><span>Data profil pelajar disediakan untuk model Pembelajaran Mesin.</span></div>
            <div class="v43-step"><div class="v43-step-num">2</div><b>Aras Awal Dikelaskan</b><span>Model menetapkan tahap penguasaan permulaan pelajar.</span></div>
            <div class="v43-step"><div class="v43-step-num">3</div><b>Kuiz Bermula</b><span>Soalan pertama dipilih mengikut aras semasa pelajar.</span></div>
            <div class="v43-step"><div class="v43-step-num">4</div><b>Sistem Menyesuaikan Aras</b><span>Aras berubah mengikut ketepatan, masa dan bantuan digunakan.</span></div>
            <div class="v43-step"><div class="v43-step-num">5</div><b>Prestasi Direkod</b><span>Hasil kuiz disimpan untuk laporan dan rekod pembelajaran.</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="v43-dashboard-grid">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.55, 0.75], gap="large")

    with left_col:
        st.markdown("""
        <div class="v43-setup-card">
            <div class="v43-setup-intro">
                <span class="v43-setup-kicker">Latihan Adaptif</span>
                <h2 class="v43-setup-title">Pilih topik, sistem tentukan aras</h2>
                <p class="v43-setup-sub">Anda hanya perlu memilih topik dan bilangan soalan. Sistem akan bermula daripada aras semasa pelajar, kemudian berubah secara adaptif semasa kuiz dijalankan.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="v43-control-title">Pilihan Latihan</div>', unsafe_allow_html=True)
        selected_topic = st.selectbox("Pilihan latihan", topic_options, key="training_topic_select", label_visibility="collapsed")

        if selected_topic == "Campuran Semua Topik":
            topic_qs = questions_df.copy()
            nama_pilihan = "Semua topik"
        else:
            topic_qs = questions_df[questions_df["Topic"].astype(str) == str(selected_topic)].copy()
            nama_pilihan = selected_topic

        level_qs = topic_qs[topic_qs["Difficulty"].astype(str) == current_level_text].copy()
        total_topic = len(topic_qs)
        total_level = len(level_qs)
        safe_topic_name = html.escape(str(nama_pilihan))

        st.markdown(f"""
            <div class="v43-stat-row">
                <div class="v43-stat-box"><small>Latihan dipilih</small><b>{safe_topic_name}</b></div>
                <div class="v43-stat-box"><small>{safe_level_label}</small><b>{safe_level}</b></div>
                <div class="v43-stat-box"><small>Soalan aras ini</small><b>{total_level}</b></div>
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
            st.markdown('<div class="v43-control-title">Bilangan Soalan</div>', unsafe_allow_html=True)
            question_count = st.slider("Bilangan soalan", min_value=1, max_value=max_slider, value=default_n, key="question_count_slider", label_visibility="collapsed")

        st.markdown(f"""
            <div class="v43-summary-note">
                <b>Ringkasan sesi</b><br>
                Jumlah soalan dalam pilihan ini: <b>{total_topic}</b><br>
                Soalan tersedia untuk aras <b>{safe_level}</b>: <b>{total_level}</b><br>
                Selepas setiap jawapan, sistem memilih soalan seterusnya berdasarkan aras adaptif terkini.
            </div>
        """, unsafe_allow_html=True)

        if st.button("Mulakan Kuiz Adaptif", type="primary", use_container_width=True, disabled=(question_count == 0)):
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
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        avatar_initial = html.escape((str(st.session_state.username).strip()[:1] or "P").upper())
        st.markdown(f"""
        <div class="v43-student-card">
            <div class="v43-student-head">
                <div class="v43-student-flex">
                    <div class="v43-avatar">{avatar_initial}</div>
                    <div><h3>Ringkasan Pelajar</h3><p>Status model dan adaptasi</p></div>
                </div>
            </div>
            <div class="v43-current-level">
                <span class="mp-glass-pill"><span class="mp-glass-item">Pelajar: <b>{safe_name}</b></span><span class="mp-glass-item">Aras: <b>{safe_level}</b></span></span>
            </div>
            <div class="v43-info-list">
                <div class="v43-info-item"><span>Nama Pelajar</span><b>{safe_name}</b></div>
                <div class="v43-info-item"><span>Keyakinan Profil</span><b>{confidence_text}</b></div>
                <div class="v43-info-item"><span>Soalan Aras Ini</span><b>{total_current_level}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="v43-side-actions">', unsafe_allow_html=True)
        if st.button("Laporan Prestasi", key="dashboard_go_report", use_container_width=True):
            go_to("Laporan Prestasi")
            st.rerun()
        if st.button("Rekod Pembelajaran", key="dashboard_go_database", use_container_width=True):
            go_to("Pangkalan Data")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def page_quiz():
    app_topbar("Kuiz Adaptif")

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
            @st.dialog("Petunjuk Soalan")
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
    <div class="v41-quiz-shell">
        <div class="v41-quiz-row">
            <div>
                <div class="v41-quiz-title">Kuiz Adaptif Matematik</div>
                <div class="v41-quiz-sub"><span class="mp-glass-item">Pelajar: <b>{html.escape(str(st.session_state.username))}</b></span><span class="mp-glass-item">Aras semasa: <b>{level_text}</b></span></div>
            </div>
            <span class="v41-question-pill">Soalan {idx + 1} / {total}</span>
        </div>
        <div class="v41-progress-track"><div class="v41-progress-fill" style="width:{progress_pct:.1f}%;"></div></div>
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
        submit_btn = st.button("Hantar Jawapan", type="primary", use_container_width=True, key=f"submit_answer_{idx}")

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
        with st.container(key="mp_tutor_panel", border=True):
            st.markdown(f"""
            <div class="v27-tutor-head">
                <div class="v27-tutor-title">{TUTOR_MARK_SVG} Tutor Pintar</div>
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
                    chat_rows += f'<div class="v27-msg user"><span class="v27-avatar user">Anda</span><div class="v27-bubble user">{content}</div></div>'
                else:
                    chat_rows += f'<div class="v27-msg bot"><span class="v27-avatar bot">{TUTOR_MARK_SVG} Tutor</span><div class="v27-bubble bot">{content}</div></div>'

            st.markdown(f'<div class="v27-chat-window">{chat_rows}</div>', unsafe_allow_html=True)

            with st.form(f"borang_tutor_{idx}", clear_on_submit=True, border=False):
                row_c1, row_c2 = st.columns([0.78, 0.22], gap="small")
                with row_c1:
                    student_msg = st.text_area(
                        "Tanya Tutor Pintar",
                        placeholder="Contoh: Saya tak faham formula yang perlu digunakan.",
                        height=62,
                        label_visibility="collapsed"
                    )
                with row_c2:
                    with st.container(key="mp_send_btn_container"):
                        ask_btn = st.form_submit_button("Hantar", type="primary", use_container_width=True)
                with st.container(key="mp_hint_btn_container"):
                    hint_btn = st.form_submit_button("Petunjuk", use_container_width=True)

        if hint_btn:
            st.session_state.used_hint_current = True
            st.session_state.hint_count_current += 1
            st.session_state.hints_used += 1
            st.session_state.current_hint_text = str(q.get('Hint', 'Tiada petunjuk bertulis bagi soalan ini.'))
            st.session_state.show_hint_popup = True
            st.rerun()

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
    final_level_text = LEVEL_TEXT[st.session_state.current_level]
    st.markdown(f"""
    <div class="v41-result-wrap">
        <h2>Sesi Kuiz Selesai</h2>
        <p>Prestasi sesi ini telah disimpan dan aras adaptif pelajar telah dikemas kini.</p>
        <div class="v41-score-ring">
            <div class="v41-score-inner">
                <div class="v41-score-value">{float(st.session_state.score):.1f}</div>
                <div class="v41-score-label">Jumlah Skor</div>
            </div>
        </div>
        <div class="v41-result-grid">
            <div class="v41-result-card"><b>{st.session_state.correct_count} / {total_q}</b><span>Jawapan Betul</span></div>
            <div class="v41-result-card"><b>{accuracy:.0f}%</b><span>Ketepatan</span></div>
            <div class="v41-result-card"><b>{final_level_text}</b><span>Aras Akhir</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mula Sesi Baharu", type="primary", use_container_width=True):
        reset_quiz()
        go_to("Papan Pemuka")
        st.rerun()


def render_table(df):
    """Papar DataFrame sebagai jadual HTML bergaya tema sistem (label dan data kekal sama)."""
    st.markdown(
        f'<div class="mp-table-wrap">{df.to_html(classes="mp-table", index=False, escape=True, border=0)}</div>',
        unsafe_allow_html=True
    )

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
            <span class="v24-soft-pill">Sesi Selesai: {len(df)}</span>
            <span class="v24-soft-pill">Aras Terkini: {html.escape(str(df.iloc[-1]['final_level_text']))}</span>
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
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        x_values = list(range(1, len(df) + 1))
        y_values = df["percent"].tolist()
        ax.fill_between(x_values, y_values, 0, color="#1A56DB", alpha=0.08, zorder=1)
        ax.plot(x_values, y_values, marker="o", linewidth=2.4, color="#1A56DB",
                markersize=6, markerfacecolor="#F2994A", markeredgecolor="#FFFFFF",
                markeredgewidth=1.4, label="Ketepatan (%)", zorder=3)
        ax.axhline(avg_percent, linestyle="--", linewidth=1.3, color="#F2994A", alpha=0.85,
                   label=f"Purata ({avg_percent:.0f}%)", zorder=2)
        ax.set_xlabel("Sesi Latihan", fontsize=10, color="#4A5578")
        ax.set_ylabel("Ketepatan (%)", fontsize=10, color="#4A5578")
        ax.set_ylim(0, 105)
        ax.set_xticks(x_values)
        ax.tick_params(colors="#4A5578", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C6D2EE")
        ax.spines["bottom"].set_color("#C6D2EE")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#C6D2EE", linewidth=0.8, alpha=0.8)
        ax.legend(frameon=False, fontsize=9, labelcolor="#4A5578")
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
            render_table(topic_summary)

    with right:
        st.markdown(f"""
        <div class="v24-section-card">
            <h3>Ringkasan Pembelajaran</h3>
            <p>Maklumat ringkas sesi terkini dan corak penggunaan sistem.</p>
            <span class="v24-soft-pill">Jumlah Masa: {total_time/60:.1f} min</span>
            <span class="v24-soft-pill">Purata Masa: {avg_time:.1f}s</span>
            <span class="v24-soft-pill">Aras Akhir: {html.escape(str(df.iloc[-1]['final_level_text']))}</span>
        </div>
        """, unsafe_allow_html=True)

        display_df = df[["completed_at", "initial_level_text", "final_level_text", "correct_count", "total_questions", "hints_used", "percent"]].copy()
        display_df["completed_at"] = display_df["completed_at"].dt.strftime("%d/%m/%Y %H:%M")
        display_df["percent"] = display_df["percent"].round(1)
        display_df.columns = ["Tarikh", "Aras Awal", "Aras Akhir", "Betul", "Jumlah", "Bantuan", "Ketepatan (%)"]
        st.markdown("##### Senarai Sesi")
        render_table(display_df.sort_values("Tarikh", ascending=False))

    fsrs_progress = fetch_fsrs_progress(st.session_state.user_id)
    if fsrs_progress:
        fsrs_df = pd.DataFrame(fsrs_progress)
        if not fsrs_df.empty:
            fsrs_df["current_level"] = fsrs_df["current_level"].map(LEVEL_TEXT)
            fsrs_show = fsrs_df[["topic", "current_level", "review_count", "latest_grade"]].copy()
            fsrs_show.columns = ["Topik", "Aras Semasa", "Bil. Ulang Kaji", "Gred Terkini"]
            st.markdown("""
            <div class="v24-section-card">
                <h3>Kemajuan Adaptif Topik</h3>
                <p>Aras semasa dan bilangan ulang kaji bagi setiap topik berdasarkan sistem pengulangan adaptif.</p>
            </div>
            """, unsafe_allow_html=True)
            render_table(fsrs_show)

def page_database():
    app_topbar("Rekod Pembelajaran")
    if not st.session_state.logged_in:
        st.warning("Sila log masuk dahulu.")
        return

    st.markdown("""
    <div class="v40-panel" style="margin:16px 0 18px 0;">
        <span class="v40-section-kicker">Rekod Pembelajaran</span>
        <h2 class="v40-panel-title" style="margin-top:14px;">Rekod Pembelajaran Pelajar</h2>
        <p class="v40-panel-sub">Bahagian ini memaparkan profil pelajar, sesi kuiz, jawapan, bimbingan Tutor Pintar dan kemajuan adaptif yang disimpan oleh sistem sebagai rekod pembelajaran.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        uid = st.session_state.user_id

        st.markdown("##### Profil Pengguna")
        u = pd.read_sql_query("SELECT * FROM users WHERE id = ?", conn, params=(uid,))
        st.dataframe(tukar_label_melayu(u), use_container_width=True)

        st.markdown("##### Profil Pembelajaran")
        sp = pd.read_sql_query("SELECT * FROM profiles WHERE user_id = ?", conn, params=(uid,))
        st.dataframe(tukar_label_melayu(sp), use_container_width=True)

        st.markdown("##### Sesi Kuiz")
        qs = pd.read_sql_query("SELECT * FROM quiz_sessions WHERE user_id = ?", conn, params=(uid,))
        st.dataframe(tukar_label_melayu(qs), use_container_width=True)

        st.markdown("##### Rekod Jawapan")
        sa = pd.read_sql_query(
            "SELECT qa.* FROM quiz_answers qa JOIN quiz_sessions qs ON qa.session_id = qs.id WHERE qs.user_id = ?",
            conn, params=(uid,)
        )
        st.dataframe(tukar_label_melayu(sa), use_container_width=True)

        st.markdown("##### Log Bimbingan Tutor Pintar")
        cl = pd.read_sql_query(
            "SELECT cl.* FROM chat_logs cl JOIN quiz_sessions qs ON cl.session_id = qs.id WHERE qs.user_id = ?",
            conn, params=(uid,)
        )
        st.dataframe(tukar_label_melayu(cl), use_container_width=True)

        st.markdown("##### Kemajuan Adaptif Topik")
        fs = pd.read_sql_query("SELECT * FROM fsrs_progress WHERE user_id = ?", conn, params=(uid,))
        st.dataframe(tukar_label_melayu(fs), use_container_width=True)

        conn.close()
    except Exception as e:
        st.error(f"Gagal membaca rekod pembelajaran: {e}")

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
    app_footer()

if __name__ == "__main__":
    main()
