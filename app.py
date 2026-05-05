"""
AI Shorts Generator - Redesigned UI/UX (v2)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import os, json, time, threading, queue, shutil
from pathlib import Path

from video_downloader           import download_youtube_video, validate_youtube_url
from audio_extractor            import extract_audio
from transcription              import transcribe_audio
from viral_detector             import detect_viral_segments
from clip_generator             import generate_clips
from face_tracker               import apply_face_tracking
from caption_generator          import generate_captions
from whisperx_caption_generator import generate_whisperx_captions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Shorts Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MASTER CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: #080812;
  color: #e2e8f0;
}
.stApp { background: #080812; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #3b3b6b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #667eea; }

/* ── Animated Background Orbs ── */
.bg-orbs {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none; z-index: 0; overflow: hidden;
}
.orb {
  position: absolute; border-radius: 50%;
  filter: blur(80px); opacity: 0.06; animation: orbFloat 20s infinite ease-in-out;
}
.orb-1 { width: 600px; height: 600px; background: #667eea; top: -200px; left: -200px; animation-delay: 0s; }
.orb-2 { width: 400px; height: 400px; background: #f093fb; top: 30%; right: -150px; animation-delay: -7s; }
.orb-3 { width: 500px; height: 500px; background: #4fd1c5; bottom: -150px; left: 30%; animation-delay: -14s; }
@keyframes orbFloat {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33%       { transform: translate(30px, -50px) scale(1.05); }
  66%       { transform: translate(-20px, 30px) scale(0.95); }
}

/* ── Hero Header ── */
.hero {
  position: relative; z-index: 1;
  background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.1) 50%, rgba(240,147,251,0.08) 100%);
  border: 1px solid rgba(102,126,234,0.25);
  border-radius: 28px; padding: 3rem 2.5rem;
  text-align: center; margin-bottom: 2rem;
  backdrop-filter: blur(20px);
  box-shadow: 0 0 80px rgba(102,126,234,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
  overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; top: 0; left: 50%;
  transform: translateX(-50%);
  width: 60%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(102,126,234,0.8), transparent);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(102,126,234,0.15); border: 1px solid rgba(102,126,234,0.3);
  border-radius: 50px; padding: 6px 16px; font-size: 0.75rem;
  color: #a78bfa; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; margin-bottom: 1.2rem;
  animation: badgePulse 3s ease-in-out infinite;
}
@keyframes badgePulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(167,139,250,0.4); }
  50%       { box-shadow: 0 0 0 8px rgba(167,139,250,0); }
}
.hero h1 {
  font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 900;
  margin: 0 0 0.8rem 0; line-height: 1.1;
  background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #f093fb 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 1rem; color: #94a3b8; margin: 0;
  display: flex; justify-content: center; flex-wrap: wrap; gap: 1.5rem;
}
.hero-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 50px; padding: 4px 14px; font-size: 0.82rem; color: #cbd5e1;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0d0d1a !important;
  border-right: 1px solid rgba(102,126,234,0.12) !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
  color: #a78bfa; font-size: 0.7rem; letter-spacing: 0.1em;
  text-transform: uppercase; font-weight: 700;
  border-bottom: 1px solid rgba(102,126,234,0.2);
  padding-bottom: 0.4rem; margin-bottom: 0.6rem;
}

/* ── Glass Cards ── */
.glass-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px; padding: 1.5rem;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  position: relative; overflow: hidden;
}
.glass-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
  opacity: 0; transition: opacity 0.3s ease;
}
.glass-card:hover::before { opacity: 1; }
.glass-card:hover {
  border-color: rgba(102,126,234,0.2);
  box-shadow: 0 8px 32px rgba(102,126,234,0.08);
  transform: translateY(-1px);
}

/* ── Section Headers ── */
.section-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 1.2rem;
}
.section-icon {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
  border: 1px solid rgba(102,126,234,0.25);
  flex-shrink: 0;
}
.section-title {
  font-size: 1.05rem; font-weight: 700; color: #f1f5f9;
  margin: 0;
}
.section-subtitle { font-size: 0.78rem; color: #64748b; margin: 0; }

/* ── Input Fields ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(102,126,234,0.2) !important;
  border-radius: 12px !important; color: #e2e8f0 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: rgba(102,126,234,0.6) !important;
  box-shadow: 0 0 0 3px rgba(102,126,234,0.1) !important;
  background: rgba(102,126,234,0.05) !important;
}

/* ── Radio & Checkbox ── */
.stRadio > div { gap: 0.4rem !important; }
.stRadio > div > label {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px; padding: 0.6rem 1rem;
  cursor: pointer; transition: all 0.2s ease;
}
.stRadio > div > label:hover {
  border-color: rgba(102,126,234,0.3);
  background: rgba(102,126,234,0.05);
}

/* ── Selectbox ── */
.stSelectbox > div > div {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(102,126,234,0.2) !important;
  border-radius: 12px !important; color: #e2e8f0 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
  background: linear-gradient(90deg, #667eea, #764ba2) !important;
}
.stSlider > div > div > div > div > div {
  background: #667eea !important;
  box-shadow: 0 0 8px rgba(102,126,234,0.6) !important;
}

/* ── Toggle ── */
.stCheckbox > label > div[data-testid="stCheckboxLabel"] { color: #cbd5e1 !important; }

/* ── Generate Button ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
  color: white !important; border: none !important;
  border-radius: 16px !important; font-weight: 700 !important;
  font-size: 1.1rem !important; padding: 1rem 2rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
  position: relative; overflow: hidden;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(102,126,234,0.25) !important;
  color: #a78bfa !important; border-radius: 12px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(102,126,234,0.08) !important;
  border-color: rgba(102,126,234,0.5) !important;
}

/* ── Progress ── */
div[data-testid="stProgress"] { border-radius: 50px; overflow: hidden; }
div[data-testid="stProgress"] > div {
  background: rgba(255,255,255,0.05) !important;
  border-radius: 50px;
}
div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
  border-radius: 50px;
  box-shadow: 0 0 12px rgba(102,126,234,0.5);
  transition: width 0.5s ease !important;
}

/* ── Pipeline Steps ── */
.pipeline-wrap { display: flex; flex-direction: column; gap: 6px; }

.pipeline-step {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; border-radius: 12px;
  border: 1px solid transparent;
  transition: all 0.4s ease; position: relative; overflow: hidden;
}
.pipeline-step::after {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  border-radius: 12px; opacity: 0;
  transition: opacity 0.4s ease;
}
.ps-done {
  background: rgba(34,197,94,0.06);
  border-color: rgba(34,197,94,0.2);
}
.ps-done::after { background: linear-gradient(90deg, rgba(34,197,94,0.08), transparent); opacity: 1; }
.ps-running {
  background: rgba(102,126,234,0.08);
  border-color: rgba(102,126,234,0.35);
  animation: stepPulse 1.5s ease-in-out infinite;
}
.ps-running::after { background: linear-gradient(90deg, rgba(102,126,234,0.1), transparent); opacity: 1; }
.ps-waiting { background: rgba(255,255,255,0.01); border-color: rgba(255,255,255,0.04); }
@keyframes stepPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(102,126,234,0.2); }
  50%       { box-shadow: 0 0 0 4px rgba(102,126,234,0.05); }
}

.ps-icon {
  width: 28px; height: 28px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
}
.ps-icon-done    { background: rgba(34,197,94,0.15);  color: #22c55e; }
.ps-icon-running { background: rgba(102,126,234,0.2); color: #818cf8; }
.ps-icon-waiting { background: rgba(255,255,255,0.04); color: #374151; }

.ps-label { font-size: 0.82rem; font-weight: 500; flex: 1; }
.ps-label-done    { color: #86efac; }
.ps-label-running { color: #c7d2fe; }
.ps-label-waiting { color: #374151; }

.ps-pct { font-size: 0.7rem; color: #4b5563; font-family: 'JetBrains Mono', monospace; }
.ps-pct-done    { color: #22c55e; }
.ps-pct-running { color: #667eea; }

/* ── Log Box ── */
.log-container {
  background: #030309; border: 1px solid rgba(102,126,234,0.15);
  border-radius: 14px; overflow: hidden;
}
.log-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid rgba(102,126,234,0.1);
  background: rgba(102,126,234,0.05);
}
.log-header-title { font-size: 0.72rem; color: #667eea; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.log-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; animation: logBlink 1.5s ease-in-out infinite; }
@keyframes logBlink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.log-body {
  padding: 12px 14px; font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; max-height: 280px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 2px;
}
.log-line { display: flex; gap: 8px; align-items: flex-start; line-height: 1.5; }
.log-ts   { color: #2d3748; font-size: 0.65rem; flex-shrink: 0; padding-top: 1px; }
.log-info    { color: #60a5fa; }
.log-success { color: #34d399; }
.log-warning { color: #fbbf24; }
.log-error   { color: #f87171; }

/* ── Metric Cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1rem 0; }
.metric-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding: 1.2rem 1rem; text-align: center;
  transition: all 0.3s ease; position: relative; overflow: hidden;
}
.metric-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #667eea, #f093fb);
}
.metric-card:hover {
  transform: translateY(-3px);
  border-color: rgba(102,126,234,0.2);
  box-shadow: 0 12px 32px rgba(102,126,234,0.12);
}
.metric-val {
  font-size: 2rem; font-weight: 800; line-height: 1;
  background: linear-gradient(135deg, #a78bfa, #f093fb);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.metric-lbl { font-size: 0.72rem; color: #4b5563; margin-top: 0.4rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Input Mode Tabs ── */
.input-tabs {
  display: flex; gap: 8px; margin-bottom: 1rem;
}
.input-tab {
  flex: 1; padding: 10px 16px; border-radius: 12px; text-align: center;
  font-size: 0.85rem; font-weight: 600; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02); color: #6b7280;
  transition: all 0.2s ease;
}
.input-tab.active {
  background: rgba(102,126,234,0.12);
  border-color: rgba(102,126,234,0.35); color: #a78bfa;
}

/* ── URL Input ── */
.url-wrapper {
  position: relative; margin-bottom: 1rem;
}
.url-icon {
  position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
  font-size: 1rem; z-index: 10; pointer-events: none;
}

/* ── Video Card ── */
.video-result-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px; overflow: hidden;
  transition: all 0.3s ease;
}
.video-result-card:hover {
  border-color: rgba(102,126,234,0.2);
  box-shadow: 0 8px 32px rgba(102,126,234,0.1);
  transform: translateY(-2px);
}
.video-card-header { padding: 12px 14px 8px 14px; }
.video-card-title { font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
.video-card-meta  { font-size: 0.72rem; color: #4b5563; }
.video-card-body  { padding: 0 14px; }
.video-card-footer {
  padding: 10px 14px 14px 14px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 6px;
}

.score-badge {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 8px; padding: 3px 10px; font-size: 0.72rem; font-weight: 700;
}
.score-high   { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.score-medium { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.score-low    { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ── Download Button ── */
.stDownloadButton > button {
  background: rgba(102,126,234,0.1) !important;
  border: 1px solid rgba(102,126,234,0.25) !important;
  color: #a78bfa !important; border-radius: 10px !important;
  font-size: 0.8rem !important; font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
  background: rgba(102,126,234,0.2) !important;
  border-color: rgba(102,126,234,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Complete Banner ── */
.complete-banner {
  background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.08));
  border: 1px solid rgba(34,197,94,0.25); border-radius: 20px;
  padding: 1.8rem 2rem; text-align: center; margin: 1.5rem 0;
  position: relative; overflow: hidden;
}
.complete-banner::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(34,197,94,0.8), transparent);
}
.complete-banner h2 { margin: 0 0 0.4rem 0; font-size: 1.8rem; font-weight: 800; color: #34d399; }
.complete-banner p  { margin: 0; opacity: 0.85; color: #86efac; font-size: 0.9rem; }

/* ── Progress Ring ── */
.prog-wrap { display: flex; align-items: center; gap: 14px; margin: 10px 0 14px 0; }
.prog-pct-badge {
  min-width: 52px; height: 52px; border-radius: 50%;
  background: rgba(102,126,234,0.1); border: 2px solid rgba(102,126,234,0.3);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem; font-weight: 800; color: #a78bfa;
  font-family: 'JetBrains Mono', monospace;
  flex-shrink: 0; transition: all 0.5s ease;
}
.prog-info { flex: 1; }
.prog-stage { font-size: 0.82rem; color: #94a3b8; font-weight: 500; }
.prog-bar-wrap { margin-top: 6px; }

/* ── Status Badge ── */
.status-idle       { color: #4b5563; }
.status-processing {
  color: #667eea;
  animation: statusBlink 1.5s ease-in-out infinite;
}
.status-done { color: #22c55e; }
@keyframes statusBlink { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── Divider ── */
.fancy-divider {
  width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
  margin: 1.5rem 0;
}

/* ── Sidebar compact ── */
.sidebar-section {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 14px; padding: 0.9rem;
  margin-bottom: 0.8rem;
}
.sidebar-label {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: #6366f1; margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 6px;
}
.sidebar-label::after {
  content: ''; flex: 1; height: 1px;
  background: rgba(102,126,234,0.2);
}

/* ── Status Pills ── */
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  border-radius: 50px; padding: 3px 10px; font-size: 0.7rem; font-weight: 600;
}
.pill-ok   { background: rgba(34,197,94,0.1);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.2); }
.pill-warn { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
.pill-info { background: rgba(99,102,241,0.1); color: #818cf8;  border: 1px solid rgba(99,102,241,0.2); }
.pill-err  { background: rgba(239,68,68,0.1);  color: #f87171;  border: 1px solid rgba(239,68,68,0.2); }

div[data-testid="stVideo"] video {
  border-radius: 12px; max-height: 220px !important;
  width: 100% !important; object-fit: contain;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
  border: 2px dashed rgba(102,126,234,0.2) !important;
  border-radius: 16px !important; background: rgba(102,126,234,0.02) !important;
  transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(102,126,234,0.4) !important;
  background: rgba(102,126,234,0.05) !important;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Background orbs ───────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-orbs">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)

# ── Folders ───────────────────────────────────────────────────────────────────
FOLDERS = {
    "input":        "ai_shorts_generator/input",
    "audio":        "ai_shorts_generator/audio",
    "transcripts":  "ai_shorts_generator/transcripts",
    "clips":        "ai_shorts_generator/clips",
    "final_shorts": "ai_shorts_generator/final_shorts",
    "metadata":     "ai_shorts_generator/metadata",
}
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "processing": False, "completed": False,
    "progress": 0, "stage": "", "log": [],
    "results": [], "metadata": [], "q": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Queue helpers ─────────────────────────────────────────────────────────────
def _qlog(q, msg, level="info"):
    icons = {"info": "[i]", "success": "[OK]", "warning": "[!]", "error": "[X]"}
    q.put({"type": "log", "text": f"{icons.get(level,'[i]')} {msg}", "level": level})

def _qprog(q, pct, stage=""):
    q.put({"type": "progress", "pct": pct, "stage": stage})


# ── Pipeline thread ───────────────────────────────────────────────────────────
def _pipeline_thread(source, is_youtube, num_shorts,
                     whisper_model, whisper_engine,
                     min_dur, max_dur,
                     use_captions, use_face,
                     caption_cfg, caption_mode,
                     openrouter_key, q):
    import datetime
    t_start = time.time()
    try:
        video_path = os.path.join(FOLDERS["input"], "video.mp4")
        _qprog(q, 2, "Downloading video...")
        if is_youtube:
            _qlog(q, f"Downloading YouTube video: {source}")
            video_path = download_youtube_video(source, FOLDERS["input"], progress_queue=q)
            _qlog(q, f"Download complete → {video_path}", "success")
        else:
            _qlog(q, f"Using uploaded file → {video_path}", "success")
        _qprog(q, 14, "Download complete")

        _qprog(q, 15, "Extracting audio...")
        _qlog(q, "Extracting audio with FFmpeg (WAV PCM 16kHz)...")
        audio_path = extract_audio(video_path, os.path.join(FOLDERS["audio"], "audio.wav"))
        _qlog(q, f"Audio extracted → {audio_path}", "success")
        _qprog(q, 26, "Audio extracted")

        _ENGINES = {"openai": "OpenAI-Whisper", "faster": "Faster-Whisper", "whisperx": "WhisperX"}
        engine_label = _ENGINES.get(whisper_engine, whisper_engine)
        _qprog(q, 27, f"Transcribing [{whisper_model}] via {engine_label}...")
        _qlog(q, f"Transcribing | engine={engine_label} | model={whisper_model} | lang=AUTO")
        transcript_path = os.path.join(FOLDERS["transcripts"], "transcript.json")
        transcript = transcribe_audio(audio_path, transcript_path,
                                      model_name=whisper_model, language=None, engine=whisper_engine)
        n_seg    = len(transcript.get("segments", []))
        n_words  = sum(len(s.get("words", [])) for s in transcript.get("segments", []))
        det_lang = transcript.get("language", "unknown")
        preview  = transcript.get("text", "")[:120].replace("\n", " ")
        _qlog(q, f"Transcription done — {n_seg} segs | {n_words} words | lang={det_lang}", "success")
        _qlog(q, f"Preview: {preview or '[empty — try medium/large model]'}")
        if n_seg == 0:
            _qlog(q, "EMPTY TRANSCRIPT! Try medium/large model or faster/whisperx engine.", "error")
        _qprog(q, 46, "Transcription done")

        _qprog(q, 47, "Detecting viral moments...")
        if openrouter_key and openrouter_key.strip():
            _qlog(q, "Using OpenRouter AI for viral analysis...")
        else:
            _qlog(q, "Analysing transcript locally for viral moments...")
        segments = detect_viral_segments(transcript, num_shorts, min_duration=min_dur,
                                         max_duration=max_dur, openrouter_key=openrouter_key,
                                         progress_queue=q)
        _qlog(q, f"Detected {len(segments)} viral segments", "success")
        for i, s in enumerate(segments):
            _qlog(q, f"  Clip {i+1}: {s['start_time']:.1f}s→{s['end_time']:.1f}s "
                      f"score={s['viral_score']} — {s['reason'][:60]}")
        _qprog(q, 56, "Viral segments found")

        _qprog(q, 57, "Cutting raw clips with FFmpeg (CPU)...")
        _qlog(q, "Cutting clips with FFmpeg (libx264 CPU, GPU fallback)...")
        clip_paths = generate_clips(video_path, segments, FOLDERS["clips"])
        _qlog(q, f"Cut {len(clip_paths)} raw clips", "success")
        _qprog(q, 65, "Clips cut")

        tracked_paths = clip_paths
        if use_face:
            _qprog(q, 66, "Applying face tracking + 9:16 conversion...")
            _qlog(q, "Face tracking: DNN GPU → MediaPipe → Haar cascade...")
            tracked_paths = apply_face_tracking(clip_paths, FOLDERS["clips"], progress_queue=q)
            _qlog(q, "Face tracking complete", "success")
        else:
            _qlog(q, "Face tracking skipped", "warning")
        _qprog(q, 80, "Face tracking done")

        final_paths = tracked_paths
        if use_captions:
            mode_label = "WhisperX ASS" if caption_mode == "whisperx" else "Standard Pillow"
            _qprog(q, 81, f"Rendering captions [{mode_label}]...")
            _qlog(q, f"Captions: mode={mode_label} | style={caption_cfg.get('style','Bold White')} | lang={det_lang}")
            if caption_mode == "whisperx":
                _qlog(q, "WhisperX: re-aligning audio for word-level timestamps...")
                final_paths = generate_whisperx_captions(tracked_paths, transcript, segments,
                                                          FOLDERS["final_shorts"], caption_cfg=caption_cfg,
                                                          detected_lang=det_lang, progress_queue=q)
            else:
                final_paths = generate_captions(tracked_paths, transcript, segments,
                                                 FOLDERS["final_shorts"], caption_cfg=caption_cfg,
                                                 detected_lang=det_lang, progress_queue=q)
            _qlog(q, f"Captions rendered [{mode_label}]", "success")
        else:
            _qlog(q, "Captions skipped", "warning")
            os.makedirs(FOLDERS["final_shorts"], exist_ok=True)
            final_paths = []
            for i, cp in enumerate(tracked_paths):
                dest = os.path.join(FOLDERS["final_shorts"], f"short_{i+1:03d}_final.mp4")
                shutil.copy(cp, dest); final_paths.append(dest)
        _qprog(q, 95, "Captions done")

        _qprog(q, 96, "Writing metadata...")
        metadata = []
        for i, (seg, fp) in enumerate(zip(segments, final_paths)):
            dur = seg["end_time"] - seg["start_time"]
            metadata.append({
                "index": i+1, "title": f"Short #{i+1} — {seg['reason'][:50]}",
                "description": seg.get("description", seg["reason"]),
                "viral_score": seg["viral_score"], "clip_length": round(dur, 2),
                "start_time": round(seg["start_time"], 2), "end_time": round(seg["end_time"], 2),
                "reason": seg["reason"], "output_file": str(fp),
            })
        meta_path = os.path.join(FOLDERS["metadata"], "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t_start
        _qprog(q, 100, "Done!")
        _qlog(q, f"Pipeline complete! Generated {len(final_paths)} shorts in {int(elapsed//60)}m {int(elapsed%60)}s", "success")

        print("\n" + "═"*60)
        print(f"  ✅  {len(final_paths)} shorts generated in {int(elapsed//60)}m {int(elapsed%60)}s")
        for fp in final_paths:
            sz = os.path.getsize(fp) / (1024*1024) if os.path.exists(fp) else 0
            print(f"     {fp}  ({sz:.1f} MB)")
        print("═"*60 + "\n")

        q.put({"type": "done", "metadata": metadata, "results": final_paths})

    except Exception as exc:
        import traceback
        _qlog(q, f"Pipeline error: {exc}", "error")
        _qlog(q, traceback.format_exc(), "error")
        q.put({"type": "error", "msg": str(exc)})


def _drain_queue():
    q = st.session_state.get("q")
    if q is None: return False
    done = False
    while not q.empty():
        msg = q.get_nowait()
        if msg["type"] == "log":
            st.session_state.log.append(msg)
        elif msg["type"] == "progress":
            st.session_state.progress = msg["pct"]
            st.session_state.stage    = msg.get("stage", "")
        elif msg["type"] == "done":
            st.session_state.metadata   = msg["metadata"]
            st.session_state.results    = msg["results"]
            st.session_state.completed  = True
            st.session_state.processing = False
            done = True
        elif msg["type"] == "error":
            st.session_state.processing = False
            done = True
    return done


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">
    <span>⚡</span> <span>AI-Powered · GPU-Accelerated · Local</span>
  </div>
  <h1>AI Shorts Generator</h1>
  <div class="hero-sub">
    <span class="hero-pill">🎙️ WhisperX Engine</span>
    <span class="hero-pill">🎯 Viral Detection</span>
    <span class="hero-pill">👤 Face Tracking</span>
    <span class="hero-pill">📝 Auto Captions</span>
    <span class="hero-pill">📱 9:16 Output</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # System status at top
    st.markdown('<div class="sidebar-label">⚡ System Status</div>', unsafe_allow_html=True)
    with st.container():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram     = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                st.markdown(f'<div class="pill pill-ok">🟢 GPU · {gpu_name[:20]} · {vram}MB</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pill pill-warn">🟡 CPU Mode · No CUDA GPU</div>', unsafe_allow_html=True)
        except ImportError:
            st.markdown('<div class="pill pill-err">🔴 PyTorch not installed</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            try:
                import faster_whisper
                st.markdown('<div class="pill pill-ok" style="margin-top:4px">✓ faster-whisper</div>', unsafe_allow_html=True)
            except ImportError:
                st.markdown('<div class="pill pill-info" style="margin-top:4px">○ faster-whisper</div>', unsafe_allow_html=True)
        with col_b:
            try:
                import whisperx
                st.markdown('<div class="pill pill-ok" style="margin-top:4px">✓ whisperx</div>', unsafe_allow_html=True)
            except ImportError:
                st.markdown('<div class="pill pill-info" style="margin-top:4px">○ whisperx</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── AI Analysis ──────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🤖 Viral Detection</div>', unsafe_allow_html=True)
    openrouter_key = st.text_input(
        "OpenRouter API Key", type="password",
        placeholder="sk-or-v1-...  (optional)",
        help="Free key from openrouter.ai. Leave blank for local NLP.",
    )
    if openrouter_key:
        st.markdown('<div class="pill pill-ok">✓ AI Analysis via OpenRouter</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-info">○ Local NLP (no key required)</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Transcription ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🎙️ Transcription</div>', unsafe_allow_html=True)
    whisper_engine = st.radio(
        "Engine",
        options=["openai", "faster", "whisperx"],
        format_func=lambda x: {
            "openai":   "🔵 OpenAI Whisper",
            "faster":   "🟢 Faster-Whisper  (recommended)",
            "whisperx": "🟣 WhisperX  (word-aligned)",
        }.get(x, x),
        index=1,
    )
    whisper_model = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        index=3,
    )
    st.caption("🌐 Language auto-detected from first 30s · Hindi → Hinglish captions")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Clip Settings ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">✂️ Clip Duration</div>', unsafe_allow_html=True)
    min_clip_dur = st.slider("Min (seconds)", 15, 30, 20)
    max_clip_dur = st.slider("Max (seconds)", 45, 90, 60)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Features ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🎨 Features</div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        use_captions   = st.toggle("📝 Captions",  value=True)
    with col_f2:
        use_face_track = st.toggle("👤 Face Track", value=True)

    # ── Caption Mode ──────────────────────────────────────────────────────────
    caption_mode = "standard"
    if use_captions:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">📝 Caption Engine</div>', unsafe_allow_html=True)
        caption_mode = st.radio(
            "Mode",
            options=["standard", "whisperx"],
            format_func=lambda x: (
                "⚡ Standard (Whisper)" if x == "standard" else "🔮 Advanced ASS (WhisperX)"
            ),
            index=0,
        )

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">💬 Caption Style</div>', unsafe_allow_html=True)

        caption_style = st.selectbox("Preset", [
            "Bold White", "Neon Green", "Yellow Pop", "TikTok Outlined",
            "Instagram Pill", "Karaoke Highlight", "Fire Red", "Minimal Clean",
            "Cinematic Gold", "Neon Blue", "Shadow Box", "Typewriter",
        ])
        caption_font = st.selectbox("Font", [
            "Impact", "Arial Bold", "DejaVu Bold",
            "Liberation Bold", "FreeSans Bold", "Helvetica",
        ])
        caption_font_size = st.slider("Size (px)", 36, 120, 72, step=4)

        col_tc, col_sc = st.columns(2)
        with col_tc:
            st.caption("Text")
            caption_text_color = st.color_picker("Text Color", "#FFFFFF", label_visibility="collapsed")
        with col_sc:
            st.caption("Stroke")
            caption_stroke_color = st.color_picker("Stroke Color", "#000000", label_visibility="collapsed")

        caption_stroke_width = st.slider("Stroke (px)", 0, 12, 4)
        caption_bg_enable    = st.checkbox("Background box", value=False)
        caption_bg_color     = "#000000"
        caption_bg_alpha     = 140
        if caption_bg_enable:
            caption_bg_color = st.color_picker("BG Color", "#000000")
            caption_bg_alpha = st.slider("BG Opacity", 0, 255, 140)

        caption_position = st.selectbox("Position", [
            "Lower Center (72%)", "Center (50%)", "Upper Center (25%)",
            "Bottom (85%)", "Very Bottom (92%)",
        ])
        _pos_map = {
            "Lower Center (72%)": 0.72, "Center (50%)": 0.50,
            "Upper Center (25%)": 0.25, "Bottom (85%)": 0.85, "Very Bottom (92%)": 0.92,
        }
        caption_y_frac = _pos_map[caption_position]
        caption_animation    = st.selectbox("Animation", ["None", "Word by Word", "Fade In", "Pop Scale", "Typewriter"])
        caption_uppercase    = st.checkbox("UPPERCASE", value=True)
        caption_words_per_chunk = st.slider("Words per chunk", 1, 6, 3)

        if caption_mode == "whisperx":
            st.markdown('<div class="sidebar-label" style="margin-top:8px">🔮 WhisperX ASS</div>', unsafe_allow_html=True)
            wx_bottom_margin = st.slider("Bottom margin (px)", 60, 300, 160, step=10)
            wx_fade_ms       = st.slider("Fade (ms)", 0, 200, 80, step=10)
            wx_bold          = st.checkbox("Bold", value=True)
        else:
            wx_bottom_margin, wx_fade_ms, wx_bold = 160, 80, True

        caption_cfg = {
            "style": caption_style, "font": caption_font,
            "font_size": caption_font_size, "text_color": caption_text_color,
            "stroke_color": caption_stroke_color, "stroke_width": caption_stroke_width,
            "bg_enable": caption_bg_enable, "bg_color": caption_bg_color,
            "bg_alpha": caption_bg_alpha, "y_frac": caption_y_frac,
            "animation": caption_animation, "uppercase": caption_uppercase,
            "words_per_chunk": caption_words_per_chunk,
            "bottom_margin": wx_bottom_margin, "fade_ms": wx_fade_ms, "bold": wx_bold,
        }
    else:
        caption_mode = "standard"
        caption_cfg  = {}

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📁 Output Paths</div>', unsafe_allow_html=True)
    for name, path in FOLDERS.items():
        st.code(path, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Two columns
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ── LEFT: Input & Controls ────────────────────────────────────────────────────
with col_left:

    # ── Input Card ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">📹</div>
      <div>
        <div class="section-title">Video Source</div>
        <div class="section-subtitle">YouTube link or upload your video file</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode", ["YouTube Link", "Upload File"],
        horizontal=True, label_visibility="collapsed"
    )

    youtube_url   = ""
    uploaded_file = None
    is_youtube    = False
    url_ok        = False

    if input_mode == "YouTube Link":
        is_youtube  = True
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any public YouTube video URL",
        )
        if youtube_url:
            url_ok = validate_youtube_url(youtube_url)
            if not url_ok:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;background:rgba(239,68,68,0.08);
                            border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:10px 14px;margin-top:4px">
                  <span>⚠️</span>
                  <span style="font-size:0.82rem;color:#f87171">Please enter a valid YouTube URL</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;background:rgba(34,197,94,0.08);
                            border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:10px 14px;margin-top:4px">
                  <span>✅</span>
                  <span style="font-size:0.82rem;color:#34d399">Valid YouTube URL — ready to process</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader(
            "Drop your video here",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Supports MP4, MOV, AVI, MKV, WebM",
        )
        if uploaded_file:
            video_path = os.path.join(FOLDERS["input"], "video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            url_ok = True
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;background:rgba(34,197,94,0.08);
                        border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:10px 14px;margin-top:4px">
              <span>✅</span>
              <span style="font-size:0.82rem;color:#34d399">
                Saved <strong>{uploaded_file.name}</strong> ({uploaded_file.size / (1024*1024):.1f} MB)
              </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Clips Count ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">🎯</div>
      <div>
        <div class="section-title">Output Configuration</div>
        <div class="section-subtitle">Control how many shorts to generate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    auto_detect = st.checkbox("🤖 Auto-detect optimal clip count", value=True)
    num_shorts  = None
    if not auto_detect:
        num_shorts = st.number_input("Number of shorts", min_value=1, max_value=20, value=5)
        st.caption(f"Will generate exactly **{num_shorts}** short clip{'s' if num_shorts != 1 else ''}")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Feature Summary Chips ─────────────────────────────────────────────────
    engine_labels = {"openai": "OpenAI Whisper", "faster": "Faster-Whisper", "whisperx": "WhisperX"}
    e_label = engine_labels.get(whisper_engine, whisper_engine)
    ai_label = "OpenRouter AI" if (openrouter_key and openrouter_key.strip()) else "Local NLP"
    cap_label = "WhisperX ASS" if caption_mode == "whisperx" else "Standard"

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:1.2rem">
      <span class="pill pill-info">🎙️ {e_label} · {whisper_model}</span>
      <span class="pill pill-info">🤖 {ai_label}</span>
      <span class="pill {'pill-ok' if use_captions else 'pill-warn'}">📝 Captions {'· ' + cap_label if use_captions else '· off'}</span>
      <span class="pill {'pill-ok' if use_face_track else 'pill-warn'}">👤 Face Tracking {'· on' if use_face_track else '· off'}</span>
      <span class="pill pill-info">⏱️ {min_clip_dur}s–{max_clip_dur}s clips</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Generate Button ───────────────────────────────────────────────────────
    ready = url_ok or (uploaded_file is not None)

    if st.session_state.processing:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;background:rgba(102,126,234,0.08);
                    border:1px solid rgba(102,126,234,0.2);border-radius:16px;
                    padding:16px 20px;margin-bottom:8px">
          <div style="width:10px;height:10px;border-radius:50%;background:#667eea;
                      animation:logBlink 1s infinite"></div>
          <span style="color:#818cf8;font-weight:600;font-size:0.95rem">Pipeline running… please wait</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        generate_clicked = st.button(
            "🚀  Generate Shorts" if ready else "⬆️  Add a video to begin",
            type="primary",
            use_container_width=True,
            disabled=(not ready) or st.session_state.processing,
        )

        if generate_clicked and ready and not st.session_state.processing:
            st.session_state.update({
                "log": [], "progress": 0, "stage": "Starting…",
                "completed": False, "results": [], "metadata": [],
                "processing": True,
            })
            q = queue.Queue()
            st.session_state.q = q
            src = youtube_url if is_youtube else os.path.join(FOLDERS["input"], "video.mp4")
            threading.Thread(
                target=_pipeline_thread,
                args=(src, is_youtube, num_shorts,
                      whisper_model, whisper_engine,
                      min_clip_dur, max_clip_dur,
                      use_captions, use_face_track,
                      caption_cfg, caption_mode,
                      openrouter_key, q),
                daemon=True,
            ).start()
            st.rerun()


# ── RIGHT: Live Status ────────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">📊</div>
      <div>
        <div class="section-title">Pipeline Status</div>
        <div class="section-subtitle">Real-time progress tracking</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    prog  = st.session_state.progress
    stage = st.session_state.stage

    # ── Progress ring + bar ───────────────────────────────────────────────────
    if st.session_state.processing:
        status_cls = "status-processing"
        status_txt = "● Processing"
    elif st.session_state.completed:
        status_cls = "status-done"
        status_txt = "✓ Complete"
    else:
        status_cls = "status-idle"
        status_txt = "○ Idle"

    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-pct-badge">{prog}%</div>
      <div class="prog-info">
        <div class="{status_cls}" style="font-weight:700;font-size:0.85rem">{status_txt}</div>
        <div class="prog-stage">{stage if stage else 'Waiting to start…'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    prog_bar = st.progress(prog / 100)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Step indicators ───────────────────────────────────────────────────────
    STEPS = [
        ("Download / Load Video",                    14),
        ("Extract Audio  WAV 16kHz",                 26),
        ("Transcribe  30s lang probe",               46),
        ("Detect Viral Segments",                    56),
        ("Cut Clips  CPU libx264",                   65),
        ("Face Track + 9:16  DNN→MP→Haar",           80),
        ("Render Captions",                          95),
        ("Export & Metadata",                       100),
    ]

    step_ph = st.empty()
    with step_ph.container():
        st.markdown('<div class="pipeline-wrap">', unsafe_allow_html=True)
        for idx, (label, threshold) in enumerate(STEPS):
            prev_thresh = max([t for _, t in STEPS if t < threshold], default=0)
            if prog >= threshold:
                state = "done"
                icon  = "✓"
            elif prog >= prev_thresh:
                state = "running"
                icon  = "◉"
            else:
                state = "waiting"
                icon  = str(idx + 1)

            pct_txt = f"{threshold}%" if state == "done" else ("▸" if state == "running" else "")
            st.markdown(f"""
            <div class="pipeline-step ps-{state}">
              <div class="ps-icon ps-icon-{state}">{icon}</div>
              <div class="ps-label ps-label-{state}">{label}</div>
              <div class="ps-pct ps-pct-{state}">{pct_txt}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Live Log ──────────────────────────────────────────────────────────────
    if st.session_state.log:
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        log_dot_vis = "block" if st.session_state.processing else "none"
        log_lines = ""
        for entry in st.session_state.log[-40:]:
            lvl  = entry.get("level", "info") if isinstance(entry, dict) else "info"
            text = entry.get("text", str(entry))  if isinstance(entry, dict) else str(entry)
            text_esc = text.replace("<","&lt;").replace(">","&gt;")
            log_lines += f'<div class="log-line"><span class="log-{lvl}">{text_esc}</span></div>'

        st.markdown(f"""
        <div class="log-container">
          <div class="log-header">
            <span class="log-header-title">Live Output</span>
            <div class="log-dot" style="display:{log_dot_vis}"></div>
          </div>
          <div class="log-body">{log_lines}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.processing:
    _drain_queue()
    time.sleep(0.5)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.completed and st.session_state.metadata:
    meta = st.session_state.metadata

    st.markdown(f"""
    <div class="complete-banner">
      <h2>🎉 {len(meta)} Shorts Generated!</h2>
      <p>Saved to <code style="background:rgba(255,255,255,0.1);border-radius:6px;padding:2px 8px">
        ai_shorts_generator/final_shorts/</code></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    avg_score  = sum(m["viral_score"] for m in meta) / len(meta)
    total_dur  = sum(m["clip_length"] for m in meta)
    top_score  = max(m["viral_score"] for m in meta)
    total_size = sum(
        os.path.getsize(m["output_file"]) / (1024*1024)
        for m in meta if os.path.exists(m["output_file"])
    )

    st.markdown(f"""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-val">{len(meta)}</div>
        <div class="metric-lbl">Shorts Created</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{avg_score:.0f}</div>
        <div class="metric-lbl">Avg Viral Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{total_dur:.0f}s</div>
        <div class="metric-lbl">Total Content</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{top_score}</div>
        <div class="metric-lbl">Top Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Video grid ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">🎬</div>
      <div>
        <div class="section-title">Generated Shorts</div>
        <div class="section-subtitle">Preview, review scores, and download individual clips</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols_per_row = 3
    for row_start in range(0, len(meta), cols_per_row):
        row_meta = meta[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")
        for col, m in zip(cols, row_meta):
            with col:
                score = m["viral_score"]
                s_cls = "high" if score >= 70 else ("medium" if score >= 40 else "low")
                fp    = m["output_file"]
                sz_mb = os.path.getsize(fp) / (1024*1024) if os.path.exists(fp) else 0

                st.markdown(f"""
                <div class="video-result-card">
                  <div class="video-card-header">
                    <div style="display:flex;align-items:center;justify-content:space-between">
                      <div class="video-card-title">Short #{m['index']}</div>
                      <div class="score-badge score-{s_cls}">★ {score}</div>
                    </div>
                    <div class="video-card-meta">
                      ⏱ {m['clip_length']}s &nbsp;·&nbsp;
                      {m['start_time']}s → {m['end_time']}s &nbsp;·&nbsp;
                      {sz_mb:.1f} MB
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if os.path.exists(fp):
                    st.video(fp)
                else:
                    st.info("File not found")

                reason_short = m["reason"][:65] + "…" if len(m["reason"]) > 65 else m["reason"]
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                            border-radius:10px;padding:8px 10px;margin:6px 0;font-size:0.75rem;
                            color:#94a3b8;line-height:1.4">
                  💡 {reason_short}
                </div>
                """, unsafe_allow_html=True)

                if os.path.exists(fp):
                    with open(fp, "rb") as vf:
                        st.download_button(
                            f"⬇️ Download Short #{m['index']}",
                            vf, f"short_{m['index']}.mp4", "video/mp4",
                            use_container_width=True,
                        )

        if row_start + cols_per_row < len(meta):
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── Metadata download ─────────────────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    meta_path = os.path.join(FOLDERS["metadata"], "metadata.json")
    if os.path.exists(meta_path):
        col_dl, col_info = st.columns([2, 3])
        with col_dl:
            with open(meta_path, encoding="utf-8") as f:
                meta_json = f.read()
            st.download_button(
                "📄 Download metadata.json",
                meta_json, "metadata.json", "application/json",
                use_container_width=True,
            )
        with col_info:
            st.markdown(f"""
            <div style="padding:10px;background:rgba(102,126,234,0.05);border:1px solid rgba(102,126,234,0.15);
                        border-radius:12px;font-size:0.8rem;color:#94a3b8">
              📂 All files in <code style="color:#a78bfa">ai_shorts_generator/final_shorts/</code><br>
              📊 {len(meta)} clips · {total_size:.1f} MB total · avg {avg_score:.0f}/100 viral score
            </div>
            """, unsafe_allow_html=True)