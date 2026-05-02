"""
AI Shorts Generator - Main Streamlit App  (v12)

Run with:  python -m streamlit run app.py

v12 Changes:
  - WhisperX added as third transcription engine option
  - All engines use 30-second language probe (fast auto-detect)
  - OpenCV DNN face detector (GPU) -> MediaPipe -> Haar fallback
  - FFmpeg clip cutting: CPU first (libx264), GPU fallback
  - Split-screen for exactly 2 faces (top/bottom)
  - Advanced Captions (WhisperX ASS) + Standard Captions
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

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 1.8rem 2rem; border-radius: 20px; text-align: center;
    color: white; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
  }
  .main-header h1 { font-size: 2.2rem; margin: 0 0 0.3rem 0; }
  .main-header p  { font-size: 0.95rem; opacity: .88; margin: 0; }

  .step-card {
    background: #1e1e2e; border: 1px solid #2d2d3f; border-radius: 12px;
    padding: 0.75rem 1rem; margin-bottom: .45rem;
    display: flex; align-items: center; gap: .6rem;
  }
  .step-done    { border-left: 4px solid #22c55e; }
  .step-running { border-left: 4px solid #f59e0b; animation: pulse 1s infinite; }
  .step-waiting { border-left: 4px solid #374151; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }

  .log-box {
    background: #0d1117; border: 1px solid #21262d; border-radius: 10px;
    padding: .9rem; font-family: 'Courier New', monospace; font-size: .76rem;
    max-height: 320px; overflow-y: auto; color: #e6edf3;
  }
  .log-info    { color: #79c0ff; }
  .log-success { color: #56d364; }
  .log-warning { color: #e3b341; }
  .log-error   { color: #f85149; }

  .metric-card {
    background: #1e1e2e; border: 1px solid #2d2d3f; border-radius: 14px;
    padding: 1rem; text-align: center;
  }
  .metric-val  { font-size: 1.9rem; font-weight: 700; color: #a78bfa; }
  .metric-lbl  { font-size: .78rem; color: #6b7280; margin-top: .2rem; }

  .viral-high   { color: #22c55e; font-weight: 700; font-size: 1.5rem; }
  .viral-medium { color: #f59e0b; font-weight: 700; font-size: 1.5rem; }
  .viral-low    { color: #ef4444; font-weight: 700; font-size: 1.5rem; }

  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 1.05rem !important; padding: .75rem 1.5rem !important;
  }
  div[data-testid="stProgress"] > div > div { background-color: #667eea !important; }

  div[data-testid="stVideo"] video {
    max-height: 200px !important;
    width: auto !important;
    max-width: 135px !important;
    object-fit: contain !important;
    border-radius: 10px;
  }

  .complete-banner {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    padding: 1.2rem 1.8rem; border-radius: 16px; text-align: center;
    color: white; margin: 1rem 0;
    box-shadow: 0 6px 20px rgba(34,197,94,0.35);
  }
  .complete-banner h2 { margin: 0 0 0.3rem 0; font-size: 1.6rem; }
  .complete-banner p  { margin: 0; opacity: .9; }
</style>
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
    "processing": False,
    "completed":  False,
    "progress":   0,
    "stage":      "",
    "log":        [],
    "results":    [],
    "metadata":   [],
    "q":          None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _qlog(q, msg, level="info"):
    icons = {"info": "[i]", "success": "[OK]", "warning": "[!]", "error": "[X]"}
    q.put({"type": "log", "text": f"{icons.get(level,'[i]')} {msg}", "level": level})

def _qprog(q, pct, stage=""):
    q.put({"type": "progress", "pct": pct, "stage": stage})


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

        # Step 1 - Download / copy
        _qprog(q, 2, "Downloading video...")
        if is_youtube:
            _qlog(q, f"Downloading YouTube video: {source}")
            video_path = download_youtube_video(source, FOLDERS["input"], progress_queue=q)
            _qlog(q, f"Download complete -> {video_path}", "success")
        else:
            _qlog(q, f"Using uploaded file -> {video_path}", "success")
        _qprog(q, 14, "Download complete")

        # Step 2 - Extract audio
        _qprog(q, 15, "Extracting audio...")
        _qlog(q, "Extracting audio with FFmpeg (WAV PCM 16kHz) ...")
        audio_path = extract_audio(
            video_path,
            os.path.join(FOLDERS["audio"], "audio.wav"),
        )
        _qlog(q, f"Audio extracted -> {audio_path}", "success")
        _qprog(q, 26, "Audio extracted")

        # Step 3 - Transcribe
        _ENGINES = {"openai": "OpenAI-Whisper", "faster": "Faster-Whisper", "whisperx": "WhisperX"}
        engine_label = _ENGINES.get(whisper_engine, whisper_engine)
        _qprog(q, 27, f"Transcribing [{whisper_model}] via {engine_label} ...")
        _qlog(q, f"Transcribing | engine={engine_label} | model={whisper_model} | lang=AUTO (30s probe) ...")
        transcript_path = os.path.join(FOLDERS["transcripts"], "transcript.json")
        transcript = transcribe_audio(
            audio_path, transcript_path,
            model_name = whisper_model,
            language   = None,
            engine     = whisper_engine,
        )
        n_seg    = len(transcript.get("segments", []))
        n_words  = sum(len(s.get("words", [])) for s in transcript.get("segments", []))
        det_lang = transcript.get("language", "unknown")
        preview  = transcript.get("text", "")[:120].replace("\n", " ")
        _qlog(q,
              f"Transcription done - {n_seg} segs | {n_words} words | "
              f"lang={det_lang} | engine={engine_label}", "success")
        _qlog(q, f"Preview: {preview or '[empty - try medium/large model]'}")
        if n_seg == 0:
            _qlog(q,
                  "EMPTY TRANSCRIPT! Try:\n"
                  "  1. Switch to 'medium' or 'large-v3' Whisper model\n"
                  "  2. Use 'faster' or 'whisperx' engine (better VAD for Hindi)\n"
                  "  3. Check audio: ffplay ai_shorts_generator/audio/audio.wav",
                  "error")
        _qprog(q, 46, "Transcription done")

        # Step 4 - Viral detection
        _qprog(q, 47, "Detecting viral moments...")
        if openrouter_key and openrouter_key.strip():
            _qlog(q, "Using OpenRouter AI for viral analysis...")
        else:
            _qlog(q, "Analysing transcript locally for viral moments...")
        segments = detect_viral_segments(
            transcript, num_shorts,
            min_duration   = min_dur,
            max_duration   = max_dur,
            openrouter_key = openrouter_key,
            progress_queue = q,
        )
        _qlog(q, f"Detected {len(segments)} viral segments", "success")
        for i, s in enumerate(segments):
            _qlog(q, f"  Clip {i+1}: {s['start_time']:.1f}s->{s['end_time']:.1f}s "
                      f"score={s['viral_score']} - {s['reason'][:60]}")
        _qprog(q, 56, "Viral segments found")

        # Step 5 - Cut clips (CPU first)
        _qprog(q, 57, "Cutting raw clips with FFmpeg (CPU) ...")
        _qlog(q, "Cutting clips with FFmpeg (libx264 CPU, GPU fallback) ...")
        clip_paths = generate_clips(video_path, segments, FOLDERS["clips"])
        _qlog(q, f"Cut {len(clip_paths)} raw clips", "success")
        _qprog(q, 65, "Clips cut")

        # Step 6 - Face tracking
        tracked_paths = clip_paths
        if use_face:
            _qprog(q, 66, "Applying face tracking + 9:16 conversion ...")
            _qlog(q, "Face tracking: DNN GPU -> MediaPipe -> Haar cascade, split-screen for 2 faces ...")
            tracked_paths = apply_face_tracking(clip_paths, FOLDERS["clips"], progress_queue=q)
            _qlog(q, "Face tracking complete", "success")
        else:
            _qlog(q, "Face tracking skipped", "warning")
        _qprog(q, 80, "Face tracking done")

        # Step 7 - Captions
        final_paths = tracked_paths
        if use_captions:
            mode_label = "WhisperX ASS" if caption_mode == "whisperx" else "Standard Pillow"
            _qprog(q, 81, f"Rendering captions [{mode_label}] ...")
            _qlog(q,
                  f"Captions: mode={mode_label} | "
                  f"style={caption_cfg.get('style','Bold White')} | "
                  f"lang={det_lang} | "
                  f"transliterate={'yes' if det_lang not in ('en','english') else 'no'} ...")

            if caption_mode == "whisperx":
                _qlog(q, "WhisperX: re-aligning audio for word-level timestamps...")
                final_paths = generate_whisperx_captions(
                    tracked_paths, transcript, segments,
                    FOLDERS["final_shorts"],
                    caption_cfg    = caption_cfg,
                    detected_lang  = det_lang,
                    progress_queue = q,
                )
            else:
                final_paths = generate_captions(
                    tracked_paths, transcript, segments,
                    FOLDERS["final_shorts"],
                    caption_cfg    = caption_cfg,
                    detected_lang  = det_lang,
                    progress_queue = q,
                )
            _qlog(q, f"Captions rendered [{mode_label}]", "success")
        else:
            _qlog(q, "Captions skipped", "warning")
            os.makedirs(FOLDERS["final_shorts"], exist_ok=True)
            final_paths = []
            for i, cp in enumerate(tracked_paths):
                dest = os.path.join(FOLDERS["final_shorts"], f"short_{i+1:03d}_final.mp4")
                shutil.copy(cp, dest)
                final_paths.append(dest)
        _qprog(q, 95, "Captions done")

        # Step 8 - Metadata
        _qprog(q, 96, "Writing metadata...")
        metadata = []
        for i, (seg, fp) in enumerate(zip(segments, final_paths)):
            dur = seg["end_time"] - seg["start_time"]
            metadata.append({
                "index":       i + 1,
                "title":       f"Short #{i+1} - {seg['reason'][:50]}",
                "description": seg.get("description", seg["reason"]),
                "viral_score": seg["viral_score"],
                "clip_length": round(dur, 2),
                "start_time":  round(seg["start_time"], 2),
                "end_time":    round(seg["end_time"], 2),
                "reason":      seg["reason"],
                "output_file": str(fp),
            })

        meta_path = os.path.join(FOLDERS["metadata"], "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t_start
        mins    = int(elapsed // 60)
        secs    = int(elapsed % 60)

        _qprog(q, 100, "Done!")
        _qlog(q, f"Pipeline complete! Generated {len(final_paths)} shorts", "success")

        # ── Terminal summary ──────────────────────────────────────────────────
        print("")
        print("=" * 62)
        print("  AI SHORTS GENERATOR v12 - PIPELINE COMPLETE")
        print("=" * 62)
        print(f"  Shorts generated  : {len(final_paths)}")
        print(f"  Language detected : {det_lang}")
        print(f"  Whisper engine    : {engine_label} ({whisper_model})")
        print(f"  Caption mode      : {'WhisperX ASS' if caption_mode == 'whisperx' else 'Standard Pillow'}")
        print(f"  Total time        : {mins}m {secs}s")
        print("")
        print("  Output files:")
        for fp in final_paths:
            size_mb = os.path.getsize(fp) / (1024*1024) if os.path.exists(fp) else 0
            print(f"    {fp}  ({size_mb:.1f} MB)")
        print("")
        print(f"  Metadata : {meta_path}")
        print(f"  Open     : {FOLDERS['final_shorts']}")
        print("=" * 62)
        print("")

        q.put({"type": "done", "metadata": metadata, "results": final_paths})

    except Exception as exc:
        import traceback
        _qlog(q, f"Pipeline error: {exc}", "error")
        _qlog(q, traceback.format_exc(), "error")
        print(f"\n[PIPELINE ERROR] {exc}")
        print(traceback.format_exc())
        q.put({"type": "error", "msg": str(exc)})


def _drain_queue():
    q = st.session_state.get("q")
    if q is None:
        return False
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


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎬 AI Shorts Generator <span style="font-size:1rem;opacity:.8">v12</span></h1>
  <p>Long video to Viral Shorts &nbsp;|&nbsp; 100% Local &nbsp;|&nbsp; GPU-Accelerated &nbsp;|&nbsp; WhisperX Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # OpenRouter
    st.subheader("🤖 AI Viral Detection")
    openrouter_key = st.text_input(
        "OpenRouter API Key (optional)",
        type="password",
        placeholder="sk-or-v1-...",
        help="Free key from openrouter.ai. Leave blank for local NLP.",
    )
    if openrouter_key:
        st.success("AI analysis enabled (OpenRouter)")
    else:
        st.info("Local NLP mode (no API key needed)")

    st.divider()

    # Whisper engine
    st.subheader("🎙️ Transcription Engine")
    whisper_engine = st.radio(
        "Engine",
        options=["openai", "faster", "whisperx"],
        format_func=lambda x: {
            "openai":   "OpenAI Whisper (original)",
            "faster":   "Faster-Whisper / CTranslate2 (recommended)",
            "whisperx": "WhisperX (word-aligned, best accuracy)",
        }.get(x, x),
        index=1,
        help=(
            "All engines auto-detect language from first 30s.\n"
            "Faster-Whisper: 4x faster, best for Hindi VAD.\n"
            "WhisperX: word-level timestamps, slowest but most accurate."
        ),
    )

    # Show availability badge for whisperx
    if whisper_engine == "whisperx":
        try:
            import whisperx  # noqa: F401
            st.success("WhisperX installed")
        except ImportError:
            st.error("WhisperX not installed\npip install whisperx")

    whisper_model = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        index=3,
        help=(
            "tiny/base  - fast, mostly English\n"
            "medium     - recommended for Hindi/multilingual\n"
            "large-v3   - best accuracy (slowest)"
        ),
    )
    st.info("Language auto-detected (first 30s probe). Hindi -> Hinglish captions.")

    st.divider()

    # Clip settings
    st.subheader("✂️ Clip Settings")
    min_clip_dur = st.slider("Min clip duration (s)", 15, 30, 20)
    max_clip_dur = st.slider("Max clip duration (s)", 45, 90, 60)

    st.divider()

    # Features
    st.subheader("🎨 Features")
    use_captions   = st.toggle("Captions",      value=True)
    use_face_track = st.toggle("Face Tracking", value=True)

    # Caption mode
    caption_mode = "standard"
    if use_captions:
        st.subheader("📝 Caption Mode")
        caption_mode = st.radio(
            "Caption engine",
            options=["standard", "whisperx"],
            format_func=lambda x: (
                "Standard Captions (Whisper)"
                if x == "standard"
                else "Advanced Captions (WhisperX ASS)"
            ),
            index=0,
        )
        if caption_mode == "whisperx":
            try:
                import whisperx  # noqa: F401
                st.success("WhisperX ready for captions")
            except ImportError:
                st.warning("whisperx not installed - will fall back to Whisper timestamps")

    st.divider()

    # Caption config
    if use_captions:
        st.subheader("💬 Caption Style")

        caption_style = st.selectbox(
            "Style Preset",
            [
                "Bold White", "Neon Green", "Yellow Pop", "TikTok Outlined",
                "Instagram Pill", "Karaoke Highlight", "Fire Red", "Minimal Clean",
                "Cinematic Gold", "Neon Blue", "Shadow Box", "Typewriter",
            ],
        )

        caption_font = st.selectbox(
            "Font",
            ["Impact", "Arial Bold", "DejaVu Bold", "Liberation Bold", "FreeSans Bold", "Helvetica"],
        )

        caption_font_size = st.slider("Font Size", 36, 120, 72, step=4)

        col_tc, col_sc = st.columns(2)
        with col_tc:
            st.markdown("**Text Color**")
            caption_text_color = st.color_picker("Text", "#FFFFFF", label_visibility="collapsed")
        with col_sc:
            st.markdown("**Stroke Color**")
            caption_stroke_color = st.color_picker("Stroke", "#000000", label_visibility="collapsed")

        caption_stroke_width = st.slider("Stroke Width (px)", 0, 12, 4)

        caption_bg_enable = st.checkbox("Background behind text", value=False)
        caption_bg_color  = "#000000"
        caption_bg_alpha  = 140
        if caption_bg_enable:
            caption_bg_color = st.color_picker("Background Color", "#000000")
            caption_bg_alpha = st.slider("Background Opacity", 0, 255, 140)

        caption_position = st.selectbox(
            "Caption Position",
            ["Lower Center (72%)", "Center (50%)", "Upper Center (25%)",
             "Bottom (85%)", "Very Bottom (92%)"],
        )
        _pos_map = {
            "Lower Center (72%)":  0.72,
            "Center (50%)":        0.50,
            "Upper Center (25%)":  0.25,
            "Bottom (85%)":        0.85,
            "Very Bottom (92%)":   0.92,
        }
        caption_y_frac = _pos_map[caption_position]

        caption_animation = st.selectbox(
            "Animation",
            ["None", "Word by Word", "Fade In", "Pop Scale", "Typewriter"],
        )

        caption_uppercase     = st.checkbox("UPPERCASE text", value=True)
        caption_words_per_chunk = st.slider("Words per caption chunk", 1, 6, 3)

        # WhisperX-specific controls
        if caption_mode == "whisperx":
            st.markdown("**WhisperX ASS Style**")
            wx_bottom_margin = st.slider("Bottom margin (px)", 60, 300, 160, step=10)
            wx_fade_ms       = st.slider("Fade duration (ms)", 0, 200, 80, step=10)
            wx_bold          = st.checkbox("Bold text", value=True)
        else:
            wx_bottom_margin = 160
            wx_fade_ms       = 80
            wx_bold          = True

        caption_cfg = {
            "style":           caption_style,
            "font":            caption_font,
            "font_size":       caption_font_size,
            "text_color":      caption_text_color,
            "stroke_color":    caption_stroke_color,
            "stroke_width":    caption_stroke_width,
            "bg_enable":       caption_bg_enable,
            "bg_color":        caption_bg_color,
            "bg_alpha":        caption_bg_alpha,
            "y_frac":          caption_y_frac,
            "animation":       caption_animation,
            "uppercase":       caption_uppercase,
            "words_per_chunk": caption_words_per_chunk,
            "bottom_margin":   wx_bottom_margin,
            "fade_ms":         wx_fade_ms,
            "bold":            wx_bold,
        }
    else:
        caption_mode = "standard"
        caption_cfg  = {}

    st.divider()

    # System status
    st.subheader("System Status")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            st.success(f"CUDA GPU: {gpu_name} ({vram} MB)")
        else:
            st.warning("No CUDA GPU - CPU mode")
    except ImportError:
        st.error("PyTorch not installed")

    try:
        from clip_generator import _NVENC_OK, ENCODER
        if _NVENC_OK:
            st.success(f"FFmpeg nvenc: available (GPU fallback ready)")
        else:
            st.info("FFmpeg: libx264 CPU (nvenc not available)")
    except Exception:
        pass

    try:
        import faster_whisper
        st.success("faster-whisper: installed")
    except ImportError:
        st.info("faster-whisper: not installed (optional)")

    try:
        import whisperx
        st.success("whisperx: installed")
    except ImportError:
        st.info("whisperx: not installed (optional)")

    st.divider()
    st.markdown("**Output Folders**")
    for name, path in FOLDERS.items():
        st.code(path, language=None)


# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("📹 Video Input")
    input_mode = st.radio("Input mode", ["YouTube Link", "Upload File"], horizontal=True)

    youtube_url   = ""
    uploaded_file = None
    is_youtube    = False
    url_ok        = False

    if input_mode == "YouTube Link":
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        is_youtube = True
        if youtube_url:
            url_ok = validate_youtube_url(youtube_url)
            if not url_ok:
                st.error("Please enter a valid YouTube URL")
            else:
                st.success("URL looks valid")
    else:
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
        )
        if uploaded_file:
            video_path = os.path.join(FOLDERS["input"], "video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Saved -> {video_path}")
            url_ok = True

    st.subheader("🎯 Number of Shorts")
    auto_detect = st.checkbox("Auto-detect best number of clips", value=True)
    num_shorts  = None
    if not auto_detect:
        num_shorts = st.number_input("How many shorts?", 1, 20, 5)

    ready = url_ok or (uploaded_file is not None)

    generate_clicked = st.button(
        "🚀 Generate Shorts",
        type="primary",
        use_container_width=True,
        disabled=(not ready) or st.session_state.processing,
    )

    if generate_clicked and ready and not st.session_state.processing:
        st.session_state.log        = []
        st.session_state.progress   = 0
        st.session_state.stage      = "Starting..."
        st.session_state.completed  = False
        st.session_state.results    = []
        st.session_state.metadata   = []
        st.session_state.processing = True

        q = queue.Queue()
        st.session_state.q = q

        src = youtube_url if is_youtube else os.path.join(FOLDERS["input"], "video.mp4")

        t = threading.Thread(
            target=_pipeline_thread,
            args=(src, is_youtube, num_shorts,
                  whisper_model, whisper_engine,
                  min_clip_dur, max_clip_dur,
                  use_captions, use_face_track,
                  caption_cfg, caption_mode,
                  openrouter_key, q),
            daemon=True,
        )
        t.start()
        st.rerun()


# ── Right column: live status ─────────────────────────────────────────────────
with col_right:
    st.subheader("📊 Live Pipeline Status")

    STEPS = [
        ("Download / Load Video",               14),
        ("Extract Audio (WAV 16kHz)",           26),
        ("Transcribe (30s lang probe)",          46),
        ("Detect Viral Segments",               56),
        ("Cut Clips (CPU libx264)",             65),
        ("Face Track + 9:16 (DNN GPU->MP->Haar)", 80),
        ("Render Captions",                     95),
        ("Export & Metadata",                  100),
    ]

    prog  = st.session_state.progress
    stage = st.session_state.stage

    prog_bar  = st.progress(prog / 100)
    prog_text = st.empty()
    prog_text.caption(
        f"{'Processing... ' if st.session_state.processing else ''}{prog}%  {stage}"
    )

    step_placeholder = st.empty()
    with step_placeholder.container():
        for label, threshold in STEPS:
            prev = max([t for _, t in STEPS if t < threshold], default=0)
            if prog >= threshold:
                icon, css = "✅", "step-done"
            elif prog >= prev:
                icon, css = "🔄", "step-running"
            else:
                icon, css = "⏳", "step-waiting"
            st.markdown(
                f'<div class="step-card {css}">'
                f'<span style="color:#e2e8f0;font-size:.85rem">{icon} {label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.log:
        st.markdown("**📋 Live Log**")
        log_html = "<div class='log-box'>"
        for entry in st.session_state.log[-50:]:
            lvl  = entry.get("level", "info") if isinstance(entry, dict) else "info"
            text = entry.get("text", str(entry)) if isinstance(entry, dict) else str(entry)
            text_esc = text.replace("<", "&lt;").replace(">", "&gt;")
            log_html += f'<div class="log-{lvl}">{text_esc}</div>'
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.processing:
    _drain_queue()
    time.sleep(0.6)
    st.rerun()


# ── Results section ───────────────────────────────────────────────────────────
if st.session_state.completed and st.session_state.metadata:
    meta = st.session_state.metadata

    st.markdown(f"""
    <div class="complete-banner">
      <h2>🎉 Done! {len(meta)} Shorts Generated</h2>
      <p>All clips saved to <strong>ai_shorts_generator/final_shorts/</strong></p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(meta),                                       "Clips"),
        (c2, f"{sum(m['viral_score'] for m in meta)/len(meta):.0f}", "Avg Score"),
        (c3, f"{sum(m['clip_length'] for m in meta):.0f}s",   "Total Content"),
        (c4, max(meta, key=lambda x: x['viral_score'])['viral_score'], "Top Score"),
    ]:
        with col:
            col.markdown(
                f'<div class="metric-card"><div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    cols_per_row = 3
    for row_start in range(0, len(meta), cols_per_row):
        row_meta = meta[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, m in zip(cols, row_meta):
            with col:
                score = m["viral_score"]
                cls   = "high" if score >= 70 else ("medium" if score >= 40 else "low")
                st.markdown(f"**Short #{m['index']}**")
                st.markdown(
                    f'<div class="viral-{cls}" style="font-size:1rem">Score: {score}/100</div>',
                    unsafe_allow_html=True,
                )
                fp = m["output_file"]
                if os.path.exists(fp):
                    st.video(fp)
                else:
                    st.info("File not found")
                st.caption(f"{m['clip_length']}s | {m['start_time']}s -> {m['end_time']}s")
                st.caption(m["reason"][:60])
                if os.path.exists(fp):
                    with open(fp, "rb") as vf:
                        st.download_button(
                            "⬇️ Download",
                            vf, f"short_{m['index']}.mp4", "video/mp4",
                            use_container_width=True,
                        )

    meta_path = os.path.join(FOLDERS["metadata"], "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            st.download_button(
                "📄 Download metadata.json", f.read(),
                "metadata.json", "application/json",
            )
