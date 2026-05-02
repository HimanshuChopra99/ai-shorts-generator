"""
whisperx_caption_generator.py  (v12)

Advanced captions via WhisperX ASS subtitles.
  - Word-level forced alignment (wav2vec2)
  - 2-4 word smart chunks with pause detection
  - ASS subtitle file -> FFmpeg burn-in
  - GPU encode (h264_nvenc) with CPU fallback
  - Hindi/Devanagari -> Hinglish transliteration
"""

import os, re, subprocess, shutil
from pathlib import Path


# ── Devanagari -> Hinglish transliteration ────────────────────────────────────
def _build_deva_map():
    m = {}
    m[chr(0x0905)] = "a";   m[chr(0x0906)] = "aa"
    m[chr(0x0907)] = "i";   m[chr(0x0908)] = "ee"
    m[chr(0x0909)] = "u";   m[chr(0x090A)] = "oo"
    m[chr(0x090F)] = "e";   m[chr(0x0910)] = "ai"
    m[chr(0x0913)] = "o";   m[chr(0x0914)] = "au"
    m[chr(0x0915)] = "k";   m[chr(0x0916)] = "kh"
    m[chr(0x0917)] = "g";   m[chr(0x0918)] = "gh";  m[chr(0x0919)] = "ng"
    m[chr(0x091A)] = "ch";  m[chr(0x091B)] = "chh"
    m[chr(0x091C)] = "j";   m[chr(0x091D)] = "jh";  m[chr(0x091E)] = "ny"
    m[chr(0x091F)] = "t";   m[chr(0x0920)] = "th"
    m[chr(0x0921)] = "d";   m[chr(0x0922)] = "dh";  m[chr(0x0923)] = "n"
    m[chr(0x0924)] = "t";   m[chr(0x0925)] = "th"
    m[chr(0x0926)] = "d";   m[chr(0x0927)] = "dh";  m[chr(0x0928)] = "n"
    m[chr(0x092A)] = "p";   m[chr(0x092B)] = "ph"
    m[chr(0x092C)] = "b";   m[chr(0x092D)] = "bh";  m[chr(0x092E)] = "m"
    m[chr(0x092F)] = "y";   m[chr(0x0930)] = "r";   m[chr(0x0932)] = "l"
    m[chr(0x0935)] = "v";   m[chr(0x0936)] = "sh";  m[chr(0x0937)] = "sh"
    m[chr(0x0938)] = "s";   m[chr(0x0939)] = "h"
    m[chr(0x093E)] = "a";   m[chr(0x093F)] = "i";   m[chr(0x0940)] = "ee"
    m[chr(0x0941)] = "u";   m[chr(0x0942)] = "oo";  m[chr(0x0947)] = "e"
    m[chr(0x0948)] = "ai";  m[chr(0x094B)] = "o";   m[chr(0x094C)] = "au"
    m[chr(0x0902)] = "n";   m[chr(0x0903)] = "h";   m[chr(0x094D)] = ""
    m[chr(0x0901)] = "n";   m[chr(0x0945)] = "e";   m[chr(0x094A)] = "o"
    m[chr(0x0964)] = ".";   m[" "] = " "
    m[chr(0x0966)] = "0";   m[chr(0x0967)] = "1";   m[chr(0x0968)] = "2"
    m[chr(0x0969)] = "3";   m[chr(0x096A)] = "4";   m[chr(0x096B)] = "5"
    m[chr(0x096C)] = "6";   m[chr(0x096D)] = "7";   m[chr(0x096E)] = "8"
    m[chr(0x096F)] = "9"
    return m

_DEVA_MAP = _build_deva_map()


def _transliterate_hindi(text):
    if not text:
        return text
    result = []
    i = 0
    while i < len(text):
        two = text[i:i+2] if i+1 < len(text) else ""
        if two in _DEVA_MAP:
            result.append(_DEVA_MAP[two]); i += 2
        elif text[i] in _DEVA_MAP:
            result.append(_DEVA_MAP[text[i]]); i += 1
        else:
            result.append(text[i]); i += 1
    out = "".join(result)
    out = re.sub(r" +", " ", out).strip()
    return out


def _needs_transliteration(text):
    for ch in text:
        cp = ord(ch)
        if (0x0900 <= cp <= 0x097F or
            0x0600 <= cp <= 0x06FF or
            0x4E00 <= cp <= 0x9FFF):
            return True
    return False


def _clean_caption_text(text, uppercase=True):
    if _needs_transliteration(text):
        text = _transliterate_hindi(text)
    text = text.strip()
    if uppercase:
        text = text.upper()
    return text


# ── WhisperX alignment ────────────────────────────────────────────────────────
def _align_with_whisperx(audio_path, transcript, detected_lang, device="cpu"):
    """
    Align with WhisperX for exact word-level timestamps.
    Falls back to original Whisper word timestamps if unavailable.
    """
    try:
        import whisperx
        print(f"[WhisperX] Aligning | lang={detected_lang} | device={device}")
        lang = detected_lang if detected_lang != "unknown" else "en"
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        segments = transcript.get("segments", [])
        wx_input = [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in segments]
        result = whisperx.align(wx_input, model_a, metadata, audio_path, device,
                                return_char_alignments=False)
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word  = w.get("word", "").strip()
                start = w.get("start")
                end   = w.get("end")
                if word and start is not None and end is not None:
                    words.append({"word": word, "start": float(start), "end": float(end)})
        print(f"[WhisperX] {len(words)} aligned words")
        return words
    except ImportError:
        print("[WhisperX] Not installed - falling back to Whisper timestamps")
    except Exception as e:
        print(f"[WhisperX] Alignment failed ({e}) - falling back")

    # Fallback: Whisper word timestamps
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            word  = w.get("word", "").strip()
            start = w.get("start")
            end   = w.get("end")
            if word and start is not None and end is not None:
                words.append({"word": word, "start": float(start), "end": float(end)})
    print(f"[WhisperX] Fallback: {len(words)} Whisper word timestamps")
    return words


# ── Smart chunk builder ───────────────────────────────────────────────────────
SENTENCE_ENDS = {".", "!", "?", "...", "?!", "!!", ".."}
MAX_WORDS     = 4
MIN_WORDS     = 2
PAUSE_THRESH  = 0.40


def _build_ass_chunks(words, seg_start, seg_end, clip_duration):
    seg_words = [w for w in words
                 if w["start"] >= seg_start - 0.1 and w["end"] <= seg_end + 0.1]
    if not seg_words:
        return []

    chunks  = []
    current = []

    def flush(grp):
        if not grp:
            return
        start = max(0.0, grp[0]["start"] - seg_start - 0.04)
        end   = min(clip_duration, grp[-1]["end"] - seg_start + 0.04)
        if start >= end:
            end = min(clip_duration, start + 0.5)
        text = " ".join(w["word"] for w in grp)
        text = _clean_caption_text(text, uppercase=True)
        if text:
            chunks.append({"text": text, "start": start, "end": end})

    for i, word in enumerate(seg_words):
        current.append(word)
        next_word  = seg_words[i + 1] if i + 1 < len(seg_words) else None
        pause      = (next_word["start"] - word["end"]) if next_word else 9999
        ends_sent  = any(word["word"].rstrip().endswith(p) for p in SENTENCE_ENDS)
        is_max     = len(current) >= MAX_WORDS
        is_last    = next_word is None
        should_split = (is_last or is_max or pause > PAUSE_THRESH or
                        (ends_sent and len(current) >= MIN_WORDS))
        if should_split:
            flush(current)
            current = []

    flush(current)
    return chunks


# ── ASS file generator ────────────────────────────────────────────────────────
def _secs_to_ass(secs):
    h  = int(secs // 3600)
    m  = int((secs % 3600) // 60)
    s  = int(secs % 60)
    cs = int(round((secs % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _build_ass_file(chunks, video_w=1080, video_h=1920,
                    font_name="Impact", font_size=80,
                    primary_color="&H00FFFFFF",
                    outline_color="&H00000000",
                    back_color="&H80000000",
                    outline_width=4, shadow=0, bold=True,
                    bottom_margin=160, fade_ms=80):
    bold_flag = -1 if bold else 0
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary_color},"
        f"&H000000FF,{outline_color},{back_color},"
        f"{bold_flag},0,0,0,100,100,2,0,1,{outline_width},{shadow},"
        f"2,30,30,{bottom_margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for chunk in chunks:
        start = _secs_to_ass(chunk["start"])
        end   = _secs_to_ass(chunk["end"])
        text  = chunk["text"]
        fade_tag = "{\\fad(" + str(fade_ms) + "," + str(fade_ms) + ")}"
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{fade_tag}{text}\n")
    return "".join(lines)


# ── FFmpeg burn subtitles ─────────────────────────────────────────────────────
def _burn_ass_subtitles(clip_path, ass_path, output_path):
    try:
        from clip_generator import ENCODER, ENCODER_FLAGS, _NVENC_OK
    except Exception:
        ENCODER       = "libx264"
        ENCODER_FLAGS = ["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]
        _NVENC_OK     = False

    safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")

    def _cmd(enc, flags):
        return [
            "ffmpeg", "-y", "-i", clip_path,
            "-vf", f"subtitles='{safe_ass}'",
            "-c:v", enc, "-c:a", "copy",
        ] + flags + [output_path]

    if _NVENC_OK:
        r = subprocess.run(_cmd(ENCODER, ENCODER_FLAGS), capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[WhisperX-Caps] Burned with GPU ({ENCODER})")
            return output_path
        print(f"[WhisperX-Caps] GPU burn failed -> CPU fallback")

    cpu_flags = ["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]
    r2 = subprocess.run(_cmd("libx264", cpu_flags), capture_output=True, text=True)
    if r2.returncode == 0:
        print("[WhisperX-Caps] Burned with CPU (libx264)")
        return output_path

    print("[WhisperX-Caps] Subtitle burn failed - copying without captions")
    shutil.copy(clip_path, output_path)
    return output_path


# ── Public API ────────────────────────────────────────────────────────────────
def add_whisperx_captions(clip_path, output_path, transcript,
                           seg_start, seg_end, detected_lang="en", caption_cfg=None):
    import torch, cv2
    if caption_cfg is None:
        caption_cfg = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(clip_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_dur = tot / fps if fps > 0 else (seg_end - seg_start)
    cap.release()

    tmp_wav = clip_path + ".whisperx_align.wav"
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-i", clip_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav,
        ], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp_wav):
            raise RuntimeError(f"Audio extract failed: {r.stderr[-200:]}")
        words = _align_with_whisperx(tmp_wav, transcript, detected_lang, device)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    chunks = _build_ass_chunks(words, seg_start, seg_end, clip_dur)
    if not chunks:
        shutil.copy(clip_path, output_path)
        return output_path

    print(f"[WhisperX-Caps] {len(chunks)} chunks | lang={detected_lang} | device={device}")

    tmp_ass = clip_path + ".whisperx.ass"
    try:
        def _hex_to_ass(h, alpha=0):
            h = h.lstrip("#")
            if len(h) == 3:
                h = "".join(c*2 for c in h)
            r2, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            aa = format(alpha, "02X")
            return f"&H{aa}{b:02X}{g:02X}{r2:02X}"

        vid_w = int(cv2.VideoCapture(clip_path).get(3)) or 1080
        vid_h = int(cv2.VideoCapture(clip_path).get(4)) or 1920

        ass_content = _build_ass_file(
            chunks,
            video_w       = vid_w,
            video_h       = vid_h,
            font_name     = caption_cfg.get("font", "Impact"),
            font_size     = caption_cfg.get("font_size", 80),
            primary_color = _hex_to_ass(caption_cfg.get("text_color", "#FFFFFF"), 0),
            outline_color = _hex_to_ass(caption_cfg.get("stroke_color", "#000000"), 0),
            outline_width = caption_cfg.get("stroke_width", 4),
            bold          = caption_cfg.get("bold", True),
            bottom_margin = caption_cfg.get("bottom_margin", 160),
            fade_ms       = caption_cfg.get("fade_ms", 80),
        )
        with open(tmp_ass, "w", encoding="utf-8") as fh:
            fh.write(ass_content)

        return _burn_ass_subtitles(clip_path, tmp_ass, output_path)
    finally:
        if os.path.exists(tmp_ass):
            try:
                os.remove(tmp_ass)
            except Exception:
                pass


def generate_whisperx_captions(tracked_clips, transcript, segments, output_dir,
                                caption_cfg=None, detected_lang="en", progress_queue=None):
    os.makedirs(output_dir, exist_ok=True)
    if caption_cfg is None:
        caption_cfg = {}

    final_paths = []
    for i, (clip, seg) in enumerate(zip(tracked_clips, segments)):
        output = os.path.join(output_dir, f"short_{i+1:03d}_final.mp4")
        print(f"[WhisperX-Caps] {i+1}/{len(tracked_clips)}: {Path(clip).name}")

        if progress_queue:
            pct = 81 + int((i / len(tracked_clips)) * 13)
            progress_queue.put({"type": "progress", "pct": pct,
                                "stage": f"WhisperX captions {i+1}/{len(tracked_clips)}..."})

        try:
            add_whisperx_captions(
                clip_path     = clip,
                output_path   = output,
                transcript    = transcript,
                seg_start     = seg["start_time"],
                seg_end       = seg["end_time"],
                detected_lang = detected_lang,
                caption_cfg   = caption_cfg,
            )
        except Exception as e:
            import traceback
            print(f"[WhisperX-Caps] Error on clip {i+1}: {e}")
            traceback.print_exc()
            shutil.copy(clip, output)

        final_paths.append(output)
    return final_paths
