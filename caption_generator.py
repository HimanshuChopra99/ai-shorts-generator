"""
caption_generator.py  (v12)

Standard caption renderer using Pillow + FFmpeg NVENC pipe.
- Caption timing: word-level timestamps with 0.05s lookahead
- GPU encode: raw frames piped to FFmpeg NVENC subprocess
- Hindi/Devanagari -> Hinglish transliteration
- 12 styles, 5 animations, full color/font/size/position control
"""

import os, shutil, subprocess, textwrap, re
from pathlib import Path
from transcription import get_words_in_range

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
            0x4E00 <= cp <= 0x9FFF or
            0xAC00 <= cp <= 0xD7AF):
            return True
    return False


def _process_caption_text(text, uppercase=True):
    if _needs_transliteration(text):
        text = _transliterate_hindi(text)
    if uppercase and text:
        text = text.upper()
    return text.strip()


# ── Color helpers ─────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c*2 for c in hex_color)
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

def _hex_to_rgb_alpha(hex_color, alpha=140):
    r, g, b = _hex_to_rgb(hex_color)
    return (r, g, b, alpha)


# ── Style presets ─────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "Bold White":        {"fill":"#FFFFFF","stroke":"#000000","stroke_width":4,"font_size":72,"y_frac":0.72,"uppercase":True},
    "Neon Green":        {"fill":"#00FF50","stroke":"#003C00","stroke_width":3,"font_size":68,"y_frac":0.72,"uppercase":True},
    "Yellow Pop":        {"fill":"#FFE600","stroke":"#000000","stroke_width":4,"font_size":74,"y_frac":0.72,"uppercase":True},
    "TikTok Outlined":   {"fill":"#FFFFFF","stroke":"#FF0064","stroke_width":6,"font_size":70,"y_frac":0.70,"uppercase":True},
    "Instagram Pill":    {"fill":"#FFFFFF","stroke":"#000000","stroke_width":2,"font_size":64,"y_frac":0.74,"bg_enable":True,"bg_color":"#000000","bg_alpha":140,"uppercase":False},
    "Karaoke Highlight": {"fill":"#FFDC00","stroke":"#000000","stroke_width":3,"font_size":68,"y_frac":0.72,"uppercase":True,"animation":"Word by Word"},
    "Fire Red":          {"fill":"#FF3C00","stroke":"#FFB400","stroke_width":5,"font_size":72,"y_frac":0.70,"uppercase":True},
    "Minimal Clean":     {"fill":"#F0F0F0","stroke":"#1E1E1E","stroke_width":1,"font_size":56,"y_frac":0.80,"uppercase":False},
    "Cinematic Gold":    {"fill":"#FFD700","stroke":"#4A3000","stroke_width":4,"font_size":70,"y_frac":0.78,"uppercase":True},
    "Neon Blue":         {"fill":"#00CFFF","stroke":"#001A4D","stroke_width":4,"font_size":70,"y_frac":0.72,"uppercase":True},
    "Shadow Box":        {"fill":"#FFFFFF","stroke":"#000000","stroke_width":2,"font_size":68,"y_frac":0.74,"bg_enable":True,"bg_color":"#1A1A2E","bg_alpha":180,"uppercase":False},
    "Typewriter":        {"fill":"#00FF00","stroke":"#003300","stroke_width":2,"font_size":60,"y_frac":0.76,"uppercase":False,"animation":"Typewriter"},
}


def _resolve_cfg(caption_cfg):
    style  = caption_cfg.get("style", "Bold White")
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Bold White"]).copy()
    for key in ("font_size","stroke_width","y_frac","uppercase","animation"):
        if key in caption_cfg:
            preset[key] = caption_cfg[key]
    if "text_color"   in caption_cfg: preset["fill"]      = caption_cfg["text_color"]
    if "stroke_color" in caption_cfg: preset["stroke"]    = caption_cfg["stroke_color"]
    if "bg_enable"    in caption_cfg: preset["bg_enable"] = caption_cfg["bg_enable"]
    if "bg_color"     in caption_cfg: preset["bg_color"]  = caption_cfg["bg_color"]
    if "bg_alpha"     in caption_cfg: preset["bg_alpha"]  = caption_cfg["bg_alpha"]
    return preset


# ── Font finder ───────────────────────────────────────────────────────────────
_FONT_MAP = {
    "Impact": [
        "C:/Windows/Fonts/impact.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "/usr/share/fonts/truetype/impact.ttf",
    ],
    "Arial Bold": [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "DejaVu Bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
    ],
    "Liberation Bold": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/LiberationSans-Bold.ttf",
    ],
    "FreeSans Bold": ["/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"],
    "Helvetica": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}
_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

def _find_font(font_name="Impact", font_size=72):
    from PIL import ImageFont
    candidates = _FONT_MAP.get(font_name, []) + _FALLBACK_FONTS
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=font_size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_outlined(draw, x, y, text, font, fill, stroke, sw):
    for dx in range(-sw, sw + 1, max(1, sw)):
        for dy in range(-sw, sw + 1, max(1, sw)):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=stroke)
    draw.text((x, y), text, font=font, fill=fill)


def _apply_animation(draw, pil_img, text, x, y, font, fill_rgb, stroke_rgb,
                     stroke_width, animation, t, chunk_start, chunk_end):
    from PIL import Image, ImageDraw
    duration = max(chunk_end - chunk_start, 0.001)
    progress = min(1.0, (t - chunk_start) / duration)

    if animation == "Fade In":
        alpha   = int(255 * min(1.0, progress * 3))
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        odraw   = ImageDraw.Draw(overlay)
        _draw_text_outlined(odraw, x, y, text, font,
                            fill_rgb + (alpha,), stroke_rgb + (alpha,), stroke_width)
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

    elif animation == "Pop Scale":
        scale = min(1.0, 0.5 + progress * 2.5) if progress < 0.4 else 1.0
        if scale < 0.99:
            from PIL import ImageFont
            scaled_size = max(8, int(font.size * scale))
            try:
                sf = ImageFont.truetype(font.path, scaled_size)
            except Exception:
                sf = font
            _draw_text_outlined(draw, x, y, text, sf, fill_rgb, stroke_rgb, stroke_width)
        else:
            _draw_text_outlined(draw, x, y, text, font, fill_rgb, stroke_rgb, stroke_width)

    elif animation == "Typewriter":
        n_chars = max(1, int(len(text) * progress * 2))
        partial = text[:n_chars]
        _draw_text_outlined(draw, x, y, partial, font, fill_rgb, stroke_rgb, stroke_width)

    else:
        _draw_text_outlined(draw, x, y, text, font, fill_rgb, stroke_rgb, stroke_width)

    return pil_img


# ── GPU-accelerated caption renderer ─────────────────────────────────────────
def _render_captions_gpu(clip_path, output_path, caption_chunks, caption_cfg):
    from PIL import Image, ImageDraw
    import cv2, numpy as np

    cfg        = _resolve_cfg(caption_cfg)
    fill_rgb   = _hex_to_rgb(cfg["fill"])
    stroke_rgb = _hex_to_rgb(cfg["stroke"])
    sw         = cfg.get("stroke_width", 4)
    fsz        = cfg.get("font_size", 72)
    y_frac     = cfg.get("y_frac", 0.72)
    bg_enable  = cfg.get("bg_enable", False)
    bg_color   = cfg.get("bg_color", "#000000")
    bg_alpha   = cfg.get("bg_alpha", 140)
    animation  = cfg.get("animation", "None")
    font_name  = caption_cfg.get("font", "Impact")

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {clip_path}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    font   = _find_font(font_name, fsz)

    print(f"[Captions] {total} frames | {fps:.1f}fps | style={cfg.get('style','?')} | font={font_name}@{fsz}")

    caption_chunks_sorted = sorted(caption_chunks, key=lambda c: c["start"])

    try:
        from clip_generator import _NVENC_OK, ENCODER_FLAGS, ENCODER
        use_gpu  = _NVENC_OK
        enc      = ENCODER
        enc_args = ENCODER_FLAGS
    except Exception:
        use_gpu  = False
        enc      = "libx264"
        enc_args = ["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]

    tmp_raw = output_path + ".rawvid.mp4"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{src_w}x{src_h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an", "-c:v", enc,
    ] + enc_args + [tmp_raw]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = (frame_idx / fps) - 0.05
        frame_idx += 1

        active = None
        for chunk in caption_chunks_sorted:
            if chunk["start"] <= t + 0.05 < chunk["end"] + 0.05:
                active = chunk
                break

        if active:
            text = active["text"]
            pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            draw = ImageDraw.Draw(pil)

            lines  = textwrap.wrap(text, width=16)
            line_h = fsz + 12
            total_h = len(lines) * line_h
            y_start = int(src_h * y_frac) - total_h // 2

            for li, line in enumerate(lines):
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    tw   = bbox[2] - bbox[0]
                    th   = bbox[3] - bbox[1]
                except AttributeError:
                    tw = len(line) * (fsz // 2)
                    th = fsz

                x = max(10, (src_w - tw) // 2)
                y = y_start + li * line_h

                if bg_enable:
                    bg_rgba = _hex_to_rgb_alpha(bg_color, bg_alpha)
                    pad     = 16
                    bg_img  = Image.new("RGBA", pil.size, (0, 0, 0, 0))
                    bg_drw  = ImageDraw.Draw(bg_img)
                    bg_drw.rounded_rectangle(
                        [x - pad, y - pad//2, x + tw + pad, y + th + pad//2],
                        radius=14, fill=bg_rgba,
                    )
                    pil  = Image.alpha_composite(pil, bg_img)
                    draw = ImageDraw.Draw(pil)

                pil = _apply_animation(
                    draw, pil, line, x, y, font,
                    fill_rgb, stroke_rgb, sw,
                    animation, t + 0.05, active["start"], active["end"],
                )
                draw = ImageDraw.Draw(pil)

            frame_rgb = np.array(pil.convert("RGB"))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            proc.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError:
            break

    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()

    if proc.returncode != 0 or not os.path.exists(tmp_raw):
        # CPU fallback VideoWriter
        cap2 = cv2.VideoCapture(clip_path)
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        tmp_raw = output_path + ".rawvid.mp4"
        writer  = cv2.VideoWriter(tmp_raw, fourcc, fps, (src_w, src_h))
        frame_idx = 0
        while True:
            ret, frame = cap2.read()
            if not ret: break
            writer.write(frame)
            frame_idx += 1
        cap2.release()
        writer.release()

    # Merge audio
    audio_merge = [
        "ffmpeg", "-y",
        "-i", tmp_raw, "-i", clip_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        output_path,
    ]
    r = subprocess.run(audio_merge, capture_output=True, text=True)
    if r.returncode != 0:
        try:
            from clip_generator import ENCODER, ENCODER_FLAGS
            gpu_merge = [
                "ffmpeg", "-y",
                "-i", tmp_raw, "-i", clip_path,
                "-c:v", ENCODER, "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            ] + ENCODER_FLAGS + [output_path]
            r2 = subprocess.run(gpu_merge, capture_output=True, text=True)
            if r2.returncode != 0:
                shutil.copy(tmp_raw, output_path)
        except Exception:
            shutil.copy(tmp_raw, output_path)

    if os.path.exists(tmp_raw):
        os.remove(tmp_raw)
    return output_path


# ── Caption chunk builder ─────────────────────────────────────────────────────
def _build_chunks(transcript, seg_start, seg_end, clip_duration, caption_cfg):
    words_per = caption_cfg.get("words_per_chunk", 3)
    uppercase = caption_cfg.get("uppercase", True)

    words  = get_words_in_range(transcript, seg_start, seg_end)
    offset = seg_start
    chunks = []

    for i in range(0, len(words), words_per):
        group = words[i:i + words_per]
        if not group:
            continue
        start = max(0.0,           group[0]["start"] - offset - 0.05)
        end   = min(clip_duration, group[-1]["end"]  - offset + 0.05)
        if start >= end:
            end = min(clip_duration, start + 0.8)

        raw_text = " ".join(w["word"] for w in group).strip()
        text     = _process_caption_text(raw_text, uppercase=uppercase)
        if text:
            chunks.append({"text": text, "start": start, "end": end})

    return chunks


def add_captions_to_clip(clip_path, output_path, transcript,
                          seg_start, seg_end, caption_cfg, detected_lang="en"):
    import cv2
    cap = cv2.VideoCapture(clip_path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    chunks = _build_chunks(transcript, seg_start, seg_end, dur, caption_cfg)
    if not chunks:
        print(f"[Captions] No word data for {clip_path} - copying as-is")
        shutil.copy(clip_path, output_path)
        return output_path

    print(f"[Captions] {len(chunks)} chunks | lang={detected_lang} | clip_dur={dur:.1f}s")
    try:
        return _render_captions_gpu(clip_path, output_path, chunks, caption_cfg)
    except Exception as exc:
        print(f"[Captions] Renderer error ({exc}) - copying without captions")
        shutil.copy(clip_path, output_path)
        return output_path


def generate_captions(tracked_clips, transcript, segments, output_dir,
                      caption_cfg=None, detected_lang="en", progress_queue=None):
    os.makedirs(output_dir, exist_ok=True)
    if caption_cfg is None:
        caption_cfg = {}
    final_paths = []

    for i, (clip, seg) in enumerate(zip(tracked_clips, segments)):
        output = os.path.join(output_dir, f"short_{i+1:03d}_final.mp4")
        print(f"[Captions] {i+1}/{len(tracked_clips)}: {Path(clip).name}")

        if progress_queue:
            pct = 81 + int((i / len(tracked_clips)) * 13)
            progress_queue.put({
                "type":  "progress",
                "pct":   pct,
                "stage": f"Captions {i+1}/{len(tracked_clips)}...",
            })

        add_captions_to_clip(
            clip_path    = clip,
            output_path  = output,
            transcript   = transcript,
            seg_start    = seg["start_time"],
            seg_end      = seg["end_time"],
            caption_cfg  = caption_cfg,
            detected_lang = detected_lang,
        )
        final_paths.append(output)

    return final_paths
