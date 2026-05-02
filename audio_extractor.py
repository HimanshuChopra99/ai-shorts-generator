"""
audio_extractor.py  (v1)

Outputs WAV PCM 16kHz mono - Whisper native format.
GPU CUDA hwaccel for video decode (faster), CPU fallback.
"""

import os, subprocess


def _has_cuda():
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                           capture_output=True, text=True, timeout=10)
        return "cuda" in r.stdout.lower() or "nvdec" in r.stdout.lower()
    except Exception:
        return False


_CUDA_OK = _has_cuda()
print(f"[Audio] CUDA hwaccel: {'YES' if _CUDA_OK else 'NO'}")


def extract_audio(video_path, output_path, sample_rate=16000, channels=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    wav_path = os.path.splitext(output_path)[0] + ".wav"
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)

    out_args = [
        "-vn",
        "-acodec",  "pcm_s16le",
        "-ar",      str(sample_rate),
        "-ac",      str(channels),
        "-y",       wav_path,
    ]

    if _CUDA_OK:
        cmd = ["ffmpeg", "-hide_banner",
               "-hwaccel", "cuda",
               "-i", video_path] + out_args
        r   = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(wav_path):
            print(f"[Audio] WAV saved (GPU cuda hwaccel) -> {wav_path}")
            return wav_path
        print(f"[Audio] GPU decode failed (rc={r.returncode}) -> CPU fallback")

    cmd2 = ["ffmpeg", "-hide_banner", "-i", video_path] + out_args
    r2   = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n"
            f"STDERR: {r2.stderr[-600:]}"
        )

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
        raise RuntimeError(f"Audio extraction produced empty file: {wav_path}")

    print(f"[Audio] WAV saved (CPU) -> {wav_path}")
    return wav_path


def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0
