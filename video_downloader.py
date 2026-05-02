"""
video_downloader.py - yt-dlp YouTube downloader with real-time progress.
"""

import os, re, subprocess
from pathlib import Path


def validate_youtube_url(url):
    patterns = [
        r"^https?://(www\.)?youtube\.com/watch\?v=[\w-]{11}",
        r"^https?://youtu\.be/[\w-]{11}",
        r"^https?://(www\.)?youtube\.com/shorts/[\w-]{11}",
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def download_youtube_video(url, output_dir, progress_queue=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "video.mp4")

    cmd = [
        "yt-dlp",
        "--format",  "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output",  output_path,
        "--no-playlist",
        "--retries", "3",
        "--newline",
        url,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in proc.stdout:
        line = line.strip()
        if progress_queue and "[download]" in line and "%" in line:
            m = re.search(r"(\d+\.?\d*)%", line)
            if m:
                pct     = float(m.group(1))
                overall = 2 + int(pct * 0.12)
                progress_queue.put({
                    "type": "progress",
                    "pct":  overall,
                    "stage": f"Downloading... {pct:.1f}%",
                })
        print(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {proc.returncode})")

    if not os.path.exists(output_path):
        candidates = list(Path(output_dir).glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"No .mp4 in {output_dir}")
        candidates[0].rename(output_path)

    return output_path
