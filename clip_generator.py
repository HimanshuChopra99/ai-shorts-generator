"""
clip_generator.py  (v12)

FFmpeg clip cutting strategy:
  - CPU FIRST  (libx264) for cutting - more compatible, avoids seek drift
  - GPU NVENC  as fallback for re-encode when available
  - This matches what users expect and avoids nvenc seek issues

GPU strategy for re-encode / vertical convert:
  1. Check ffmpeg -encoders for h264_nvenc
  2. Smoke-test with minimal flags
  3. If fails -> CPU libx264 fallback
"""

import os, subprocess, shutil, tempfile
from pathlib import Path


def _probe_nvenc():
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    )
    if "h264_nvenc" not in probe.stdout:
        print("[FFmpeg] h264_nvenc not compiled in -> libx264")
        return False

    tmp = tempfile.mktemp(suffix=".mp4")
    test = subprocess.run([
        "ffmpeg", "-y",
        "-f",       "lavfi",
        "-i",       "color=c=black:s=256x144:r=30:d=1",
        "-c:v",     "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-an",      tmp,
    ], capture_output=True, text=True, timeout=30)

    try: os.remove(tmp)
    except OSError: pass

    if test.returncode == 0:
        print("[FFmpeg] h264_nvenc smoke-test PASSED -> GPU encode available")
        return True

    print("[FFmpeg] h264_nvenc smoke-test FAILED -> libx264 fallback")
    print("  " + (test.stderr or "")[-300:])
    return False


def _probe_nvenc_flags():
    tmp = tempfile.mktemp(suffix=".mp4")
    flag_sets = [
        (["-preset", "p4", "-rc:v", "cbr",  "-b:v", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "-gpu", "0"], "p4 cbr gpu:0"),
        (["-preset", "p4", "-rc:v", "cbr",  "-b:v", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p"],               "p4 cbr"),
        (["-preset", "p4", "-rc:v", "vbr",  "-cq", "23",  "-pix_fmt", "yuv420p"],                                   "p4 vbr"),
        (["-rc:v",   "vbr","-cq",   "23",   "-pix_fmt", "yuv420p"],                                                  "vbr cq23"),
        (["-rc:v",   "cbr", "-b:v", "6M",   "-pix_fmt", "yuv420p"],                                                  "cbr 6M"),
        (["-pix_fmt", "yuv420p"],                                                                                      "minimal"),
    ]
    for flags, label in flag_sets:
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=256x144:r=30:d=1",
            "-c:v", "h264_nvenc",
        ] + flags + ["-an", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try: os.remove(tmp)
        except OSError: pass
        if r.returncode == 0:
            print(f"[FFmpeg] nvenc flags: {label}")
            return flags
    return ["-pix_fmt", "yuv420p"]


_NVENC_OK = _probe_nvenc()

if _NVENC_OK:
    ENCODER       = "h264_nvenc"
    ENCODER_FLAGS = _probe_nvenc_flags()
else:
    ENCODER       = "libx264"
    ENCODER_FLAGS = ["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]

print(f"[FFmpeg] Encoder: {ENCODER} | Flags: {ENCODER_FLAGS}")

_CPU_FLAGS = ["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]


def _run_cpu_first(cpu_cmd, gpu_cmd, label):
    """
    Try CPU first (more compatible for cutting), GPU as fallback.
    """
    r = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        return True  # True = used CPU
    print(f"[FFmpeg] {label}: CPU failed (rc={r.returncode}) -> GPU fallback")
    if r.stderr:
        print(f"  CPU stderr: {r.stderr[-200:]}")

    if gpu_cmd:
        r2 = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=600)
        if r2.returncode == 0:
            return False  # False = used GPU
        raise RuntimeError(f"[FFmpeg] {label} both CPU and GPU failed:\n{r2.stderr[-500:]}")
    raise RuntimeError(f"[FFmpeg] {label} CPU failed:\n{r.stderr[-500:]}")


def _run_gpu_first(gpu_cmd, cpu_cmd, label):
    """
    Try GPU first, CPU fallback - for re-encode operations.
    """
    if gpu_cmd:
        try:
            r = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                return False  # used GPU
            print(f"[FFmpeg] {label}: GPU failed -> CPU fallback")
        except subprocess.TimeoutExpired:
            print(f"[FFmpeg] {label}: GPU timeout -> CPU fallback")

    r2 = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=600)
    if r2.returncode != 0:
        raise RuntimeError(f"[FFmpeg] {label} CPU also failed:\n{r2.stderr[-500:]}")
    return True  # used CPU


def _ffmpeg_cut(source, start, end, output, vf=None):
    """
    Cut a clip from source video.
    CPU FIRST (-ss before -i for fast seek) -> GPU fallback.
    """
    dur = round(end - start, 3)

    # CPU command (fast seek with -ss before -i)
    cpu_cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", source,
        "-t", f"{dur:.3f}",
    ]
    if vf:
        cpu_cmd += ["-vf", vf]
    cpu_cmd += ["-c:v", "libx264", "-c:a", "aac", "-b:a", "192k"] + _CPU_FLAGS + [output]

    # GPU fallback command
    gpu_cmd = None
    if _NVENC_OK:
        gpu_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", source,
            "-t", f"{dur:.3f}",
        ]
        if vf:
            gpu_cmd += ["-vf", vf]
        gpu_cmd += ["-c:v", ENCODER, "-c:a", "aac", "-b:a", "192k"] + ENCODER_FLAGS + [output]

    _run_cpu_first(cpu_cmd, gpu_cmd, f"cut {start:.1f}-{end:.1f}s")
    return output


def _ffmpeg_vertical(inp, out, vf):
    """
    Convert to vertical format. GPU first for re-encode, CPU fallback.
    """
    gpu_cmd = None
    if _NVENC_OK:
        gpu_cmd = [
            "ffmpeg", "-y", "-i", inp, "-vf", vf,
            "-c:v", ENCODER, "-c:a", "aac", "-r", "30",
        ] + ENCODER_FLAGS + [out]

    cpu_cmd = [
        "ffmpeg", "-y", "-i", inp, "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-r", "30", out,
    ]
    _run_gpu_first(gpu_cmd, cpu_cmd, "vertical convert")


def generate_clips(video_path, segments, output_dir, padding=0.5):
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, seg in enumerate(segments):
        start = max(0.0, seg["start_time"] - padding)
        end   = seg["end_time"] + padding
        out   = os.path.join(output_dir, f"clip_{i+1:03d}.mp4")
        print(f"[Clip] {i+1}/{len(segments)}: {start:.1f}s->{end:.1f}s [CPU libx264 first]")
        _ffmpeg_cut(video_path, start, end, out)
        clip_paths.append(out)
    return clip_paths


def convert_to_vertical(clip_path, output_path, width=1080, height=1920,
                         crop_x=None, crop_y=None):
    if crop_x is None:
        vf = f"scale=-1:{height},crop={width}:{height}"
    else:
        cy = crop_y or 0
        vf = f"scale=-1:{height},crop={width}:{height}:{crop_x}:{cy}"
    _ffmpeg_vertical(clip_path, output_path, vf)
    return output_path
