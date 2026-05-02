"""
face_tracker.py  (v12)

Face detection priority (GPU first where available):
  1. OpenCV DNN Face Detector   (res10_300x300_ssd - CUDA GPU backend)
  2. MediaPipe BlazeFace        (CPU / TFLite)
  3. OpenCV Haar Cascade        (CPU, always available)

Face count logic:
  - 0 faces : hold last known position (center fallback)
  - 1 face  : standard single-face tracking (center crop)
  - 2 faces : SPLIT SCREEN (top half = face1, bottom half = face2)
  - 3+ faces: pick main speaker (largest + most centered face)

Crop / Resize / Encode pipeline (unchanged from v9):
  - OpenCV VideoWriter for raw frames (CPU)
  - Re-encode with FFmpeg NVENC (GPU) or libx264 (CPU) fallback
  - Audio merged with stream-copy
"""

import os, cv2, numpy as np, subprocess, shutil, collections, urllib.request
from pathlib import Path

# ─── Backend 1: OpenCV DNN Face Detector (GPU first) ─────────────────────────
# Model: res10_300x300_ssd_iter_140000.caffemodel
# Download from OpenCV GitHub releases / opencv_extra
_DNN_AVAILABLE  = False
_dnn_net        = None
_DNN_PROTO_URL  = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_MODEL_URL  = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_DNN_DIR        = os.path.dirname(os.path.abspath(__file__))
_DNN_PROTO      = os.path.join(_DNN_DIR, "deploy.prototxt")
_DNN_MODEL      = os.path.join(_DNN_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


def _try_load_dnn():
    global _DNN_AVAILABLE, _dnn_net
    try:
        # Download model files if missing
        if not os.path.exists(_DNN_PROTO):
            print("[FaceTracker] Downloading DNN prototxt ...")
            urllib.request.urlretrieve(_DNN_PROTO_URL, _DNN_PROTO)
        if not os.path.exists(_DNN_MODEL):
            print("[FaceTracker] Downloading DNN caffemodel (~2MB) ...")
            urllib.request.urlretrieve(_DNN_MODEL_URL, _DNN_MODEL)

        net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)

        # Try CUDA GPU backend first
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Quick test to verify CUDA works
            test_blob = cv2.dnn.blobFromImage(
                np.zeros((300, 300, 3), dtype=np.uint8), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False,
            )
            net.setInput(test_blob)
            net.forward()
            _dnn_net = net
            _DNN_AVAILABLE = True
            print("[FaceTracker] OpenCV DNN loaded on CUDA GPU.")
            return
        except Exception as gpu_err:
            print(f"[FaceTracker] DNN CUDA failed ({gpu_err}) -> CPU backend")

        # CPU fallback
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _dnn_net = net
        _DNN_AVAILABLE = True
        print("[FaceTracker] OpenCV DNN loaded on CPU backend.")

    except Exception as e:
        print(f"[FaceTracker] DNN unavailable: {e}")

_try_load_dnn()


# ─── Backend 2: MediaPipe BlazeFace (CPU) ────────────────────────────────────
_MP_AVAILABLE = False
_mp_detector  = None

if not _DNN_AVAILABLE:
    try:
        import mediapipe as mp
        from mediapipe.tasks        import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        _MP_MODEL_PATH = os.path.join(_DNN_DIR, "blaze_face_short_range.tflite")
        _MP_MODEL_URL  = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite"
        )
        if not os.path.exists(_MP_MODEL_PATH):
            print("[FaceTracker] Downloading MediaPipe BlazeFace model...")
            urllib.request.urlretrieve(_MP_MODEL_URL, _MP_MODEL_PATH)

        if os.path.exists(_MP_MODEL_PATH):
            _opts = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_MP_MODEL_PATH),
                min_detection_confidence=0.45,
            )
            _mp_detector  = mp_vision.FaceDetector.create_from_options(_opts)
            _MP_AVAILABLE = True
            print("[FaceTracker] MediaPipe BlazeFace loaded (CPU).")
    except Exception as e:
        print(f"[FaceTracker] MediaPipe unavailable: {e}")


# ─── Backend 3: Haar Cascade (CPU, always available) ─────────────────────────
_HAAR = None
try:
    _haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(_haar_path):
        _HAAR = cv2.CascadeClassifier(_haar_path)
        print("[FaceTracker] Haar cascade ready (CPU last-resort fallback).")
except Exception:
    pass


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_W     = 1080
TARGET_H     = 1920
RATIO        = TARGET_W / TARGET_H

EMA_SLOW     = 0.10   # stable tracking
EMA_FAST     = 0.35   # fast on jump
JUMP_THRESH  = 0.08

VOTE_WIN     = 7
DETECT_EVERY = 2
MIN_FACE_H   = 20
FACE_Y_BIAS  = 0.28   # keep face at top 28% of output frame


# ─── Detection backends ───────────────────────────────────────────────────────
def _dnn_detect_all(frame):
    """OpenCV DNN SSD detector. Returns [(cx, cy, conf, area), ...]"""
    if not _DNN_AVAILABLE or _dnn_net is None:
        return []
    try:
        h, w  = frame.shape[:2]
        blob  = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        _dnn_net.setInput(blob)
        detections = _dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.50:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bw = x2 - x1
            bh = y2 - y1
            if bh < MIN_FACE_H:
                continue
            cx   = float(np.clip((x1 + bw / 2) / w, 0, 1))
            cy   = float(np.clip((y1 + bh / 2) / h, 0, 1))
            area = float((bw * bh) / (w * h))
            faces.append((cx, cy, conf, area))
        return faces
    except Exception as e:
        print(f"[FaceTracker] DNN error: {e}")
        return []


def _mp_detect_all(frame):
    """MediaPipe BlazeFace. Returns [(cx, cy, conf, area), ...]"""
    if not _MP_AVAILABLE or _mp_detector is None:
        return []
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = _mp_detector.detect(img)
        if not res.detections:
            return []
        h, w = frame.shape[:2]
        faces = []
        for det in res.detections:
            bb   = det.bounding_box
            if bb.height < MIN_FACE_H:
                continue
            cx   = float(np.clip((bb.origin_x + bb.width  / 2) / w, 0, 1))
            cy   = float(np.clip((bb.origin_y + bb.height / 2) / h, 0, 1))
            conf = det.categories[0].score if det.categories else 0.7
            area = float((bb.width * bb.height) / (w * h))
            faces.append((cx, cy, float(conf), area))
        return faces
    except Exception as e:
        print(f"[FaceTracker] MP error: {e}")
        return []


def _haar_detect_all(frame):
    """Haar cascade. Returns [(cx, cy, conf, area), ...]"""
    if _HAAR is None:
        return []
    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_H, MIN_FACE_H))
        if not len(faces):
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = _HAAR.detectMultiScale(small, 1.1, 4, minSize=(16, 16))
            if not len(faces):
                return []
            faces = (faces * 2).astype(int)
        h_, w_ = frame.shape[:2]
        result = []
        for (x, y, fw, fh) in faces:
            cx   = float(np.clip((x + fw / 2) / w_, 0, 1))
            cy   = float(np.clip((y + fh / 2) / h_, 0, 1))
            area = float((fw * fh) / (w_ * h_))
            result.append((cx, cy, 0.5, area))
        return result
    except Exception:
        return []


def _detect_all_backends(frame):
    """DNN (GPU/CPU) -> MediaPipe -> Haar. Returns (faces, backend_name)."""
    faces = _dnn_detect_all(frame)
    if faces:
        return faces, "dnn"
    faces = _mp_detect_all(frame)
    if faces:
        return faces, "mediapipe"
    faces = _haar_detect_all(frame)
    return faces, "haar"


def _pick_speaking_face(faces):
    """
    Pick the main on-camera speaker from multiple faces.
    Score = 55% area + 25% y-position + 20% center-x proximity.
    Returns (cx, cy, conf) of best face.
    """
    if not faces:
        return None, None, 0.0
    if len(faces) == 1:
        cx, cy, conf, _ = faces[0]
        return cx, cy, conf

    max_area = max(f[3] for f in faces) or 1e-6
    best_score, best = -1.0, None

    for (cx, cy, conf, area) in faces:
        area_score   = area / max_area
        y_score      = cy
        center_score = 1.0 - abs(cx - 0.5) * 2
        score = 0.55 * area_score + 0.25 * y_score + 0.20 * center_score
        if score > best_score:
            best_score = score
            best = (cx, cy, conf)

    return best if best else (None, None, 0.0)


def _sort_faces_by_area(faces):
    """Sort faces by area descending. Returns sorted list."""
    return sorted(faces, key=lambda f: f[3], reverse=True)


# ─── Frame composers ──────────────────────────────────────────────────────────
def _crop_frame_single(frame, smooth_cx, smooth_cy, src_w, src_h, crop_w, crop_h):
    """Crop frame to portrait around a single face position."""
    cx_px   = int(smooth_cx * src_w)
    x_start = int(np.clip(cx_px - crop_w // 2, 0, src_w - crop_w))

    face_px  = int(smooth_cy * src_h)
    scale_h  = TARGET_H / crop_h
    want_src = int((TARGET_H * FACE_Y_BIAS) / scale_h)
    y_offset = int(np.clip(face_px - want_src, 0, src_h - crop_h))

    cropped = frame[y_offset:y_offset + crop_h, x_start:x_start + crop_w]
    return cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)


def _compose_split_screen(frame, face1, face2, src_w, src_h, crop_w, crop_h):
    """
    Split-screen for exactly 2 faces.
    Top half (0..TARGET_H//2)  -> face1 (larger / more prominent)
    Bottom half (TARGET_H//2..) -> face2
    Each half is a portrait crop of the original frame centered on the face.
    """
    half_h = TARGET_H // 2
    output = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    def _crop_half(face_cx, face_cy):
        cx_px   = int(face_cx * src_w)
        x_start = int(np.clip(cx_px - crop_w // 2, 0, src_w - crop_w))
        # For split screen use top 50% of source height per face
        sub_h   = min(src_h // 2, crop_h)
        cy_px   = int(face_cy * src_h)
        y_start = int(np.clip(cy_px - sub_h // 2, 0, src_h - sub_h))
        cropped = frame[y_start:y_start + sub_h, x_start:x_start + crop_w]
        return cv2.resize(cropped, (TARGET_W, half_h), interpolation=cv2.INTER_LINEAR)

    cx1, cy1, _, _ = face1
    cx2, cy2, _, _ = face2

    output[0:half_h,  :] = _crop_half(cx1, cy1)
    output[half_h:TARGET_H, :] = _crop_half(cx2, cy2)

    # Divider line
    cv2.line(output, (0, half_h), (TARGET_W, half_h), (30, 30, 30), 3)
    return output


# ─── Main face tracker class ──────────────────────────────────────────────────
class FaceTracker:
    def __init__(self):
        self.smooth_cx  = 0.5
        self.smooth_cy  = 0.40
        self.last_cx    = None
        self.last_cy    = None
        self.vote_buf   = collections.deque(maxlen=VOTE_WIN)
        self.no_detect  = 0
        self.backend    = "none"
        self.last_faces = []

    def _weighted_vote(self):
        if not self.vote_buf:
            return None, None
        tot  = sum(v[2] for v in self.vote_buf) or 1e-6
        vcx  = sum(v[0] * v[2] for v in self.vote_buf) / tot
        vcy  = sum(v[1] * v[2] for v in self.vote_buf) / tot
        return vcx, vcy

    def _ema(self, smooth, target):
        delta = abs(target - smooth)
        alpha = EMA_FAST if delta > JUMP_THRESH else EMA_SLOW
        return smooth + alpha * (target - smooth)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {input_path}")

        src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        crop_w = int(src_h * RATIO)
        crop_h = src_h

        if crop_w >= src_w:
            cap.release()
            return _pad_to_vertical(input_path, output_path)

        tmp_raw = output_path + ".raw.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(tmp_raw, fourcc, fps, (TARGET_W, TARGET_H))

        self.smooth_cx = 0.5
        self.smooth_cy = 0.40
        self.vote_buf.clear()
        self.no_detect  = 0
        self.backend    = "none"
        self.last_faces = []

        backend_counts = {"dnn": 0, "mediapipe": 0, "haar": 0, "none": 0}
        frame_idx = 0

        _dnn_mode = "GPU" if (_DNN_AVAILABLE and _dnn_net is not None) else "N/A"
        print(f"[FaceTracker] {total} frames @ {fps:.1f}fps | {src_w}x{src_h} -> {TARGET_W}x{TARGET_H}")
        print(f"[FaceTracker] Backends: DNN({_dnn_mode}) | "
              f"MediaPipe({'ON' if _MP_AVAILABLE else 'OFF'}) | "
              f"Haar({'ON' if _HAAR is not None else 'OFF'})")
        print(f"[FaceTracker] Face logic: 1->single | 2->split-screen | 3+->pick main")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Detect every Nth frame ──────────────────────────────────────
            if frame_idx % DETECT_EVERY == 0:
                faces, backend = _detect_all_backends(frame)
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
                self.backend = backend
                self.last_faces = faces

                if faces:
                    cx, cy, conf = _pick_speaking_face(faces)
                    if cx is not None:
                        self.vote_buf.append((cx, cy, conf))
                        self.last_cx   = cx
                        self.last_cy   = cy
                        self.no_detect = 0
                else:
                    self.no_detect += 1
                    backend_counts["none"] = backend_counts.get("none", 0) + 1

            # ── Smooth position ─────────────────────────────────────────────
            vcx, vcy = self._weighted_vote()
            target_cx = vcx if vcx is not None else (self.last_cx or 0.5)
            target_cy = vcy if vcy is not None else (self.last_cy or 0.40)
            self.smooth_cx = self._ema(self.smooth_cx, target_cx)
            self.smooth_cy = self._ema(self.smooth_cy, target_cy)

            # ── Compose output frame ────────────────────────────────────────
            n_faces = len(self.last_faces)

            if n_faces == 2:
                # Split screen: top = bigger face, bottom = smaller face
                sorted_faces = _sort_faces_by_area(self.last_faces)
                out_frame = _compose_split_screen(
                    frame, sorted_faces[0], sorted_faces[1],
                    src_w, src_h, crop_w, crop_h,
                )
            else:
                # 0, 1, or 3+ faces: single crop centered on speaking face
                out_frame = _crop_frame_single(
                    frame,
                    self.smooth_cx, self.smooth_cy,
                    src_w, src_h, crop_w, crop_h,
                )

            writer.write(out_frame)

            if frame_idx % 120 == 0:
                pct    = int(frame_idx / total * 100) if total > 0 else 0
                status = f"faces={n_faces} cx={self.smooth_cx:.2f} cy={self.smooth_cy:.2f} [{self.backend}]"
                print(f"[FaceTracker]  {frame_idx}/{total} ({pct}%)  {status}")

        cap.release()
        writer.release()

        det_total = sum(backend_counts.values()) or 1
        print(f"[FaceTracker] Summary: "
              f"DNN={backend_counts['dnn']} | "
              f"MP={backend_counts['mediapipe']} | "
              f"Haar={backend_counts['haar']} | "
              f"none={backend_counts['none']}")

        tmp_enc = output_path + ".enc.mp4"
        _reencode_nvenc(tmp_raw, tmp_enc, fps)
        _merge_audio(input_path, tmp_enc, output_path)

        for f in [tmp_raw, tmp_enc]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

        return output_path


def _reencode_nvenc(src, dst, fps):
    """Re-encode mp4v raw frames to h264 via NVENC (GPU) or libx264 (CPU)."""
    from clip_generator import ENCODER, ENCODER_FLAGS

    gpu_cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", ENCODER, "-pix_fmt", "yuv420p",
    ] + ENCODER_FLAGS + [dst]

    r = subprocess.run(gpu_cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[FaceTracker] Re-encoded with {ENCODER}")
        return

    cpu_cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", dst,
    ]
    subprocess.run(cpu_cmd, capture_output=True, check=True)
    print("[FaceTracker] Re-encoded with libx264 (CPU)")


def _pad_to_vertical(input_path, output_path):
    """Pad portrait/square video to 9:16 with black bars."""
    from clip_generator import ENCODER, ENCODER_FLAGS
    vf = (
        f"scale=iw*min({TARGET_W}/iw\\,{TARGET_H}/ih):"
        f"ih*min({TARGET_W}/iw\\,{TARGET_H}/ih),"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black"
    )
    gpu = ["ffmpeg", "-y", "-i", input_path, "-vf", vf,
           "-c:v", ENCODER, "-c:a", "aac"] + ENCODER_FLAGS + [output_path]
    r = subprocess.run(gpu, capture_output=True, text=True)
    if r.returncode == 0:
        return output_path
    cpu = ["ffmpeg", "-y", "-i", input_path, "-vf", vf,
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-c:a", "aac", output_path]
    subprocess.run(cpu, capture_output=True, check=True)
    return output_path


def _merge_audio(original, video_only, output):
    """Merge audio from original clip into processed video."""
    from clip_generator import ENCODER, ENCODER_FLAGS

    # Attempt 1: stream copy (fastest)
    sc = [
        "ffmpeg", "-y",
        "-i", video_only, "-i", original,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", output,
    ]
    r = subprocess.run(sc, capture_output=True, text=True)
    if r.returncode == 0:
        print("[FaceTracker] Audio merged (stream copy)")
        return

    # Attempt 2: GPU re-encode + audio
    gpu = [
        "ffmpeg", "-y",
        "-i", video_only, "-i", original,
        "-c:v", ENCODER, "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    ] + ENCODER_FLAGS + [output]
    r2 = subprocess.run(gpu, capture_output=True, text=True)
    if r2.returncode == 0:
        print(f"[FaceTracker] Audio merged (GPU {ENCODER})")
        return

    # Attempt 3: CPU encode + audio
    cpu = [
        "ffmpeg", "-y",
        "-i", video_only, "-i", original,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", output,
    ]
    r3 = subprocess.run(cpu, capture_output=True, text=True)
    if r3.returncode == 0:
        print("[FaceTracker] Audio merged (CPU libx264)")
        return

    print("[FaceTracker] All merge attempts failed - copying video only")
    shutil.copy(video_only, output)


def apply_face_tracking(clip_paths, output_dir, progress_queue=None):
    """
    Apply face tracking to all clips.
    DNN GPU -> MediaPipe CPU -> Haar CPU cascade.
    2-face clips get split-screen. 3+ get main speaker crop.
    """
    out_paths = []

    for i, clip in enumerate(clip_paths):
        stem   = Path(clip).stem
        output = os.path.join(output_dir, f"{stem}_vertical.mp4")
        print(f"\n[FaceTracker] Clip {i+1}/{len(clip_paths)}: {Path(clip).name}")

        if progress_queue:
            pct = 66 + int((i / len(clip_paths)) * 14)
            progress_queue.put({
                "type":  "progress",
                "pct":   pct,
                "stage": f"Face tracking {i+1}/{len(clip_paths)} ...",
            })

        try:
            tracker = FaceTracker()
            tracker.process_video(clip, output)
            size_mb = os.path.getsize(output) / (1024 * 1024) if os.path.exists(output) else 0
            print(f"[FaceTracker] Done -> {output} ({size_mb:.1f} MB)")
            out_paths.append(output)
        except Exception as e:
            import traceback
            print(f"[FaceTracker] Error on {Path(clip).name}: {e}")
            traceback.print_exc()
            out_paths.append(clip)

    return out_paths
