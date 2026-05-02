"""
transcription.py  (v1)

Three transcription backends:
  1. openai-whisper   (original, GPU fp32)
  2. faster-whisper   (CTranslate2 - 4x faster, VAD filter, better Hindi)
  3. whisperx         (WhisperX full transcription - word-level aligned)

ALL backends:
  - Auto-detect language from FIRST 30 SECONDS ONLY (fast probe)
  - Transcribe in detected language (Hindi stays Hindi)
  - Return identical dict: { text, segments, language, words }
  - fp16/float16 disabled for Hindi/Arabic/CJK (prevents NaN / empty)
  - condition_on_previous_text=False (prevents hallucination loops)
  - beam_size=10, best_of=5 for accuracy
  - WAV PCM 16kHz mono input (from audio_extractor.py)
"""

import json, os, warnings
import torch

_FP16_UNSAFE_LANGS = {
    "hi", "ar", "zh", "ja", "ko", "ru", "ur", "ta", "te",
    "th", "vi", "fa", "he", "el", "tr", "bn", "mr", "gu",
    "kn", "ml", "pa", "ne", "si", "my", "km", "lo",
}

_INITIAL_PROMPTS = {
    "hi": "Transcribe this Hindi or Hinglish audio accurately. Include all spoken words.",
    "ur": "Transcribe this Urdu audio accurately.",
    "ar": "Transcribe this Arabic audio accurately.",
    "zh": "Transcribe this Chinese Mandarin audio accurately.",
    "ja": "Transcribe this Japanese audio accurately.",
    "ko": "Transcribe this Korean audio accurately.",
    "ru": "Transcribe this Russian audio accurately.",
    "ta": "Transcribe this Tamil audio accurately.",
    "te": "Transcribe this Telugu audio accurately.",
    "bn": "Transcribe this Bengali audio accurately.",
    "mr": "Transcribe this Marathi audio accurately.",
    "gu": "Transcribe this Gujarati audio accurately.",
    "pa": "Transcribe this Punjabi audio accurately.",
}


def _get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[Whisper] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Whisper] No GPU - using CPU")
    return device


def _extract_30s_probe(audio_path):
    """
    Extract first 30 seconds to a temp WAV for language detection.
    Returns path to temp file (caller must delete).
    """
    import subprocess, tempfile
    tmp = tempfile.mktemp(suffix="_probe.wav")
    r = subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-t", "30",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp,
    ], capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(tmp):
        return tmp
    return audio_path  # fallback: use full file


# ─── Backend 1: OpenAI Whisper ────────────────────────────────────────────────
def _transcribe_openai(audio_path, model_name="medium", word_timestamps=True):
    import whisper

    device   = _get_device()
    use_fp16 = (device == "cuda")

    print(f"[OpenAI-Whisper] Loading model='{model_name}' | fp16={use_fp16}")
    model = whisper.load_model(model_name, device=device)

    # Language probe on first 30s only
    probe_path = _extract_30s_probe(audio_path)
    print("[OpenAI-Whisper] Detecting language from first 30s ...")
    try:
        audio_array = whisper.load_audio(probe_path)
        audio_clip  = whisper.pad_or_trim(audio_array)
        mel         = whisper.log_mel_spectrogram(audio_clip).to(model.device)
        if use_fp16:
            mel = mel.half()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence    = probs[detected_lang]
        print(f"[OpenAI-Whisper] Detected: '{detected_lang}' (conf={confidence:.2f})")
    finally:
        if probe_path != audio_path and os.path.exists(probe_path):
            os.remove(probe_path)

    if detected_lang in _FP16_UNSAFE_LANGS:
        use_fp16 = False
        print(f"[OpenAI-Whisper] fp16 DISABLED for '{detected_lang}' -> fp32")

    kwargs = dict(
        language                    = detected_lang,
        word_timestamps             = word_timestamps,
        verbose                     = True,
        fp16                        = use_fp16,
        task                        = "transcribe",
        condition_on_previous_text  = False,
        beam_size                   = 10,
        best_of                     = 5,
        temperature                 = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold = 2.4,
        logprob_threshold           = -1.0,
        no_speech_threshold         = 0.5,
    )
    if detected_lang in _INITIAL_PROMPTS:
        kwargs["initial_prompt"] = _INITIAL_PROMPTS[detected_lang]
        print(f"[OpenAI-Whisper] Using seed prompt for '{detected_lang}'")

    print(f"[OpenAI-Whisper] Transcribing lang='{detected_lang}' fp16={use_fp16} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.transcribe(audio_path, **kwargs)

    result["language"] = detected_lang
    return result


# ─── Backend 2: Faster-Whisper (CTranslate2) ─────────────────────────────────
def _transcribe_faster(audio_path, model_name="medium", word_timestamps=True):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed.\n"
            "Run: pip install faster-whisper"
        )

    device  = _get_device()
    compute = "float16" if device == "cuda" else "int8"

    print(f"[Faster-Whisper] Loading model='{model_name}' | device={device} | compute={compute}")
    model = WhisperModel(model_name, device=device, compute_type=compute,
                         num_workers=4, cpu_threads=8)

    # Language probe: first 30s only
    probe_path = _extract_30s_probe(audio_path)
    print("[Faster-Whisper] Detecting language from first 30s ...")
    try:
        segments_gen, info = model.transcribe(
            probe_path,
            beam_size    = 5,
            language     = None,
            task         = "transcribe",
            vad_filter   = True,
            vad_parameters = dict(min_silence_duration_ms=500),
        )
        for _ in segments_gen:
            pass
        detected_lang = info.language
        confidence    = info.language_probability
        print(f"[Faster-Whisper] Detected: '{detected_lang}' (conf={confidence:.2f})")
    finally:
        if probe_path != audio_path and os.path.exists(probe_path):
            os.remove(probe_path)

    if detected_lang in _FP16_UNSAFE_LANGS and compute == "float16":
        compute = "float32"
        print(f"[Faster-Whisper] Switching to float32 for '{detected_lang}'")
        model = WhisperModel(model_name, device=device, compute_type=compute,
                             num_workers=4, cpu_threads=8)

    prompt = _INITIAL_PROMPTS.get(detected_lang, None)
    if prompt:
        print(f"[Faster-Whisper] Using seed prompt for '{detected_lang}'")

    print(f"[Faster-Whisper] Transcribing lang='{detected_lang}' compute={compute} ...")
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size              = 10,
        best_of                = 5,
        language               = detected_lang,
        task                   = "transcribe",
        initial_prompt         = prompt,
        word_timestamps        = word_timestamps,
        condition_on_previous_text = False,
        compression_ratio_threshold = 2.4,
        log_prob_threshold     = -1.0,
        no_speech_threshold    = 0.5,
        vad_filter             = True,
        vad_parameters         = dict(min_silence_duration_ms=300),
        temperature            = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    segments_list = []
    full_text     = []

    for seg in segments_gen:
        words_list = []
        if word_timestamps and seg.words:
            for w in seg.words:
                words_list.append({
                    "word":        w.word.strip(),
                    "start":       w.start,
                    "end":         w.end,
                    "probability": w.probability,
                })
        segments_list.append({
            "id":                seg.id,
            "start":             seg.start,
            "end":               seg.end,
            "text":              seg.text.strip(),
            "words":             words_list,
            "avg_logprob":       seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob":    seg.no_speech_prob,
        })
        full_text.append(seg.text.strip())

    result = {
        "text":     " ".join(full_text),
        "segments": segments_list,
        "language": detected_lang,
    }
    return result


# ─── Backend 3: WhisperX full transcription ───────────────────────────────────
def _transcribe_whisperx(audio_path, model_name="medium", word_timestamps=True):
    """
    Full WhisperX pipeline:
      1. Transcribe with WhisperX (uses faster-whisper under the hood)
      2. Forced word-level alignment with wav2vec2
      3. Returns same format as other backends
    """
    try:
        import whisperx
    except ImportError:
        raise RuntimeError(
            "whisperx not installed.\n"
            "Run: pip install whisperx"
        )

    device  = _get_device()
    compute = "float16" if device == "cuda" else "int8"

    # Language probe: first 30s
    probe_path = _extract_30s_probe(audio_path)
    print(f"[WhisperX] Detecting language from first 30s ...")
    try:
        probe_model = whisperx.load_model(
            model_name, device=device,
            compute_type=compute,
            language=None,
        )
        probe_audio = whisperx.load_audio(probe_path)
        probe_result = probe_model.transcribe(probe_audio, batch_size=4)
        detected_lang = probe_result.get("language", "en")
        print(f"[WhisperX] Detected: '{detected_lang}'")
    finally:
        if probe_path != audio_path and os.path.exists(probe_path):
            os.remove(probe_path)

    if detected_lang in _FP16_UNSAFE_LANGS and compute == "float16":
        compute = "int8"
        print(f"[WhisperX] Switching to int8 for '{detected_lang}'")

    print(f"[WhisperX] Loading model='{model_name}' | device={device} | compute={compute}")
    model = whisperx.load_model(
        model_name, device=device,
        compute_type=compute,
        language=detected_lang,
    )

    print(f"[WhisperX] Transcribing ...")
    audio   = whisperx.load_audio(audio_path)
    result  = model.transcribe(audio, batch_size=8)
    result["language"] = detected_lang

    # Word-level alignment
    if word_timestamps:
        try:
            print(f"[WhisperX] Aligning words (wav2vec2) ...")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device
            )
            aligned = whisperx.align(
                result["segments"], model_a, metadata,
                audio, device, return_char_alignments=False,
            )
            # Merge aligned words into segments
            for seg, aligned_seg in zip(result["segments"], aligned.get("segments", [])):
                seg["words"] = aligned_seg.get("words", [])
        except Exception as e:
            print(f"[WhisperX] Alignment failed ({e}) - using segment timestamps")

    # Normalise segment format to match other backends
    normalized = []
    for i, seg in enumerate(result.get("segments", [])):
        words_list = []
        for w in seg.get("words", []):
            words_list.append({
                "word":        w.get("word", "").strip(),
                "start":       w.get("start", seg["start"]),
                "end":         w.get("end",   seg["end"]),
                "probability": w.get("score", 1.0),
            })
        normalized.append({
            "id":    i,
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"].strip(),
            "words": words_list,
            "avg_logprob":       seg.get("avg_logprob", 0.0),
            "compression_ratio": seg.get("compression_ratio", 1.0),
            "no_speech_prob":    seg.get("no_speech_prob", 0.0),
        })

    result["segments"] = normalized
    return result


# ─── Public API ───────────────────────────────────────────────────────────────
def transcribe_audio(
    audio_path,
    output_json_path,
    model_name      = "medium",
    language        = None,
    word_timestamps = True,
    engine          = "faster",
):
    """
    Transcribe audio with auto language detection (first 30s probe).

    engine="openai"   -> openai-whisper
    engine="faster"   -> faster-whisper (4x faster, better Hindi VAD)
    engine="whisperx" -> WhisperX (word-level aligned transcription)

    Returns dict: { text, segments, language }
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[Whisper] Audio not found: {audio_path}")

    print(f"[Whisper] Engine='{engine}' | model='{model_name}' | audio={audio_path}")

    if engine == "whisperx":
        result = _transcribe_whisperx(audio_path, model_name, word_timestamps)
    elif engine == "faster":
        result = _transcribe_faster(audio_path, model_name, word_timestamps)
    else:
        result = _transcribe_openai(audio_path, model_name, word_timestamps)

    segs = result.get("segments", [])
    text = result.get("text", "").strip()
    det  = result.get("language", "unknown")
    print(f"[Whisper] Result: {len(segs)} segments | {len(text)} chars | lang={det}")

    if len(segs) == 0 or not text:
        raise RuntimeError(
            f"[Whisper] EMPTY transcript (lang='{det}', engine='{engine}').\n"
            f"  1) Try model='medium' or 'large'\n"
            f"  2) Check audio: ffplay {audio_path}\n"
            f"  3) Ensure clear speech in video\n"
            f"  4) If Hindi: try 'faster' or 'whisperx' engine"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"[Whisper] Saved -> {output_json_path}")
    return result


def load_transcript(json_path):
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def get_words_in_range(transcript, start, end):
    results = []
    for seg in transcript.get("segments", []):
        for wi in seg.get("words", []):
            ws = wi.get("start", 0)
            we = wi.get("end",   0)
            if ws >= start - 0.05 and we <= end + 0.05:
                results.append({
                    "word":        wi.get("word", "").strip(),
                    "start":       ws,
                    "end":         we,
                    "probability": wi.get("probability", 1.0),
                })
    return results


def get_segments_in_range(transcript, start, end):
    results = []
    for seg in transcript.get("segments", []):
        ss = seg.get("start", 0)
        se = seg.get("end",   0)
        if se >= start and ss <= end:
            results.append(seg)
    return results
