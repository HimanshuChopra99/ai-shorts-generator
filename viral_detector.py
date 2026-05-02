"""
viral_detector.py  (v1)

OpenRouter: 4 free model fallback, 429/402/503 error handling, retry.
Local NLP fallback when no API key or all models fail.
"""

import re, json, os, time
try:
    import requests as _req
    _REQ_OK = True
except ImportError:
    _REQ_OK = False

HOOK_WORDS = [
    "secret","never","always","everyone","nobody","shocking","unbelievable",
    "insane","crazy","incredible","mindblowing","you won't believe","wait for it",
    "plot twist","breaking","exclusive","exposed","truth","revealed","warning",
    "game changer","this changed everything","no one talks about",
]
EMOTIONAL_WORDS = [
    "love","hate","fear","amazing","horrible","beautiful","devastating",
    "inspiring","heartbreaking","joyful","furious","terrified","excited",
    "disgusted","proud","ashamed","miracle","nightmare","dream","tragedy","victory",
]
ACTION_WORDS = [
    "step","tip","trick","hack","method","strategy","technique",
    "how to","do this","try this","remember","important","key",
    "must","should","need to","have to","essential","critical",
    "simple","easy","fast","instantly","immediately",
]
STORY_WORDS = [
    "suddenly","then","after that","next","finally","eventually",
    "realised","realized","discovered","found out","turned out",
    "happened","moment","changed","transformed","journey",
]

_Q_PAT   = re.compile(r"\?")
_NUM_PAT = re.compile(r"\b(\d+[%$]?|\$\d+|\d+x|\d+k|\d+M)\b")
_CAP_PAT = re.compile(r"\b[A-Z]{3,}\b")


def _local_score(text, duration):
    lower   = text.lower()
    reasons = []
    score   = 0

    hits = [w for w in HOOK_WORDS if w in lower]
    if hits:
        score += min(25, len(hits) * 8)
        reasons.append(f"Hook: {', '.join(hits[:3])}")

    hits = [w for w in EMOTIONAL_WORDS if w in lower]
    if hits:
        score += min(20, len(hits) * 6)
        reasons.append(f"Emotional: {', '.join(hits[:3])}")

    hits = [w for w in ACTION_WORDS if w in lower]
    if hits:
        score += min(18, len(hits) * 5)
        reasons.append(f"Actionable: {', '.join(hits[:3])}")

    hits = [w for w in STORY_WORDS if w in lower]
    if hits:
        score += min(12, len(hits) * 4)
        reasons.append("Storytelling")

    q = len(_Q_PAT.findall(text))
    if q:
        score += min(15, q * 7)
        reasons.append("Engaging question")

    nums = _NUM_PAT.findall(text)
    if nums:
        score += min(10, len(nums) * 4)
        reasons.append(f"Numbers: {', '.join(nums[:3])}")

    caps = _CAP_PAT.findall(text)
    if caps:
        score += min(8, len(caps) * 4)
        reasons.append("High energy")

    if 20 <= duration <= 60:
        score += 10
    elif duration > 60:
        score -= int((duration - 60) / 5)

    words = len(text.split())
    if words < 15:
        score -= 15
    elif words > 30:
        score += 5

    return max(0, min(100, score)), reasons


def _merge_segments(segments, min_dur=20.0, max_dur=60.0):
    merged  = []
    current = None
    for seg in segments:
        if current is None:
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            if seg["end"] - current["start"] <= max_dur:
                current["end"]   = seg["end"]
                current["text"] += " " + seg["text"]
            else:
                if current["end"] - current["start"] >= min_dur:
                    merged.append(current)
                current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    if current and current["end"] - current["start"] >= min_dur:
        merged.append(current)
    return merged


def _extract_json_from_response(content):
    content = content.strip()
    bt3 = chr(96) * 3
    if bt3 in content:
        parts = content.split(bt3)
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                content = part
                break

    start_idx = content.find("[")
    end_idx   = content.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        content = content[start_idx:end_idx+1]

    return json.loads(content)


_OR_MODELS = [
    "openai/gpt-4o:free",
    "openai/gpt-4o-mini:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


def _ai_detect(transcript_text, segments_summary, num_shorts, key):
    if not _REQ_OK:
        print("[ViralDetector] requests not installed - using local NLP")
        return None

    lines = []
    for s in segments_summary[:80]:
        lines.append(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] ({s['end']-s['start']:.0f}s): "
            f"{s['text'][:150]}"
        )
    segs_text = "\n".join(lines)
    n_req     = num_shorts if num_shorts else "the best 5-8"

    prompt = (
        f"You are a viral short-form video expert (YouTube Shorts, TikTok, Instagram Reels). "
        f"Analyze this transcript and identify {n_req} most viral-worthy segments.\n\n"
        f"SEGMENTS:\n{segs_text}\n\n"
        f"Return a JSON array ONLY (no other text, no markdown):\n"
        f'[{{"start_time":12.5,"end_time":45.2,"viral_score":87,'
        f'"reason":"Strong hook - surprising reveal","description":"Speaker reveals unexpected fact"}}]'
    )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/ai-shorts-generator",
        "X-Title":       "AI Shorts Generator",
    }

    for model in _OR_MODELS:
        print(f"[ViralDetector] Trying model: {model}")
        for attempt in range(2):
            try:
                resp = _req.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model":       model,
                        "messages":    [{"role": "user", "content": prompt}],
                        "max_tokens":  2048,
                        "temperature": 0.3,
                    },
                    timeout=90,
                )

                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"[ViralDetector] Rate limited (429) - waiting {wait}s ...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 402:
                    print(f"[ViralDetector] {model} quota exceeded (402) - next model")
                    break

                if resp.status_code == 503:
                    print(f"[ViralDetector] {model} overloaded (503) - next model")
                    break

                if resp.status_code != 200:
                    print(f"[ViralDetector] HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                data    = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                print(f"[ViralDetector] Response ({len(content)} chars): {content[:100]}")

                parsed = _extract_json_from_response(content)
                print(f"[ViralDetector] Parsed {len(parsed)} segments from {model}")
                return parsed

            except json.JSONDecodeError as e:
                print(f"[ViralDetector] JSON parse error: {e}")
                break
            except Exception as e:
                print(f"[ViralDetector] Error with {model}: {e}")
                break

    print("[ViralDetector] All OpenRouter models failed - using local NLP")
    return None


def detect_viral_segments(
    transcript,
    num_shorts=None,
    min_duration=20.0,
    max_duration=60.0,
    min_score=25,
    openrouter_key=None,
    progress_queue=None,
):
    raw = transcript.get("segments", [])
    if not raw:
        return []

    candidates = _merge_segments(raw, min_duration, max_duration)

    if openrouter_key and openrouter_key.strip() and _REQ_OK:
        ai = _ai_detect(transcript.get("text",""), candidates, num_shorts, openrouter_key)
        if ai:
            valid = []
            for seg in ai:
                try:
                    s = float(seg.get("start_time", 0))
                    e = float(seg.get("end_time",   0))
                    if e > s and (e - s) >= 5:
                        valid.append({
                            "start_time":  s, "end_time": e,
                            "viral_score": int(seg.get("viral_score", 70)),
                            "reason":      seg.get("reason", "AI selected"),
                            "description": seg.get("description", ""),
                            "text":        "",
                        })
                except (TypeError, ValueError):
                    continue
            if valid:
                valid.sort(key=lambda x: x["start_time"])
                print(f"[ViralDetector] AI selected {len(valid)} segments")
                return valid

    if progress_queue:
        progress_queue.put({"type":"log","text":"Local NLP viral analysis...","level":"info"})

    scored = []
    for cand in candidates:
        dur   = cand["end"] - cand["start"]
        score, reasons = _local_score(cand["text"], dur)
        if score >= min_score:
            scored.append({
                "start_time":  cand["start"],
                "end_time":    cand["end"],
                "viral_score": score,
                "reason":      reasons[0] if reasons else "General interest",
                "description": " | ".join(reasons) if reasons else cand["text"][:80],
                "text":        cand["text"].strip(),
            })

    scored.sort(key=lambda x: x["viral_score"], reverse=True)

    if num_shorts is None:
        total      = raw[-1]["end"] if raw else 300
        auto_count = max(3, min(10, int(total / 180)))
        selected   = scored[:auto_count]
    else:
        selected = scored[:num_shorts]

    selected.sort(key=lambda x: x["start_time"])
    return selected
