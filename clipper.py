"""
AutoShorts core pipeline:
  download -> transcribe (Groq Whisper) -> analyze (Groq Llama)
  -> active-speaker crop + silence trim + B-roll + hook title
  -> karaoke captions + punch-in zoom + brand overlay
  -> render 9:16 / 1:1 / 4:5
Free stack: no GPU, no local install.
"""
import os
import re
import json
import glob
import urllib.parse
import urllib.request
import subprocess

import cv2
import numpy as np
from groq import Groq

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WHISPER_MODEL    = "whisper-large-v3"
WHISPER_FALLBACK = "whisper-large-v3-turbo"
LLM_MODEL        = "llama-3.3-70b-versatile"

FONTS_DIR = "/usr/share/fonts/truetype/dejavu"
COLS      = ["use", "start", "end", "title", "score", "reason"]

ASPECT_PRESETS = {
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "4:5":  (1080, 1350),
}

EMOJI_MAP = {
    "money": "💰", "free": "🆓", "secret": "🤫", "crazy": "🤯",
    "love": "❤️", "hate": "😤", "best": "🏆", "worst": "💀",
    "fast": "⚡", "easy": "👌", "hard": "😰", "win": "🏆",
    "lose": "📉", "growth": "📈", "idea": "💡", "warning": "⚠️",
    "stop": "🛑", "start": "🚀", "truth": "💯", "fake": "🚩",
}

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------------------------------- #
# Shell / probe helpers
# --------------------------------------------------------------------------- #
def _run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{p.stderr[-1500:]}"
        )
    return p.stdout


def probe_duration(path):
    out = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", path])
    data = json.loads(out)
    dur = float(data["format"]["duration"])
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    return dur, int(stream["width"]), int(stream["height"])


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def _write_cookies_from_env(workdir):
    content = os.environ.get("YTDLP_COOKIES_CONTENT")
    if not content:
        return None
    path = os.path.join(workdir, "cookies_env.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def download_video(url, workdir, cookies_path=None):
    import yt_dlp

    if not cookies_path:
        cookies_path = _write_cookies_from_env(workdir)

    outtmpl = os.path.join(workdir, "source.%(ext)s")

    def _opts(clients):
        o = {
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 5,
            "fragment_retries": 5,
            "concurrent_fragment_downloads": 4,
            "extractor_args": {"youtube": {"player_client": clients}},
        }
        if cookies_path:
            o["cookiefile"] = cookies_path
        return o

    last_err = None
    for clients in (["android", "web_safari"], ["ios", "web"], ["tv"], ["web"]):
        try:
            with yt_dlp.YoutubeDL(_opts(clients)) as ydl:
                ydl.download([url])
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err:
        raise RuntimeError(
            "Download failed. YouTube may be blocking the server. "
            "Add a YTDLP_COOKIES_CONTENT secret, upload cookies.txt, "
            "or upload the video file directly.\n"
            f"Details: {last_err}"
        )

    files = [f for f in glob.glob(os.path.join(workdir, "source.*"))
             if not f.endswith(".part")]
    if not files:
        raise RuntimeError("Download produced no file.")
    src = files[0]
    if not src.endswith(".mp4"):
        dst = os.path.join(workdir, "source.mp4")
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
              "-i", src, "-c", "copy", dst])
        src = dst
    return src


# --------------------------------------------------------------------------- #
# 2. Transcription (Groq Whisper, chunked for long audio)
# --------------------------------------------------------------------------- #
def extract_audio(video, workdir):
    out = os.path.join(workdir, "audio.mp3")
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video,
          "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", out])
    return out


def _split_audio(audio, workdir, chunk_secs=1200):
    pattern = os.path.join(workdir, "chunk_%03d.mp3")
    for f in glob.glob(os.path.join(workdir, "chunk_*.mp3")):
        os.remove(f)
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio,
          "-f", "segment", "-segment_time", str(chunk_secs),
          "-c", "copy", pattern])
    files = sorted(glob.glob(os.path.join(workdir, "chunk_*.mp3")))
    return [(f, i * chunk_secs) for i, f in enumerate(files)]


def _words_from_segments(segments):
    words = []
    for s in segments:
        toks = s["text"].split()
        if not toks:
            continue
        step = (s["end"] - s["start"]) / len(toks)
        for i, t in enumerate(toks):
            words.append({
                "word": t,
                "start": s["start"] + i * step,
                "end":   s["start"] + (i + 1) * step,
            })
    return words


def transcribe_audio(audio_path, client, language=None):
    size_mb = os.path.getsize(audio_path) / 1e6
    chunks = ([(audio_path, 0.0)] if size_mb <= 24
              else _split_audio(audio_path, os.path.dirname(audio_path)))

    all_words, all_segments = [], []
    for path, offset in chunks:
        kwargs = dict(
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        if language and language != "auto":
            kwargs["language"] = language
        try:
            with open(path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=(os.path.basename(path), fh.read()), **kwargs)
        except Exception:  # noqa: BLE001
            kwargs["model"] = WHISPER_FALLBACK
            with open(path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=(os.path.basename(path), fh.read()), **kwargs)

        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        for w in data.get("words") or []:
            all_words.append({"word": w["word"].strip(),
                              "start": float(w["start"]) + offset,
                              "end":   float(w["end"]) + offset})
        for s in data.get("segments") or []:
            all_segments.append({"start": float(s["start"]) + offset,
                                 "end":   float(s["end"]) + offset,
                                 "text":  s["text"].strip()})

    if not all_words and all_segments:
        all_words = _words_from_segments(all_segments)
    if not all_segments and all_words:
        all_segments = [{"start": all_words[0]["start"],
                         "end":   all_words[-1]["end"],
                         "text":  " ".join(w["word"] for w in all_words)}]
    return all_words, all_segments


# --------------------------------------------------------------------------- #
# 3. Highlight detection
# --------------------------------------------------------------------------- #
def _snap(t, boundaries, tol=1.5):
    if not boundaries:
        return t
    best = min(boundaries, key=lambda b: abs(b - t))
    return best if abs(best - t) <= tol else t


def analyze_clips(segments, client, n_clips, min_len, max_len, duration):
    transcript = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments
    )[:120000]

    prompt = f"""You are a viral short-form video editor. Pick the {n_clips} most clip-worthy moments from this transcript for YouTube Shorts / TikTok / Reels.

HARD RULES:
- Each clip MUST be between {min_len} and {max_len} seconds long.
- Clips must NOT overlap.
- Use ONLY the timestamps present in the transcript. Never invent times.
- A clip must open with a self-contained hook and end on a payoff/punchline.
- Strong signals: surprising claims, strong opinions, emotional stories, conflict, actionable tips, funny moments, numbers/stats.

TRANSCRIPT (format: [start-end] text):
{transcript}

Return ONLY valid JSON in this shape:
{{"clips":[{{"start":12.5,"end":48.0,"title":"max 8 words","hook_score":8,"reason":"max 20 words"}}]}}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content
    try:
        clips = json.loads(raw).get("clips", [])
    except json.JSONDecodeError:
        raise RuntimeError(f"LLM returned invalid JSON:\n{raw[:500]}")

    boundaries = sorted({round(s["start"], 2) for s in segments} |
                        {round(s["end"], 2) for s in segments})

    cleaned = []
    for c in clips:
        try:
            start = float(c["start"])
            end = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = _snap(max(0.0, start), boundaries)
        end = _snap(min(duration, end), boundaries)
        if end - start < min_len * 0.6 or end - start > max_len * 1.5:
            continue
        cleaned.append({
            "start": round(start, 2),
            "end":   round(min(end, start + max_len), 2),
            "title": str(c.get("title", "Clip"))[:80],
            "score": int(c.get("hook_score", 5) or 5),
            "reason": str(c.get("reason", ""))[:160],
        })

    cleaned.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for c in cleaned:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in kept):
            kept.append(c)
        if len(kept) >= n_clips:
            break
    if not kept:
        raise RuntimeError("No valid clips found. Try a longer video or more clips.")
    return sorted(kept, key=lambda c: c["start"])


# --------------------------------------------------------------------------- #
# 4. Cropping (face detection + scene-aware active speaker)
# --------------------------------------------------------------------------- #
def _target_crop_dims(W, H, aspect="9:16"):
    tw, th = ASPECT_PRESETS[aspect]
    target_ar = tw / th
    if W / H > target_ar:
        cw, ch = int(round(H * target_ar)), H
    else:
        cw, ch = W, int(round(W / target_ar))
    return cw // 2 * 2, ch // 2 * 2


def detect_crop(video_path, start, end, aspect="9:16"):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    centers = []
    t = start
    min_face = max(24, int(H * 0.08))
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _cascade.detectMultiScale(gray, 1.15, 5,
                                          minSize=(min_face, min_face))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + w / 2.0, y + h / 2.0))
        t += 0.75
    cap.release()

    cw, ch = _target_crop_dims(W, H, aspect)
    if centers:
        cx = float(np.median([c[0] for c in centers]))
        cy = float(np.median([c[1] for c in centers]))
    else:
        cx, cy = W / 2.0, H / 2.0

    x = int(round(min(max(cx - cw / 2, 0), max(0, W - cw)))) // 2 * 2
    y = int(round(min(max(cy - ch / 2, 0), max(0, H - ch)))) // 2 * 2
    return cw, ch, x, y


def _scene_cuts(video_path, start, end, threshold=0.3):
    out = _run(["ffmpeg", "-hide_banner", "-loglevel", "info",
                "-accurate_seek", "-ss", f"{start:.3f}",
                "-t", f"{end - start:.3f}", "-i", video_path,
                "-filter:v", f"select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-"])
    cuts = [start]
    for line in out.splitlines():
        if "pts_time:" in line:
            try:
                t = float(line.split("pts_time:")[1].split()[0])
                cuts.append(start + t)
            except (IndexError, ValueError):
                pass
    cuts.append(end)
    return sorted(set(cuts))


def detect_speaker_crops(video_path, start, end, aspect="9:16"):
    cuts = _scene_cuts(video_path, start, end)
    if len(cuts) <= 2:
        cw, ch, x, y = detect_crop(video_path, start, end, aspect)
        return [(start, end, cw, ch, x, y)]

    scenes = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s < 0.6:
            continue
        cw, ch, x, y = detect_crop(video_path, s, e, aspect)
        scenes.append((s, e, cw, ch, x, y))
    return scenes or [(start, end, *detect_crop(video_path, start, end, aspect))]


# --------------------------------------------------------------------------- #
# 5. B-roll from Pexels (free API)
# --------------------------------------------------------------------------- #
def fetch_broll(keywords, workdir, per_page=1):
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key or not keywords:
        return None
    query = urllib.parse.quote(" ".join(keywords[:3]))
    url = (f"https://api.pexels.com/videos/search?"
           f"query={query}&per_page={per_page}&orientation=landscape")
    req = urllib.request.Request(url, headers={"Authorization": api_key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        videos = data.get("videos", [])
        if not videos:
            return None
        v = videos[0]
        files = sorted(v["video_files"], key=lambda f: f.get("width", 0))
        pick = next((f for f in files
                     if 720 <= f.get("width", 0) <= 1280), files[0])
        out = os.path.join(workdir, f"broll_{v['id']}.mp4")
        urllib.request.urlretrieve(pick["link"], out)
        return out
    except Exception:
        return None


def broll_keywords_for_clip(clip_title, client):
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content":
                f"Give 2-3 short visual keywords (single English words) for "
                f"stock b-roll footage matching this clip title: "
                f"\"{clip_title}\"\n"
                'Return ONLY JSON: {"keywords":["word1","word2"]}'}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=80,
        )
        return json.loads(resp.choices[0].message.content).get("keywords", [])
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# 6. Captions (ASS) — karaoke, keyword colours, emojis, hook title
# --------------------------------------------------------------------------- #
def _ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(text):
    return (text.replace("\\", "/")
                .replace("{", "(").replace("}", ")")
                .replace("\n", " ").strip())


def _make_lines(words, max_words=4, max_chars=16,
                max_dur=2.4, max_gap=0.6):
    lines, cur = [], []
    for w in words:
        if cur:
            joined = " ".join(x["word"] for x in cur + [w])
            if (len(cur) >= max_words or len(joined) > max_chars
                    or w["end"] - cur[0]["start"] > max_dur
                    or w["start"] - cur[-1]["end"] > max_gap):
                lines.append(cur)
                cur = [w]
                continue
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _pick_emoji(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    return EMOJI_MAP.get(w, "")


def build_ass_v2(words, clip_start, clip_end, path,
                 highlight="&H00FFFF&", keyword_color="&H00FF00&",
                 normal="&HFFFFFF&", emojis=True, pop_effect=True):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,88,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,7,4,2,60,60,340,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = clip_end - clip_start
    local = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]

    keyword_set = {w["word"].strip(".,!?\"'").lower()
                   for w in local
                   if len(w["word"]) > 6 or w["word"][:1].isupper()}

    events = []
    for line in _make_lines(local):
        n = len(line)
        for i, w in enumerate(line):
            t0 = max(0.0, w["start"] - clip_start)
            t1 = ((line[i + 1]["start"] - clip_start)
                  if i + 1 < n else (line[-1]["end"] - clip_start))
            t1 = min(t1, dur)
            if t1 - t0 < 0.05:
                t1 = t0 + 0.05
            if t0 >= dur:
                continue

            parts = []
            for j, lw in enumerate(line):
                txt = _esc(lw["word"])
                is_kw = lw["word"].strip(".,!?\"'").lower() in keyword_set
                col = keyword_color if is_kw else normal
                if j == i:
                    col = highlight
                eff = ""
                if pop_effect and j == i:
                    eff = r"{\fscx110\fscy110\t(0,120,\fscx100\fscy100)}"
                parts.append(f"{eff}{{\\c&{col}&}}{txt}{{\\c&{normal}&}}")

            line_text = " ".join(parts)
            if emojis:
                em = _pick_emoji(line[-1]["word"])
                if em:
                    line_text += f"  {em}"

            events.append(
                f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Cap,,0,0,0,,"
                + line_text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return path


def build_hook_ass(title, clip_start, clip_end, path, duration=2.5):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,DejaVu Sans,110,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,9,5,8,40,40,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    safe = _esc(title)
    event = (f"Dialogue: 0,0:00:00.00,{_ass_time(duration)},Hook,,0,0,0,,"
             f"{{\\fad(200,300)}}{safe}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + event + "\n")
    return path


# --------------------------------------------------------------------------- #
# 7. Main renderer
# --------------------------------------------------------------------------- #
def render_clip(workdir, video_name, clip, words, out_name,
                aspect="9:16", captions=True, hook=True, emojis=True,
                broll=True, zoom=True, brand_logo=None,
                brand_color="&H00FFFF&"):

    start, end = clip["start"], clip["end"]
    dur = end - start
    video_path = os.path.join(workdir, video_name)
    _, W, H = probe_duration(video_path)
    out_w, out_h = ASPECT_PRESETS[aspect]

    # Speaker-aware crops -> use the longest scene for the static base
    scenes = detect_speaker_crops(video_path, start, end, aspect)
    main_scene = max(scenes, key=lambda s: s[1] - s[0])
    _, _, cw, ch, x, y = main_scene
    cw, ch = cw // 2 * 2, ch // 2 * 2
    x, y = x // 2 * 2, y // 2 * 2

    # Video filter chain
    vf_parts = [f"crop={cw}:{ch}:{x}:{y}",
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1"]

    if zoom:
        vf_parts.append(
            f"zoompan=z='min(zoom+0.0008,1.06)':d=1:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s={out_w}x{out_h}:fps=30"
        )

    # Captions + hook
    ass_files = []
    if captions:
        cap_name = f"cap_{re.sub(r'[^A-Za-z0-9]', '_', out_name)}.ass"
        build_ass_v2(words, start, end, os.path.join(workdir, cap_name),
                     highlight=brand_color, emojis=emojis)
        ass_files.append(cap_name)

    if hook and clip.get("title"):
        hook_name = f"hook_{re.sub(r'[^A-Za-z0-9]', '_', out_name)}.ass"
        build_hook_ass(clip["title"], start, end,
                       os.path.join(workdir, hook_name))
        ass_files.append(hook_name)

    for a in ass_files:
        vf_parts.append(f"ass={a}:fontsdir={FONTS_DIR}")

    # Optional B-roll
    broll_path = None
    if broll:
        broll_path = fetch_broll(clip.get("broll_keywords") or [], workdir)

    # Assemble command with filter_complex (works for all overlay cases)
    inputs = ["-accurate_seek", "-ss", f"{start:.3f}",
              "-t", f"{dur:.3f}", "-i", video_name]
    filter_parts = [f"[0:v]{','.join(vf_parts)}[v0]"]
    last_v = "v0"
    input_idx = 1

    if broll_path:
        inputs += ["-i", os.path.basename(broll_path)]
        filter_parts.append(
            f"[{input_idx}:v]scale=360:-2,trim=duration=3,"
            f"setpts=PTS-STARTPTS[broll]"
        )
        filter_parts.append(
            f"[{last_v}][broll]overlay=W-w-30:30:"
            f"enable='between(t,{dur/3:.1f},{dur/3+3:.1f})'[v1]"
        )
        last_v = "v1"
        input_idx += 1

    if brand_logo:
        inputs += ["-i", os.path.basename(brand_logo)]
        filter_parts.append(f"[{input_idx}:v]scale=120:-1[logo]")
        filter_parts.append(f"[{last_v}][logo]overlay=30:30[vout]")
        last_v = "vout"

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           *inputs,
           "-filter_complex", filter_complex,
           "-map", f"[{last_v}]",
           "-map", "0:a:0",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-r", "30",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
           "-movflags", "+faststart",
           "-t", f"{dur:.3f}",
           out_name]

    _run(cmd, cwd=workdir)
    return os.path.join(workdir, out_name)