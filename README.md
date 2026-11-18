---
title: AutoShorts
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AutoShorts

AI YouTube → Shorts clipper. **100% free stack — no GPU, no local install.**

- **Whisper large-v3** via Groq (free tier) → word-level transcript
- **Llama 3.3 70B** via Groq → viral-moment detection + virality score
- **OpenCV Haar cascades** → face-tracked 9:16 crops
- **FFmpeg + libass** → cut, crop, karaoke captions, punch-in zoom,
  B-roll overlay, hook title, loudness normalisation (-14 LUFS)

## Features

- ✅ AI highlight detection with virality scores + reasons
- ✅ Word-by-word karaoke captions with keyword colours + emojis
- ✅ Active-speaker scene-aware reframing (9:16 / 1:1 / 4:5)
- ✅ Hook title overlay for the first 2.5 s
- ✅ Optional B-roll from Pexels (free API)
- ✅ Punch-in zoom, brand logo watermark, custom highlight colour
- ✅ Human-in-the-loop table editing before render
- ✅ Auto-generated titles, descriptions & hashtags (available via API)

## Setup (5 minutes)

1. Get a free Groq key at https://console.groq.com/keys
   (no credit card required).
2. Optional: get a free Pexels key at https://www.pexels.com/api/ for B-roll.
3. Create a new **Gradio** Space on Hugging Face (free CPU tier).
4. Upload `app.py`, `clipper.py`, `requirements.txt`, `packages.txt`, `README.md`.
5. **Settings → Variables and secrets → Add a secret**:
   - `GROQ_API_KEY` (required)
   - `PEXELS_API_KEY` (optional — enables B-roll)
   - `YTDLP_COOKIES_CONTENT` (optional — paste contents of cookies.txt
     if YouTube bot-checks block downloads)
6. Space rebuilds automatically. Open the app URL and paste a YouTube link.

## Free-tier limits to know

- **Groq Whisper:** ~2,000 req/day, ~7,200 audio-seconds/hour.
  Audio is auto-chunked at 15 min, so long videos work.
- **Groq Llama 3.3 70B:** generous free rate limits.
- **CPU render:** ~1–3 min per 60 s clip at 1080×1920. 720×1280 is ~2× faster.
- **Space sleep:** free Spaces sleep after 48 h inactivity.
  First request after sleep takes ~30–60 s to wake up.

## If YouTube blocks the download

YouTube sometimes bot-checks datacenter IPs. Fixes, in order:

1. Add the `YTDLP_COOKIES_CONTENT` secret (paste full contents of a
   `cookies.txt` exported with the "Get cookies.txt LOCALLY" extension).
2. Upload a `cookies.txt` file in the app's "cookies" input for that job.
3. Upload the video file directly instead of a URL.

## Known limitations

- The crop uses a **static median** of face centres per scene. Great for
  podcasts/talking heads; weaker for fast cutaways. True frame-by-frame
  tracking is a v2 upgrade.
- Caption font is DejaVu Sans (the only font guaranteed on free CPU).
- Punch-in zoom on CPU is the slowest step — turn it off while testing.

## Updating later

- Browser: Files tab → click a file → edit → Commit. Space rebuilds automatically.
- Git: `git add . && git commit -m "update" && git push`.