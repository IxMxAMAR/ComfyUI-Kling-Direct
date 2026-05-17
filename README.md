# ComfyUI-Kling-Direct

Full-featured ComfyUI custom-node pack for [Kling AI](https://app.klingai.com/global/)'s direct API. Video generation (text-to-video, image-to-video, lip-sync, advanced face-aware lip-sync, motion control, avatar, video extend, omni-video, effects), image generation (text-to-image, image-omni, image extend, virtual try-on, multi-shot, recognize, upscale), and audio (TTS, text-to-audio, voice cloning, video-to-audio).

42 nodes. 49 tests. Direct API — no third-party middleware.

## Install

ComfyUI Manager → **Install Custom Nodes** → search **"Kling Direct"** → Install. Restart ComfyUI.

Manual:
```
cd ComfyUI/custom_nodes
git clone https://github.com/IxMxAMAR/ComfyUI-Kling-Direct
```

The pack uses only stdlib + `requests` / `Pillow` / `numpy` / `torch` / `opencv-python` (already in ComfyUI).

## Authentication

1. Get a Kling **access key** + **secret key** at https://app.klingai.com/global/dev (requires KYC activation on a new account).
2. Drop in the **Kling AI Authentication** node, paste the keys (they're masked). Or set the env vars `KLING_ACCESS_KEY` / `KLING_SECRET_KEY` and leave the fields blank.
3. Wire the `auth` output into every other Kling node.

A new **Kling API Health Check** node verifies auth + connectivity in one click — `is_healthy` BOOLEAN + a status string. Use it to debug "why isn't my workflow running."

## Region

Optional **Kling Region Selector** node sits between Auth and the rest of the workflow and switches the API base URL:
- `singapore` — `https://api-singapore.klingai.com` (default; global accounts)
- `china` — `https://api.klingai.com`
- `us` — `https://api-us.klingai.com`
- custom — paste a proxy / self-hosted gateway

## Nodes

### Video
- **Text to Video** — full Kling v3 / v2.5-turbo / v2.6 / v2-master / v1.6 with aspect ratio, duration, mode (pro/std), sound, cfg_scale, shot_type, optional camera control.
- **Image to Video** — image-driven generation, optional `image_tail` for end frame, camera control.
- **Keyframe Video (Start+End)** *(v2.1 new)* — convenience wrapper for image-to-video that takes a start AND end image.
- **Video Omni** — Kling Omni model with multi-image / video references via `@image1`, `@video1` prompt syntax.
- **Video Extend** — append N more seconds to a previous Kling video by `video_id`.
- **Lip Sync** — standard lip-sync, `audio2video` or `text2video` (TTS) mode.
- **Lip Sync (URLs)** *(v2.1 new)* — convenience wrapper for the URL-only case.
- **Advanced Lip Sync** — face-aware lip-sync with explicit face_index + volume.
- **Motion Control** — apply motion from a reference video to a still image.
- **Avatar Generation** — digital-human animation from an image + optional audio.

### Image
- **Image Generation** — Kling v3 text-to-image, batch up to 9.
- **Image Omni** — image-conditioned generation with reference images.
- **Image Extend** — outpainting / aspect-ratio expansion.
- **Virtual Try-On** — Kolors clothing transfer.
- **Multi-Shot** — up to 6 consistent shots from a single prompt.
- **Image Recognize** — Kling vision model description.

### Audio
- **Text to Audio** — ambient / SFX generation by text.
- **Text to Speech** *(v2.1: now exposes voice_speed + voice_language)* — quick TTS with a voice_id.
- **TTS Advanced** — full TTS with speed + language.
- **Voice Clone** — clone a voice from an audio sample; returns a reusable `voice_id`.
- **Video to Audio** — extract / generate audio from a video URL.

### Effects
- **Video Effects** — `hug`, `kiss`, `heart`, etc. on one or two images.
- **Effect Templates** — fetch available templates JSON.
- **AI Upscale** — image or video upscaling.

### Config / utilities
- **Kling AI Authentication** — access_key + secret_key + debug toggle.
- **Region Selector** *(v2.1)* — switch API endpoint.
- **Camera Control** — manual 6-axis camera input.
- **Camera Preset** *(v2.1)* — 15 cinematic presets (`orbit_left`, `dolly_in`, `zoom_in`, `pan_right`, etc.) with intensity scaler.
- **Aspect Ratio Picker** *(v2.1)* — closest valid Kling ratio for a given IMAGE.
- **Cost Estimator** *(v2.1)* — local credit estimate by model + duration + mode.
- **Task Status** *(v2.1)* — one-shot status check (no polling, no download) for async chaining.
- **API Health Check** *(v2.1)* — verify auth + connectivity.
- **Voice Selector** — preset voice dropdown.
- **Voice Catalog** *(v2.1)* — dump the full preset catalog as JSON.
- **Element / Asset Upload** — upload arbitrary assets to Kling materials API.
- **Video Loader / Raw File Loader / Raw File Saver** — local I/O helpers.
- **Cloud Uploader** — push media to catbox / litterbox / 0x0 / uguu / tmpfiles for URL-based Kling inputs. **Requires explicit consent** (see Security below).
- **Fast Video Saver** — stream-download a video URL to disk without loading into a tensor (OOM safe).
- **Video to File (MP4)** *(v2.1)* — write a ComfyUI IMAGE batch to `.mp4` via cv2.VideoWriter.

## v2.1 Highlights

- **Security:** JWT / API secret no longer leaks via exception tracebacks; SSRF + scheme guard on all downloads; cloud uploader requires explicit consent; path-traversal block on `task_id`.
- **Memory:** `load_video_to_tensor` pre-allocates the final tensor — ~3× less peak RAM (was OOM on 600-frame 1080p videos).
- **Correctness:** cv2 cap leak fixed on early-fail path; empty-waveform crash fixed; `time.monotonic()` everywhere; LipSync `voice_speed` omitted in `audio2video`; `1106` moved to permanent (was useless retries); high-quality `torchaudio` resampling.
- **UX:** ComfyUI cancellation propagates within ~1 s; `ProgressBar` wired; long-task polling backoff extended with jitter; `Retry-After` HTTP-date support.
- **+10 new nodes** (see "v2.1 new" tags above).
- **49 tests** — `python -m pytest tests/` (all mocked, no real Kling calls).

See [`CHANGELOG.md`](CHANGELOG.md) for full details.

## Security & Privacy

- API keys are masked in the Auth node and never logged.
- v2.1 ensures the signed JWT never appears in tracebacks even when the network errors out — safe to share your ComfyUI console log when reporting a bug.
- **Cloud Uploader** uploads to PUBLIC, unauthenticated paste-bin hosts. Anyone with the URL can view the file. Catbox & 0x0 are PERMANENT. You must tick `i_understand_uploads_are_public` before the node will run.
- Set `KLING_ALLOW_INTERNAL_HTTP=1` to allow downloads from loopback / RFC1918 hosts (off by default).

## Tests

```
python -m pytest tests/
```

All 49 tests pass in ~6 seconds. Every HTTP method is mocked — **no real Kling API calls are made**.

## License

MIT — see `LICENSE`.
