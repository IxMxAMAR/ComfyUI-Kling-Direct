# Changelog

All notable changes to ComfyUI-Kling-Direct are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.4] — 2026-05-17

Republish to resolve registry/local version drift. Prior publish attempts
shipped 2.1.1 → 2.1.3 to the Comfy Registry ahead of the GitHub tag, leaving
the local tree at v2.1.0 while the registry advertised 2.1.3. No source
changes vs the 2.1.3 registry build — this version exists so the GitHub tag
and registry version match again.

## [2.1.0] — 2026-05-17

Audit + hardening release driven by two parallel Gemini Pro full-codebase reviews
(API/auth/HTTP/retries vs ComfyUI nodes/I/O/UX) plus first-party verification of
every critical finding. Three critical security/correctness bugs, six high-impact
fixes, ten new utility nodes, and a 49-test suite covering all hardened paths.

### Security — fix immediately

- **JWT / API-secret no longer leaks via exception tracebacks.** Pre-2.1, any
  `requests.exceptions.RequestException` raised inside `_request()` bubbled up
  carrying the original `Request` object — whose `headers["Authorization"]`
  contained the signed JWT (= identity for 30 min). A user pasting their
  ComfyUI console log into a GitHub issue would leak credentials. v2.1 wraps
  every network exception in a new `KlingAPIError` that carries only a safe
  message (`type(e).__name__`) and `raise ... from None` to detach the cause.
  Same fix applied to `upload_asset`. Regression test in
  `tests/test_jwt_and_secrets.py::test_secret_not_in_network_exception_traceback`.
- **Path-traversal block on `task_id`.** `poll_task` interpolated `task_id`
  directly into the request path. A malicious / corrupted workflow passing
  `"../materials/leak"` could pivot to a different endpoint. v2.1
  URL-percent-encodes `task_id` (and the new `get_task_status` helper).
- **SSRF + scheme guard on every download URL.** `download_to_output`,
  `download_to_tensor`, and `FastVideoSaver` previously accepted any URL the
  Kling API (or a user) supplied. A `file://` / `gopher://` URL, or a URL whose
  hostname resolved to `127.0.0.1` / RFC1918 / link-local, could exfiltrate
  local files or pivot into internal services. `_safe_url()` now rejects
  non-`http(s)` schemes and resolves the hostname to block loopback / private /
  link-local / reserved / multicast addresses.
- **Cloud Uploader requires explicit consent.** The uploader writes user media
  to public, unauthenticated paste-bin hosts (catbox.moe, 0x0.st, uguu.se,
  litterbox, tmpfiles). Catbox & 0x0 are PERMANENT. A new boolean
  `i_understand_uploads_are_public` must be ticked before the node will run;
  otherwise it raises a clear `ValueError`.
- **`_strip_none` now recurses into lists of dicts.** Kling's `image_list` /
  `video_list` / `shots` / `multi_prompt` all carry dicts inside lists. The
  old function only descended into top-level dicts, so `None` values inside
  these lists reached Kling's strict JSON parser and got a `1000 Invalid
  Parameter` error blamed on the user.

### Fixed — Critical correctness

- **cv2 capture leak on early-fail path.** `load_video_to_tensor` opened the
  capture, checked `cap.isOpened()` *outside* the `try/finally`, and raised
  on failure. cv2 holds ffmpeg file descriptors even when `isOpened()`
  returns False — leaked until the next GC. Now wraps everything in
  `try/finally` with guaranteed `cap.release()`.
- **Memory blow-up on video load fixed.** Old code built a Python list of
  numpy arrays (~3.7 GB for 600 frames of 1080p), stacked it (+3.7 GB), then
  cast to `float32` (+15 GB). Total peak: >22 GB. v2.1 pre-allocates the
  final `[N, H, W, 3]` `torch.float32` tensor up front and fills it
  frame-by-frame — ~3× less peak RAM, no intermediate list, no `np.stack`
  copy.
- **`audio_to_base64_string` no longer crashes on empty waveform.** An empty
  waveform (`shape[-1] == 0`) used to hit `repeats = ceil(MIN / 0)` and
  `ZeroDivisionError`. Now raises a clean `ValueError`. Also handles
  `sample_rate=None` (was `TypeError`) and `sample_rate <= 0`.
- **`time.monotonic()` everywhere in `poll_task`.** Was `time.time()`, which
  drifts on NTP syncs — a 10-minute video generation could time out early or
  loop forever if the system clock jumped.
- **ComfyUI cancellation propagates within ~1 s.** `poll_task` now sleeps in
  1-second chunks and checks `comfy.model_management.throw_exception_if_processing_interrupted()`
  between chunks. Pre-2.1 a user clicking "Cancel" had to wait up to 30 s for
  the next poll interval to end. Also exposes the ComfyUI `ProgressBar` so the
  UI shows estimated progress instead of appearing frozen.
- **`1106 Task Failed` moved from transient to permanent.** Retrying just
  re-queries a known-failed task; net effect was wasted requests + delayed
  error report.
- **`Retry-After` header honored properly.** Was hard-capped at 60 s and
  silently fell back to 5 s on any non-integer value. Now parses
  RFC 7231 HTTP-date format (`Wed, 01 Jan 2099 00:00:00 GMT`) and caps at
  300 s (was 60 s).
- **LipSync `voice_speed` omitted in `audio2video` mode.** Was always set to
  `1.0`, risking a `1000 Invalid Parameter` from Kling for an unexpected
  field in the audio-driven flow. Now `None` unless mode is `text2video`.
- **`download_to_output`, `download_to_tensor`, `FastVideoSaver`: retries +
  partial-file cleanup.** Single-shot downloads used to die on the first
  flaky-CDN response and leave a half-written file in `output/`. Now retry
  3× with exponential backoff and `os.remove` the partial file on each
  failure.
- **`tensor_to_pil` / `tensor_to_base64_string` handles RGBA inputs.** Some
  ComfyUI nodes emit 4-channel images; the old code crashed in
  `Image.fromarray(arr, mode="RGB")` when alpha was present. Now drops alpha
  cleanly.
- **`EMPTY_AUDIO` no longer shared by reference.** `load_audio_to_tensor`'s
  fallback returned `dict(EMPTY_AUDIO)` — a *shallow* copy, so the waveform
  tensor was shared. In-place mutation downstream corrupted the singleton
  for every subsequent call in the worker. New `_empty_audio()` helper
  always clones.
- **Image base64 auto-downgrades to JPEG when >8 MB.** Lossless PNG for 4K
  images easily exceeds Kling's ~10 MB JSON payload limit. v2.1 tries PNG
  first, falls back to JPEG q95/90/85/75 until under the limit. Warns if
  still oversized.

### Fixed — Medium

- **High-quality audio resampling via `torchaudio.functional.resample`.**
  Pre-2.1 used `scipy.interp1d` (linear) or — if scipy was missing — a
  zero-order-hold via `np.linspace`. The latter produced audible aliasing.
  `torchaudio` is always present in a ComfyUI env, so v2.1 uses it first.
- **JWT `nbf` widened from -5 s to -60 s.** Tighter `nbf` rejected tokens
  when the user's local clock was even slightly ahead of Kling's gateway.
- **Polling backoff extended for long tasks + jitter.** Pre-2.1 capped at 8 s,
  so a 15-min video task hammered Kling ~110 times. v2.1 escalates to 15 s
  after 2 min and 30 s after 5 min, with `random.uniform(0, 2)` jitter to
  prevent thundering herds.
- **Token cached on instance for ~25 min.** Was re-HMACed on every request
  including every poll iteration.
- **Bounded LRU client cache + thread lock.** Old `_client_cache` was an
  unbounded module-global dict updated without a lock. v2.1 uses
  `OrderedDict` with `maxsize=8`, a `threading.Lock`, and closes the
  evicted client's `requests.Session` to release sockets. `KlingClient`
  also has a `close()` method and `__del__` fallback.
- **`upload_asset` MIME detection via `mimetypes.guess_type`.** Was a 3-entry
  hardcoded map (png/mp4/wav) that mis-labeled jpg/jpeg/mp3/m4a/etc.
- **`KlingDirect_TTS` exposes `voice_speed` and `voice_language`.** Pre-2.1
  hardcoded both, making it strictly less useful than `KlingDirect_TTSAdvanced`.
- **Filenames truncated to 200 chars.** Avoids Windows `MAX_PATH` (260)
  failures when a long URL or prompt-derived filename hits the limit.
- **Filename extension allow-list.** `download_to_output` now rejects
  URL-derived extensions that aren't `[a-z0-9]{1,8}`.
- **Exception chaining via `KlingAPIError`.** All error paths now raise a
  single typed exception with `code` and `status_code` attrs (was bare
  `Exception(f"...")`).
- **Added `1100 Invalid Video Duration` to the error-code map.**

### Added — New nodes (10)

- **`KlingDirect_RegionSelector`** (Config) — switch between
  Singapore (default) / China / US Kling endpoints, with a `custom_base_url`
  field for self-hosted proxies. Wraps an existing auth.
- **`KlingDirect_CameraPreset`** (Config) — 15 cinematic presets
  (`orbit_left`, `dolly_in`, `zoom_in`, `pan_right`, `tilt_up`, `crane_up`,
  `roll_cw`, etc.) emit a `KLING_CAMERA` dict ready to wire into
  text-to-video / image-to-video. Includes an `intensity` scaler.
- **`KlingDirect_AspectRatioPicker`** (Config) — given an IMAGE tensor,
  outputs the closest valid Kling `aspect_ratio` string. Perfect for I2V
  pipelines where you want the output ratio to match the input.
- **`KlingDirect_CostEstimator`** (Config) — pure local — no API call.
  Heuristic credit cost for text2video / image2video / image / TTS /
  upscale by model + duration + mode + n.
- **`KlingDirect_TaskStatus`** (Config) — one-shot status check (no
  polling, no download). Returns `status` string + raw `result_json` for
  asynchronous workflow chaining.
- **`KlingDirect_LipSyncFromUrl`** (Video) — convenience wrapper accepting
  `video_url` + `audio_url` directly. Skips the local download-and-reupload
  round-trip when both assets are already hosted.
- **`KlingDirect_KeyframeVideo`** (Video) — image-to-video with both
  **start** and **end** frame (Kling's `image_tail`). Smoothly interpolates.
- **`KlingDirect_VideoToFile`** (Config, OUTPUT_NODE) — write a ComfyUI
  IMAGE batch to a real `.mp4` via `cv2.VideoWriter`. Useful for exporting
  Kling output without VideoHelperSuite. Choice of `mp4v` / `avc1` / `MJPG`.
- **`KlingDirect_ApiHealthCheck`** (Config) — pings `/v1/videos/effect-templates`
  to verify auth + connectivity. Returns `is_healthy` (BOOLEAN) + status
  message. Great for debugging "why isn't my workflow running."
- **`KlingDirect_VoiceCatalog`** (Config) — pure local. Dumps the preset
  voice catalog as formatted JSON for browsing.

### Added — Tests (49)

- **`tests/test_jwt_and_secrets.py`** (10 tests) — JWT structure (HS256
  RFC 7519), bit-exact HMAC signature verification, token caching, the
  secret-leakage regression (mocked `RequestException` with an
  `Authorization` header attached), debug-log scrub, `_strip_none` list
  recursion, `Retry-After` parsing (seconds + HTTP-date + garbage + cap).
- **`tests/test_polling_and_retries.py`** (9 tests) — `1106` permanence,
  poll termination on `succeed`, raising on `fail`, URL-encoding of
  `task_id`, timeout, monotonic-clock assertion, transient HTTP retry,
  permanent Kling error fast-fail, ComfyUI interrupt propagation.
- **`tests/test_conversions_and_nodes.py`** (30 tests) — tensor↔base64
  roundtrip with RGB and RGBA, empty / None / negative `sample_rate`
  guards, `_empty_audio()` cloning, `_sanitize_filename` Windows
  reserved chars + length cap, `_safe_url` scheme guard + loopback block,
  cv2 cap-release-on-failed-open, `download_to_output` retry +
  no-partial-files, all 10 new nodes (camera preset axis values,
  aspect-ratio picker direction, cost estimator scaling, voice catalog
  JSON shape, region selector custom URL precedence, MP4 writer round-trip,
  cloud uploader consent gate).

Run: `python -m pytest tests/` — **49/49 passing**.

### Deferred to v2.2

- ComfyUI `ProgressBar` is wired but Kling has no real `%` field — we
  estimate from `elapsed / timeout`. Worth replacing if Kling adds a
  proper progress field.
- A real `KlingDirect_TaskCancel` node (Kling's cancel endpoint isn't
  publicly documented yet — verify before shipping).
- Async / non-blocking generation pair (`Submit` → `Receive`) — needs
  ComfyUI workflow-state plumbing.
- `KlingDirect_VoiceBrowser` dynamic dropdown that fetches voices via API
  (Kling's `voice/list` is undocumented).
- Auto-tweak prompt + retry on `1302 Content Policy` (UX risk — silently
  altering user input).

## [2.0.0] — 2026-04-26

Previous release. Notable fixes per git history:
- cv2 cap leak on MemoryError
- sample_rate guard
- LipSync text2video inputs
- Upscale shared empty-audio tensor
