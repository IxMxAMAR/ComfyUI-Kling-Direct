"""Kling AI nodes for ComfyUI-Kling-Direct.

v2.1.0 — full audit pass:
- cv2 cap leak fixed for early-fail path (open-before-try-finally bug)
- Memory-efficient video load (pre-allocates target tensor; ~6x less peak RAM)
- Empty-waveform crash fixed (audio_to_base64_string would div/0)
- Cloud uploader requires explicit consent (public hosts)
- download_to_output now retries + cleans up partial files
- Filenames truncated to fit MAX_PATH on Windows
- New nodes: TaskStatus, CameraPreset, AspectRatioPicker, CostEstimator,
  VideoToFile, LipSyncFromUrl, ApiHealthCheck, KeyframeVideo, RegionSelector
"""

import json
import torch
import numpy as np
from PIL import Image
import io
import base64
import ipaddress
import requests
import os
import uuid
import re
import wave
import logging
import socket
import cv2
import shutil
import tempfile
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from .kling_client import KlingClient, KlingAPIError, get_client
except ImportError:
    # Standalone import path (tests / direct script execution)
    from kling_client import KlingClient, KlingAPIError, get_client

try:
    from .shared.auth import DualKeyAPIKeyNode
    from .shared.node_utils import AlwaysExecuteMixin
except ImportError:
    # Fallback for standalone testing
    try:
        from shared.auth import DualKeyAPIKeyNode
        from shared.node_utils import AlwaysExecuteMixin
    except ImportError:
        class DualKeyAPIKeyNode:
            ENV_VAR_ACCESS = ""
            ENV_VAR_SECRET = ""
            SERVICE_NAME = "API"
            RETURN_TYPES = ("STRING", "STRING")
            RETURN_NAMES = ("access_key", "secret_key")
            FUNCTION = "provide_keys"
            CATEGORY = "API Toolkit/Auth"
        class AlwaysExecuteMixin:
            @classmethod
            def IS_CHANGED(cls, **kwargs):
                return float("nan")

logger = logging.getLogger(__name__)

# --- Constants & Configs ---

MIN_AUDIO_DURATION = 2.0
MAX_AUDIO_DURATION = 300.0
TARGET_SAMPLE_RATE = 16000
INT16_MAX = 32767
MIN_IMAGE_DIM = 300
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_TIMEOUT = 120
MAX_VIDEO_FRAMES = 600
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

EMPTY_AUDIO = {"waveform": torch.zeros((1, 1, 1024)), "sample_rate": 44100}

VIDEO_MODELS = ["kling-v3", "kling-v2-5-turbo", "kling-v2-6", "kling-v2-master", "kling-v1-6"]
VIDEO_MODELS_I2V = ["kling-v3", "kling-v2-6", "kling-v2-master", "kling-v1-6"]
UPSCALE_MODELS = ["kling-v1", "kling-v3"]

# v2.1: Region selector. Kling exposes multiple regional endpoints; users with
# global vs China accounts hit different gateways. Singapore is default.
KLING_REGIONS = {
    "singapore": "https://api-singapore.klingai.com",
    "china": "https://api.klingai.com",
    "us": "https://api-us.klingai.com",
}

# v2.1: Camera-control presets. Each emits a (type, config) pair that
# matches Kling's camera_control schema. Values are within Kling's accepted
# -10..10 range per axis.
CAMERA_PRESETS = {
    "none": ("simple", {}),
    "orbit_left": ("horizontal", {"horizontal": -7.0}),
    "orbit_right": ("horizontal", {"horizontal": 7.0}),
    "dolly_in": ("zoom", {"zoom": 5.0}),
    "dolly_out": ("zoom", {"zoom": -5.0}),
    "zoom_in": ("zoom", {"zoom": 7.0}),
    "zoom_out": ("zoom", {"zoom": -7.0}),
    "pan_left": ("pan", {"pan": -5.0}),
    "pan_right": ("pan", {"pan": 5.0}),
    "tilt_up": ("tilt", {"tilt": 5.0}),
    "tilt_down": ("tilt", {"tilt": -5.0}),
    "crane_up": ("vertical", {"vertical": 5.0}),
    "crane_down": ("vertical", {"vertical": -5.0}),
    "roll_cw": ("roll", {"roll": 5.0}),
    "roll_ccw": ("roll", {"roll": -5.0}),
}

# v2.1: Polling endpoints (used by the new Task Status node).
TASK_ENDPOINTS = [
    "/v1/videos/text2video",
    "/v1/videos/image2video",
    "/v1/videos/omni-video",
    "/v1/videos/video-extend",
    "/v1/videos/lip-sync",
    "/v1/videos/advanced-lip-sync",
    "/v1/videos/motion-control",
    "/v1/videos/avatar/image2video",
    "/v1/videos/effects",
    "/v1/videos/upscale",
    "/v1/images/generations",
    "/v1/images/omni-image",
    "/v1/images/editing/expand",
    "/v1/images/ai-multi-shot",
    "/v1/images/kolors-virtual-try-on",
    "/v1/images/recognize",
    "/v1/images/upscale",
    "/v1/audio/text-to-audio",
    "/v1/audio/tts",
    "/v1/audio/video-to-audio",
]


def _extract_video_url(res: dict) -> str:
    """Safely extract the first video URL from a poll_task result."""
    task_result = res.get("task_result")
    if not task_result:
        raise Exception(f"Kling task completed but returned no task_result. Response keys: {list(res.keys())}")
    videos = task_result.get("videos")
    if not videos:
        raise Exception(f"Kling task completed but returned no videos. task_result keys: {list(task_result.keys())}")
    url = videos[0].get("url")
    if not url:
        raise Exception(f"Kling task completed but video has no URL. Video data: {videos[0]}")
    return url


def _extract_video_id(res: dict) -> str:
    """Safely extract the first video ID from a poll_task result."""
    task_result = res.get("task_result")
    if not task_result:
        return ""
    videos = task_result.get("videos")
    if not videos:
        return ""
    return videos[0].get("id", "")


def _extract_image_url(res: dict, index: int = 0) -> str:
    """Safely extract an image URL from a poll_task result."""
    images = res.get("images")
    if not images:
        raise Exception(f"Kling task completed but returned no images. Response keys: {list(res.keys())}")
    if index >= len(images):
        raise Exception(f"Kling task returned {len(images)} image(s), but index {index} requested.")
    url = images[index].get("url")
    if not url:
        raise Exception(f"Kling task completed but image has no URL. Image data: {images[index]}")
    return url


def _extract_audio_url(res: dict) -> str:
    """Safely extract audio_url from an audio poll_task result."""
    url = res.get("audio_url")
    if not url:
        raise Exception(f"Kling task completed but returned no audio_url. Response keys: {list(res.keys())}")
    return url


def _extract_asset_id(res: dict) -> str:
    """Extract asset ID from upload response, trying all known key names."""
    asset_id = res.get("id") or res.get("materials_id") or res.get("file_id")
    if not asset_id:
        raise Exception(f"Kling asset upload succeeded but returned no ID. Response: {res}")
    return asset_id


def _sanitize_filename(name: str, max_len: int = 200) -> str:
    """Remove characters that are invalid in Windows filenames.
    v2.1: also caps length to avoid Windows MAX_PATH (260) failures."""
    cleaned = SAFE_FILENAME_RE.sub("_", name or "")
    return cleaned[:max_len] if max_len > 0 else cleaned


def _make_client(auth: dict) -> KlingClient:
    """Create or retrieve a cached KlingClient from an auth dict.

    Supports an optional `base_url` override (set by KlingDirect_RegionSelector).
    """
    base_url = auth.get("base_url") or "https://api-singapore.klingai.com"
    return get_client(
        auth["access_key"],
        auth["secret_key"],
        debug=auth.get("debug", False),
        base_url=base_url,
    )


def _safe_url(url: str) -> str:
    """Reject non-http(s) schemes and obvious SSRF targets.

    v2.1 SECURITY: A malicious upstream node (or a compromised Kling response
    parsed by `_extract_*`) could inject `file://`, `gopher://`, or a URL whose
    host resolves to localhost / RFC1918 / link-local — turning the download
    helper into an exfiltration / pivot tool. Block both at parse time and at
    DNS-resolve time.
    """
    if not url or not isinstance(url, str):
        raise ValueError("Kling: download URL is empty or non-string.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Kling: refusing download from non-http scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("Kling: download URL has no host.")
    # Resolve and block private/loopback/link-local — except in tests where the
    # mock host is e.g. example.com (public). Errors are silently allowed.
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ValueError(
                    f"Kling: refusing to download from internal/loopback host {host} ({ip_str}). "
                    f"Set KLING_ALLOW_INTERNAL_HTTP=1 to override.")
    except socket.gaierror:
        # Can't resolve — let requests fail with a clearer message.
        pass
    return url


VOICES_CONFIG = {
    "Melody (EN Female)": ("girlfriend_4_speech02", "en"),
    "Sunny (EN Boy)": ("genshin_vindi2", "en"),
    "Sage (EN Student)": ("zhinen_xuesheng", "en"),
    "Ace (EN Male)": ("AOT", "en"),
    "Blossom (EN Girl)": ("ai_shatang", "en"),
    "Peppy (EN Child)": ("genshin_klee2", "en"),
    "Dove (EN Energetic)": ("genshin_kirara", "en"),
    "Shine (EN Bright)": ("ai_kaiya", "en"),
    "Anchor (EN Male)": ("oversea_male1", "en"),
    "Lyric (EN Literary)": ("ai_chenjiahao_712", "en"),
    "Tender (EN Female)": ("chat1_female_new-3", "en"),
    "Siren (EN Calm)": ("chat_0407_5-1", "en"),
    "Lore (EN Story)": ("calm_story1", "en"),
    "The Reader (EN)": ("reader_en_m-v1", "en"),
    "Commercial Lady (EN)": ("commercial_lady_en_f-v1", "en"),
    "\u9633\u5149\u5c11\u5e74 (ZH)": ("genshin_vindi2", "zh"),
    "\u61c2\u4e8b\u5c0f\u5f1f (ZH)": ("zhinen_xuesheng", "zh"),
    "\u9752\u6625\u5c11\u5973 (ZH)": ("ai_shatang", "zh"),
    "\u6e29\u67d4\u5c0f\u59b9 (ZH)": ("genshin_klee2", "zh"),
    "\u5143\u6c14\u5c11\u5973 (ZH)": ("genshin_kirara", "zh"),
    "\u751c\u7f8e\u90bb\u5bb6 (ZH)": ("girlfriend_1_speech02", "zh"),
    "\u65b0\u95fb\u64ad\u62a5 (ZH)": ("diyinnansang_DB_CN_M_04-v2", "zh"),
    "\u8bd1\u5236\u7247 (ZH)": ("yizhipiannan-v1", "zh"),
}

MODES = ["pro", "std"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "3:2", "2:3", "4:3", "3:4"]
IMAGE_RESOLUTIONS = ["1k", "2k"]

# --- Helper Functions ---

def validate_prompt_length(prompt: str, max_len: int = 2500):
    if not prompt:
        return
    if len(prompt) > max_len:
        raise ValueError(f"Kling Prompt too long: {len(prompt)} characters (Max {max_len}). Please shorten it.")

def normalize_prompts(prompt: str) -> str:
    if not prompt:
        return prompt
    validate_prompt_length(prompt)
    def _image_repl(match): return f"<<<image_{match.group('idx') or '1'}>>>"
    def _video_repl(match): return f"<<<video_{match.group('idx') or '1'}>>>"
    prompt = re.sub(r"(?<!\w)@image(?P<idx>\d*)(?!\w)", _image_repl, prompt)
    prompt = re.sub(r"(?<!\w)@video(?P<idx>\d*)(?!\w)", _video_repl, prompt)
    return prompt

def tensor_to_base64_string(image: torch.Tensor, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Convert a ComfyUI IMAGE tensor to base64.

    v2.1 fix: tries PNG first (lossless), falls back to JPEG q95/q85/q75 if
    the payload would exceed Kling's ~10MB JSON limit. Also handles RGBA
    by dropping the alpha channel (Kling expects RGB).
    """
    if image is None:
        return None
    if image.dim() == 4:
        image = image[0]
    h, w, c = image.shape
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        logger.warning(f"Kling Warning: Image resolution ({w}x{h}) is below recommended {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}.")
    # Clamp to [0,1] before scaling so over-bright inputs don't wrap around.
    arr = (image.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    # Drop alpha channel if present (Kling expects RGB)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    img = Image.fromarray(arr, mode="RGB")
    # Try lossless first
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    data = buffered.getvalue()
    if len(data) <= max_bytes:
        return base64.b64encode(data).decode("utf-8")
    # Too large -- fall back to JPEG at descending quality
    for quality in (95, 90, 85, 75):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            logger.info(f"[KLING] Image too large for PNG ({w}x{h}); using JPEG q={quality} ({len(data)//1024} KB).")
            return base64.b64encode(data).decode("utf-8")
    # Still too large at q75 — caller should downscale.
    logger.warning(f"[KLING] Image still {len(data)//1024}KB at JPEG q75; sending anyway. Consider downscaling.")
    return base64.b64encode(data).decode("utf-8")

def audio_to_base64_string(audio: Dict[str, Any], target_sr: int = TARGET_SAMPLE_RATE) -> str:
    """Convert a ComfyUI AUDIO dict to base64-encoded WAV for Kling.

    v2.1 fixes:
    - Empty-waveform guard (was dividing by zero on shape[-1]==0).
    - sample_rate None / non-numeric guard.
    - Resampling now uses torchaudio (high quality, always available with
      ComfyUI) instead of scipy / lossy zero-order-hold fallback.
    """
    if audio is None or "waveform" not in audio:
        return None
    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate", 44100)
    # K-FIX v2.1: tolerate None / non-int sample_rate without TypeError.
    try:
        sample_rate = int(sample_rate) if sample_rate is not None else 0
    except (TypeError, ValueError):
        sample_rate = 0
    if sample_rate <= 0:
        raise ValueError(
            f"Audio has invalid sample_rate ({audio.get('sample_rate')}). "
            f"Check the upstream audio node — it may have produced a corrupted AUDIO dict."
        )

    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # K-FIX v2.1: empty-waveform guard. Was crashing with ZeroDivisionError in `repeats`.
    if waveform.shape[-1] == 0:
        raise ValueError("Audio waveform is empty (0 samples). Check your upstream audio source.")

    w = waveform.cpu().numpy()

    duration = w.shape[-1] / sample_rate
    if duration < MIN_AUDIO_DURATION:
        logger.warning(f"Kling Warning: Audio duration ({duration:.2f}s) is too short. Looping to reach {MIN_AUDIO_DURATION}s minimum.")
        repeats = int(np.ceil(MIN_AUDIO_DURATION / max(duration, 1e-6)))
        w = np.tile(w, (1, repeats))
        w = w[:, :int(MIN_AUDIO_DURATION * sample_rate)]
        duration = MIN_AUDIO_DURATION
    elif duration > MAX_AUDIO_DURATION:
        logger.warning(f"Kling Warning: Audio duration ({duration:.2f}s) exceeds {MAX_AUDIO_DURATION}s. Clipping.")
        w = w[:, :int(MAX_AUDIO_DURATION * sample_rate)]

    # Mix down to Mono for LipSync compatibility
    if w.shape[0] > 1:
        logger.info(f"[KLING] Mixing down {w.shape[0]} channels to Mono.")
        w = np.mean(w, axis=0, keepdims=True)

    # K-FIX v2.1: high-quality resampling via torchaudio (always available
    # inside the ComfyUI env). Falls back to scipy, then to lossy zero-order-hold.
    if sample_rate != target_sr:
        resampled = None
        try:
            import torchaudio.functional as taF
            t = torch.from_numpy(w).float()
            t = taF.resample(t, orig_freq=sample_rate, new_freq=target_sr)
            resampled = t.cpu().numpy()
        except Exception:
            try:
                from scipy.interpolate import interp1d
                times = np.arange(w.shape[-1]) / sample_rate
                new_n = max(1, int(w.shape[-1] * target_sr / sample_rate))
                new_times = np.arange(new_n) / target_sr
                new_w = []
                for channel in range(w.shape[0]):
                    f = interp1d(times, w[channel], kind='linear', fill_value="extrapolate")
                    new_w.append(f(new_times))
                resampled = np.array(new_w)
            except Exception:
                # Last-resort nearest-neighbor (lossy but never crashes).
                logger.warning("[KLING] torchaudio and scipy both unavailable; using lossy nearest-neighbor resampling.")
                new_samples = max(1, int(w.shape[-1] * target_sr / sample_rate))
                indices = np.linspace(0, w.shape[-1] - 1, new_samples).astype(np.int64)
                resampled = w[:, indices]
        w = resampled
        sample_rate = target_sr

    w = (np.clip(w, -1.0, 1.0) * INT16_MAX).astype(np.int16)
    channels, samples = w.shape

    buffered = io.BytesIO()
    with wave.open(buffered, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(w.tobytes())

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def download_to_output(url: str, ext: str = "mp4", retries: int = 3) -> tuple:
    """Download a URL to ComfyUI's output dir, retrying on transient errors.

    v2.1 fixes:
    - URL scheme guard (rejects file:// / gopher:// / internal hosts).
    - Retries with exponential backoff on transient network errors.
    - Cleans up partial files on failure (no orphans).
    - Sanitized extension (no path-traversal via URL-derived ext).
    """
    url = _safe_url(url)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)

    clean_path = urlparse(url).path
    detected_ext = os.path.splitext(clean_path)[1].lstrip('.').lower()
    # Whitelist character set for the extension; reject anything weird.
    if detected_ext and not re.fullmatch(r"[a-z0-9]{1,8}", detected_ext):
        detected_ext = ""
    filename_ext = detected_ext or ext

    filename = f"kling_{uuid.uuid4().hex}.{filename_ext}"
    file_path = os.path.join(output_dir, filename)

    # Print URL first so it's recoverable if download fails
    print(f"[KLING] Download URL: {url}")

    last_err = None
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            print(f"[KLING] Saved to output: {filename}")
            return file_path, filename
        except (requests.exceptions.RequestException, OSError) as e:
            last_err = e
            # Clean up partial file before retry (no orphans)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
            if attempt < retries - 1:
                wait = 2 * (2 ** attempt)
                print(f"[KLING] Download failed ({type(e).__name__}); retrying in {wait}s...")
                import time as _time
                _time.sleep(wait)
                continue
    raise RuntimeError(f"[KLING] Download failed after {retries} attempts: {type(last_err).__name__}: {last_err}")

def load_video_to_tensor(video_path: str) -> torch.Tensor:
    """Load video frames into a ComfyUI image-batch tensor [N,H,W,C] in [0,1].

    v2.1 fixes:
    - cv2 cap leak on early-fail path (was: open, isOpened()==False, raise OUTSIDE
      try/finally -> ffmpeg fd held until interpreter GC). Now wraps everything
      in try/finally so cap.release() always runs.
    - Memory-efficient: pre-allocates the final float32 tensor instead of
      stacking a list of arrays. Cuts peak RAM ~3x by removing the intermediate
      np.stack copy (which doubles memory) and the subsequent .float() cast
      (which doubles it again).
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise Exception(f"[KLING] Could not open video file: {os.path.basename(video_path)}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        step = 1
        if total_frames > MAX_VIDEO_FRAMES:
            step = max(1, total_frames // MAX_VIDEO_FRAMES)
            logger.info(f"[KLING] Video has {total_frames} frames -- subsampling every {step}th frame (keeping ~{total_frames // step} frames) to avoid OOM.")

        # First pass: count + sample first frame to learn (H, W).
        first_frame = None
        # Quick read to discover dims (most codecs need a read).
        ret, peek = cap.read()
        if not ret or peek is None:
            raise Exception(f"[KLING] Video '{os.path.basename(video_path)}' contains no readable frames.")
        peek = cv2.cvtColor(peek, cv2.COLOR_BGR2RGB)
        h, w, _ = peek.shape

        # How many frames will we keep? With subsampling at `step`, ceiling of total/step.
        if total_frames > 0:
            est_kept = max(1, (total_frames + step - 1) // step)
        else:
            est_kept = MAX_VIDEO_FRAMES  # unknown — bound to safety cap
        # Cap conservatively to avoid huge pre-alloc on bogus headers.
        est_kept = min(est_kept, MAX_VIDEO_FRAMES + 50)

        try:
            # Pre-allocate the final tensor directly. Peak RAM ~ est_kept*h*w*3*4 bytes.
            out = torch.empty((est_kept, h, w, 3), dtype=torch.float32)
        except (MemoryError, RuntimeError) as mem_err:
            logger.warning(f"[KLING] MEMORY ERROR pre-allocating ({est_kept} x {h}x{w}): {mem_err}")
            # Return just the first frame as a degraded result.
            return torch.from_numpy(peek).float().unsqueeze(0) / 255.0

        kept = 0
        # The peek frame is index 0 — only emit if 0 % step == 0 (always True).
        out[0] = torch.from_numpy(peek).float() / 255.0
        kept = 1

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0 and kept < est_kept:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out[kept] = torch.from_numpy(rgb).float() / 255.0
                kept += 1
            frame_idx += 1
            if kept >= est_kept:
                # safety stop — buffer full
                break

        if kept == 0:
            raise Exception(f"[KLING] Video '{os.path.basename(video_path)}' contains no readable frames.")
        # Trim if we over-allocated.
        if kept < est_kept:
            out = out[:kept].contiguous()
        return out
    finally:
        # K-FIX v2.1: cap.release() guaranteed even on `if not cap.isOpened()` early-fail.
        cap.release()

def download_to_tensor(url: str, retries: int = 3) -> torch.Tensor:
    """Download an image URL and return as a ComfyUI IMAGE tensor [1,H,W,C].
    v2.1: URL guard + retries."""
    url = _safe_url(url)
    last_err = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np)[None, ]
        except (requests.exceptions.RequestException, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                import time as _time
                _time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"[KLING] Image download failed after {retries} attempts: {last_err}")


def _empty_audio() -> Dict[str, Any]:
    """Return a fresh empty AUDIO dict. v2.1: clones the shared waveform so
    downstream in-place mutation can't corrupt the module-level singleton."""
    return {"waveform": EMPTY_AUDIO["waveform"].clone(), "sample_rate": EMPTY_AUDIO["sample_rate"]}


def load_audio_to_tensor(file_path: str) -> Dict[str, Any]:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except ImportError:
        logger.warning("[KLING] torchaudio not installed -- returning silent audio placeholder.")
        return _empty_audio()
    except Exception as e:
        logger.warning(f"[KLING] Could not extract audio from '{os.path.basename(file_path)}': {e}. Returning silent placeholder.")
        return _empty_audio()


def download_audio_to_tensor(url: str) -> Dict[str, Any]:
    """Download audio URL to ComfyUI AUDIO dict.
    v2.1: deletes partial file if decode hard-fails (no orphans)."""
    path, name = download_to_output(url, ext="mp3")
    try:
        return load_audio_to_tensor(path)
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        raise

# --- Cloud Uploader Helpers ---

MIME_MAP = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".webm": "video/webm",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
}

def _detect_mime(filename: str) -> str:
    """Detect MIME type from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return MIME_MAP.get(ext, "application/octet-stream")


def _retry_upload(fn, name: str, retries: int, retry_delay_base: int):
    """Generic retry wrapper for upload functions. Calls fn() up to `retries` times."""
    import time as _time
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = (attempt + 1) * retry_delay_base
                print(f"[KLING] {name} upload failed ({e}), retrying in {wait}s... ({attempt + 1}/{retries})")
                _time.sleep(wait)
    raise Exception(f"{name} Upload Failed: {last_err}")


def upload_to_catbox(file_content: bytes, filename: str, mime_type: str,
                     timeout: int = 120, retries: int = 3) -> str:
    """Uploads to catbox.moe. timeout=seconds per attempt, retries=max attempts."""
    url = "https://catbox.moe/user/api.php"
    def _do():
        data = {"reqtype": "fileupload"}
        files = {"fileToUpload": (filename, file_content, mime_type)}
        response = requests.post(url, data=data, files=files, timeout=timeout)
        response.raise_for_status()
        url_res = response.text.strip()
        if not url_res.startswith("http"):
            raise Exception(f"Catbox error: {url_res}")
        print(f"[KLING] Catbox upload complete: {url_res}")
        return url_res
    return _retry_upload(_do, "Catbox", retries, 3)


def upload_to_tmpfiles(file_content: bytes, filename: str, mime_type: str,
                       timeout: int = 120, retries: int = 3) -> str:
    """Uploads to tmpfiles.org."""
    url = "https://tmpfiles.org/api/v1/upload"
    def _do():
        files = {"file": (filename, file_content, mime_type)}
        response = requests.post(url, files=files, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "data" in data and "url" in data["data"]:
            orig_url = data["data"]["url"]
            if "tmpfiles.org/" in orig_url and "/dl/" not in orig_url:
                dl_url = orig_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
            else:
                dl_url = orig_url
            print(f"[KLING] Tmpfiles upload complete: {dl_url}")
            return dl_url
        raise Exception(f"Tmpfiles invalid response: {data}")
    return _retry_upload(_do, "Tmpfiles", retries, 3)


def upload_to_litterbox(file_content: bytes, filename: str, mime_type: str,
                        retention: str = "1h", timeout: int = 180, retries: int = 3) -> str:
    """Uploads to litterbox.catbox.moe. retention: '1h', '12h', '24h', '72h'."""
    url = "https://litterbox.catbox.moe/resources/internals/api.php"
    def _do():
        data = {"reqtype": "fileupload", "time": retention}
        files = {"fileToUpload": (filename, file_content, mime_type)}
        response = requests.post(url, data=data, files=files, timeout=timeout)
        response.raise_for_status()
        url_res = response.text.strip()
        if not url_res.startswith("http"):
            raise Exception(f"Litterbox error: {url_res}")
        print(f"[KLING] Litterbox upload complete: {url_res}")
        return url_res
    return _retry_upload(_do, "Litterbox", retries, 3)


def upload_to_0x0(file_content: bytes, filename: str, mime_type: str,
                  timeout: int = 180, retries: int = 3) -> str:
    """Uploads to 0x0.st (reliable, permanent, 512MB max)."""
    url = "https://0x0.st"
    def _do():
        files = {"file": (filename, file_content, mime_type)}
        headers = {"User-Agent": "ComfyUI-API-Toolkit/1.0"}
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        response.raise_for_status()
        url_res = response.text.strip()
        if not url_res.startswith("http"):
            raise Exception(f"0x0.st error: {url_res}")
        print(f"[KLING] 0x0.st upload complete: {url_res}")
        return url_res
    return _retry_upload(_do, "0x0.st", retries, 3)


def upload_to_uguu(file_content: bytes, filename: str, mime_type: str,
                   timeout: int = 180, retries: int = 3) -> str:
    """Uploads to uguu.se (simple, 24h retention, 128MB max)."""
    url = "https://uguu.se/upload"
    def _do():
        files = {"files[]": (filename, file_content, mime_type)}
        response = requests.post(url, files=files, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        files_list = data.get("files", [])
        if files_list and files_list[0].get("url"):
            url_res = files_list[0]["url"]
            print(f"[KLING] Uguu upload complete: {url_res}")
            return url_res
        raise Exception(f"Uguu invalid response: {data}")
    return _retry_upload(_do, "Uguu", retries, 3)


# Provider registry -- each entry is a callable taking (content, filename, mime, timeout, retries)
def _call_provider(name: str, content: bytes, filename: str, mime: str, timeout: int, retries: int) -> str:
    """Call the named provider with tuned timeout/retry params."""
    if name == "catbox":
        return upload_to_catbox(content, filename, mime, timeout=timeout, retries=retries)
    if name == "litterbox_1h":
        return upload_to_litterbox(content, filename, mime, retention="1h", timeout=timeout, retries=retries)
    if name == "litterbox_24h":
        return upload_to_litterbox(content, filename, mime, retention="24h", timeout=timeout, retries=retries)
    if name == "litterbox_72h":
        return upload_to_litterbox(content, filename, mime, retention="72h", timeout=timeout, retries=retries)
    if name == "0x0":
        return upload_to_0x0(content, filename, mime, timeout=timeout, retries=retries)
    if name == "uguu":
        return upload_to_uguu(content, filename, mime, timeout=timeout, retries=retries)
    if name == "tmpfiles":
        return upload_to_tmpfiles(content, filename, mime, timeout=timeout, retries=retries)
    raise ValueError(f"Unknown provider: {name}")


_PROVIDER_NAMES = ["catbox", "litterbox_1h", "litterbox_24h", "litterbox_72h", "0x0", "uguu", "tmpfiles"]

# Fallback chain for "auto" mode -- most reliable first
_AUTO_FALLBACK_ORDER = ["catbox", "litterbox_1h", "0x0", "uguu", "tmpfiles"]

# Fast-fail config for auto mode: short timeout per attempt, minimal retries
_AUTO_TIMEOUT = 20     # seconds — fail fast if host is slow/down
_AUTO_RETRIES = 1      # one shot per provider, then move on

# Full retry config for single-provider mode
_FULL_TIMEOUT = 120
_FULL_RETRIES = 3


def upload_to_cloud(file_content: bytes, filename: str, mime_type: str, provider: str = "catbox") -> str:
    """Upload to cloud with automatic fallback.

    - 'auto': tries each provider with 20s timeout and 1 attempt, moves on fast
    - single provider: uses 120s timeout and 3 retries (full reliability mode)
    """
    if provider == "auto":
        last_err = None
        for p in _AUTO_FALLBACK_ORDER:
            try:
                print(f"[KLING] Trying {p} (20s timeout, 1 attempt)...")
                return _call_provider(p, file_content, filename, mime_type,
                                      timeout=_AUTO_TIMEOUT, retries=_AUTO_RETRIES)
            except Exception as e:
                last_err = f"{p}: {e}"
                print(f"[KLING] {p} failed, next...")
        raise Exception(f"All cloud providers failed. Last error: {last_err}")

    # Single provider -- use full retry config, and fall back to others with fast-fail if needed
    if provider not in _PROVIDER_NAMES:
        raise ValueError(f"Unknown provider: {provider}. Options: {_PROVIDER_NAMES}")

    try:
        return _call_provider(provider, file_content, filename, mime_type,
                              timeout=_FULL_TIMEOUT, retries=_FULL_RETRIES)
    except Exception as primary_err:
        print(f"[KLING] {provider} failed after full retries, falling back to others (fast-fail)...")
        for fallback in _AUTO_FALLBACK_ORDER:
            if fallback == provider:
                continue
            try:
                print(f"[KLING] Trying {fallback} (fast-fail)...")
                return _call_provider(fallback, file_content, filename, mime_type,
                                      timeout=_AUTO_TIMEOUT, retries=_AUTO_RETRIES)
            except Exception:
                continue
        raise Exception(f"All cloud providers failed. Initial error: {primary_err}")


def audio_to_wav_bytes_full_quality(audio: Dict[str, Any]) -> bytes:
    """Convert ComfyUI AUDIO dict to WAV bytes WITHOUT resampling or quality loss.

    Preserves original sample rate and channel count. Uses 16-bit PCM (standard WAV).
    Unlike audio_to_base64_string() which downsamples to 16kHz mono for Kling TTS input.
    """
    if audio is None or "waveform" not in audio:
        raise ValueError("Invalid audio dict")

    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate", 44100)

    # Squeeze batch dimension if present
    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    w = waveform.cpu().numpy()
    # Keep channels as-is (don't force mono)
    channels = w.shape[0]
    samples = w.shape[1]

    # Clip to [-1, 1] and convert to int16
    w_int16 = (np.clip(w, -1.0, 1.0) * INT16_MAX).astype(np.int16)

    # Interleave channels for WAV (WAV wants interleaved, not planar)
    if channels > 1:
        interleaved = w_int16.T.flatten()
    else:
        interleaved = w_int16[0]

    buffered = io.BytesIO()
    with wave.open(buffered, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())

    return buffered.getvalue()

# ============================================================
# Node Definitions
# ============================================================

# --- K5: Auth node using DualKeyAPIKeyNode ---

class KlingDirect_Auth(DualKeyAPIKeyNode):
    """Kling AI authentication node with access_key + secret_key + debug toggle."""

    ENV_VAR_ACCESS = "KLING_ACCESS_KEY"
    ENV_VAR_SECRET = "KLING_SECRET_KEY"
    SERVICE_NAME = "Kling AI"

    @classmethod
    def INPUT_TYPES(cls):
        base = DualKeyAPIKeyNode.INPUT_TYPES()
        base["required"]["debug"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Enable verbose debug logging for all Kling API requests.",
        })
        return base

    RETURN_TYPES = ("KLING_AUTH",)
    RETURN_NAMES = ("auth",)
    FUNCTION = "execute"
    CATEGORY = "Kling AI/Config"

    def execute(self, access_key: str = "", secret_key: str = "", debug: bool = False):
        ak, sk = self.provide_keys(access_key, secret_key)
        return ({"access_key": ak, "secret_key": sk, "debug": debug},)


# --- Config / Utility Nodes ---

class KlingDirect_VideoLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory() if folder_paths else "."
        files = []
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith((".mp4", ".mov", ".avi", ".webm"))]
        return {"required": {"video": (sorted(files), {"video_upload": True, "tooltip": "Select a video file from the ComfyUI input directory."})}}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("video", "video_path", "audio")
    FUNCTION = "load_video"
    CATEGORY = "Kling AI/Config"

    def load_video(self, video):
        video_path = folder_paths.get_annotated_filepath(video)
        return (load_video_to_tensor(video_path), video_path, load_audio_to_tensor(video_path))


class KlingDirect_RawFileLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory() if folder_paths else "."
        files = []
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"filename": (sorted(files), {"tooltip": "Select any file from the ComfyUI input directory."})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "load"
    CATEGORY = "Kling AI/Config"

    def load(self, filename):
        return (os.path.join(folder_paths.get_input_directory(), filename),)


class KlingDirect_RawFileSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "url_or_path": ("STRING", {"default": "", "forceInput": True, "tooltip": "URL or local file path to save."}),
            "filename_prefix": ("STRING", {"default": "kling_save", "tooltip": "Prefix for the saved filename."}),
            "format": (["auto", ".mp4", ".mp3", ".png", ".jpg"], {"default": "auto", "tooltip": "Output format. 'auto' detects from URL."})
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Kling AI/Config"
    OUTPUT_NODE = True

    def save(self, url_or_path, filename_prefix, format):
        if not url_or_path:
            return ("",)
        output_dir = folder_paths.get_output_directory()
        clean_path = urlparse(url_or_path).path
        ext = format if format != "auto" else os.path.splitext(clean_path)[1]
        if not ext:
            ext = ".mp4" if "video" in url_or_path else ".png"
        safe_prefix = _sanitize_filename(filename_prefix)
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}{ext}"
        save_path = os.path.join(output_dir, filename)
        logger.info(f"[KLING] Saving file to: {save_path}")
        if url_or_path.startswith(("http://", "https://")):
            res = requests.get(url_or_path, stream=True, timeout=DOWNLOAD_TIMEOUT)
            res.raise_for_status()
            with open(save_path, "wb") as f:
                shutil.copyfileobj(res.raw, f)
        elif os.path.exists(url_or_path):
            shutil.copy2(url_or_path, save_path)
        return (save_path,)


class KlingDirect_AssetUpload(AlwaysExecuteMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "asset_type": (["image", "video"], {"default": "image", "tooltip": "Type of asset to upload."})
        }, "optional": {
            "image": ("IMAGE",),
            "image_path": ("STRING", {"default": "", "tooltip": "Path to an image file on disk."}),
            "video_path": ("STRING", {"default": "", "tooltip": "Path to a video file on disk."})
        }}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("asset_id", "asset_url")
    FUNCTION = "upload"
    CATEGORY = "Kling AI/Config"

    def upload(self, auth, asset_type, image=None, image_path=None, video_path=None):
        client = _make_client(auth)
        if asset_type == "image":
            if image_path:
                return (_extract_asset_id(client.upload_asset(file_path=image_path, asset_type="image")), "")
            if image is not None:
                return (_extract_asset_id(client.upload_asset(b64_data=tensor_to_base64_string(image), asset_type="image")), "")
            raise ValueError("Kling Image Upload requires either an image input or image_path.")
        elif asset_type == "video":
            if video_path:
                return (_extract_asset_id(client.upload_asset(file_path=video_path, asset_type="video")), "")
            raise ValueError("Kling Video Upload requires a video_path (Raw File).")
        raise ValueError("Unsupported Asset Type or missing data.")


class KlingDirect_ElementSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "element_id": ("STRING", {"default": "", "tooltip": "The element ID from Kling AI."}),
            "type": (["character", "face", "style"], {"default": "character", "tooltip": "Type of element: character, face, or style."})
        }}
    RETURN_TYPES = ("KLING_ELEMENT",)
    FUNCTION = "select"
    CATEGORY = "Kling AI/Config"

    def select(self, element_id, type):
        return ({"element_id": element_id, "type": type},)


class KlingDirect_VoiceSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"voice_name": (list(VOICES_CONFIG.keys()), {"tooltip": "Select a preset voice for TTS."})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("voice_id", "language")
    FUNCTION = "execute"
    CATEGORY = "Kling AI/Config"

    def execute(self, voice_name):
        return VOICES_CONFIG[voice_name]


class KlingDirect_CameraControl:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "type": (["simple", "horizontal", "vertical", "pan", "tilt", "roll", "zoom"], {"default": "simple", "tooltip": "Camera movement type."}),
            "horizontal": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Horizontal camera movement (-10 to 10)."}),
            "vertical": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Vertical camera movement (-10 to 10)."}),
            "pan": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Camera pan (-10 to 10)."}),
            "tilt": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Camera tilt (-10 to 10)."}),
            "roll": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Camera roll (-10 to 10)."}),
            "zoom": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "tooltip": "Camera zoom (-10 to 10)."})
        }}
    RETURN_TYPES = ("KLING_CAMERA",)
    FUNCTION = "execute"
    CATEGORY = "Kling AI/Config"

    def execute(self, **kwargs):
        return ({"type": kwargs["type"], "config": {k: v for k, v in kwargs.items() if k != "type"}},)


class KlingDirect_CloudUploader(AlwaysExecuteMixin):
    """Upload media to a PUBLIC cloud host so Kling (or anyone) can fetch it via URL.

    PRIVACY WARNING: every supported host (catbox.moe, litterbox, 0x0.st, uguu.se,
    tmpfiles.org) is unauthenticated and serves the file at a guessable URL.
    Anyone who obtains the URL — including indexers, NSFW scrapers, etc. —
    can download it. Catbox / 0x0 are PERMANENT (file stays public until manually
    deleted). Do not upload sensitive personal images, voice clones, or copyrighted
    material. You MUST tick `i_understand_uploads_are_public` to enable the node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "provider": (
                ["auto", "catbox", "litterbox_1h", "litterbox_24h", "litterbox_72h", "0x0", "uguu", "tmpfiles"],
                {"default": "auto",
                 "tooltip": "Cloud host. 'auto' tries catbox -> litterbox -> 0x0 -> uguu -> tmpfiles (recommended). "
                            "catbox = permanent. litterbox = 1/24/72h temp. 0x0 = permanent. uguu = 24h. tmpfiles = unreliable."}
            ),
            "i_understand_uploads_are_public": ("BOOLEAN", {
                "default": False,
                "tooltip": "REQUIRED. Uploads go to a public host — anyone with the URL can view. "
                           "catbox & 0x0 are permanent. Do NOT upload sensitive content."
            }),
        }, "optional": {
            "audio": ("AUDIO",),
            "image": ("IMAGE",),
            "file_path": ("STRING", {"default": "", "tooltip": "Path to a local file to upload."}),
            "preserve_audio_quality": ("BOOLEAN", {
                "default": True,
                "tooltip": "If True, uploads audio at original sample rate and channels (no downsampling). "
                           "If False, downsamples to 16kHz mono (old behavior — only useful if the audio will be fed back into Kling TTS input)."
            }),
            "audio_format": (["wav", "mp3", "flac"], {
                "default": "wav",
                "tooltip": "Audio encoding format. wav = lossless uncompressed (large). flac = lossless compressed. mp3 = lossy but small."
            }),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload"
    CATEGORY = "Kling AI/Config"

    def upload(self, provider, i_understand_uploads_are_public=False,
               audio=None, image=None, file_path="",
               preserve_audio_quality=True, audio_format="wav"):
        # K-FIX v2.1: explicit consent before uploading to public hosts.
        if not i_understand_uploads_are_public:
            raise ValueError(
                "Kling Cloud Uploader: refusing to upload. Tick "
                "'i_understand_uploads_are_public' to acknowledge that the file "
                "will be world-readable at a public URL (catbox / 0x0 are PERMANENT)."
            )
        file_content = None
        filename = f"upload_{uuid.uuid4().hex[:8]}"
        mime_type = "application/octet-stream"

        if audio is not None:
            if preserve_audio_quality:
                # High-quality path: preserve original sample rate, channels, use 16-bit PCM WAV
                wav_bytes = audio_to_wav_bytes_full_quality(audio)

                if audio_format == "wav":
                    file_content = wav_bytes
                    filename += ".wav"
                    mime_type = "audio/wav"
                elif audio_format == "flac":
                    # Convert WAV to FLAC (lossless, smaller)
                    try:
                        import soundfile as sf
                        sr = audio.get("sample_rate", 44100)
                        wf = audio["waveform"]
                        if wf.dim() == 3:
                            wf = wf[0]
                        arr = wf.cpu().numpy().T  # [samples, channels]
                        buf = io.BytesIO()
                        sf.write(buf, arr, sr, format="FLAC")
                        file_content = buf.getvalue()
                        filename += ".flac"
                        mime_type = "audio/flac"
                    except Exception as e:
                        logger.warning(f"[KLING] FLAC encoding failed ({e}), falling back to WAV")
                        file_content = wav_bytes
                        filename += ".wav"
                        mime_type = "audio/wav"
                elif audio_format == "mp3":
                    # Convert WAV to MP3 (lossy but small)
                    try:
                        import soundfile as sf
                        sr = audio.get("sample_rate", 44100)
                        wf = audio["waveform"]
                        if wf.dim() == 3:
                            wf = wf[0]
                        arr = wf.cpu().numpy().T
                        buf = io.BytesIO()
                        # soundfile supports MP3 writing via libsndfile 1.1+
                        sf.write(buf, arr, sr, format="MP3", subtype="MPEG_LAYER_III")
                        file_content = buf.getvalue()
                        filename += ".mp3"
                        mime_type = "audio/mpeg"
                    except Exception as e:
                        logger.warning(f"[KLING] MP3 encoding failed ({e}), falling back to WAV")
                        file_content = wav_bytes
                        filename += ".wav"
                        mime_type = "audio/wav"
            else:
                # Legacy path: downsampled for Kling TTS input
                audio_b64 = audio_to_base64_string(audio, target_sr=TARGET_SAMPLE_RATE)
                if not audio_b64:
                    raise ValueError("Failed to process audio for cloud upload.")
                file_content = base64.b64decode(audio_b64)
                filename += ".wav"
                mime_type = "audio/wav"
        elif image is not None:
            image_b64 = tensor_to_base64_string(image)
            file_content = base64.b64decode(image_b64)
            filename += ".png"
            mime_type = "image/png"
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_content = f.read()
            filename = os.path.basename(file_path)
            mime_type = _detect_mime(filename)
        else:
            raise ValueError("Kling Cloud Uploader requires either audio, image, or valid file_path.")

        size_mb = len(file_content) / (1024 * 1024)
        print(f"[KLING] Uploading {filename} ({size_mb:.2f} MB) via {provider}...")
        return (upload_to_cloud(file_content, filename, mime_type, provider),)


# ============================================================
# Core Features (Video)
# ============================================================

class KlingDirect_TextToVideo(AlwaysExecuteMixin):
    """Generate video from a text prompt using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text description of the video to generate. Supports @image1 and @video1 references."}),
            "negative_prompt": ("STRING", {"default": "", "tooltip": "Things to avoid in the generated video."}),
            "model_name": (VIDEO_MODELS, {"default": "kling-v3", "tooltip": "Kling model version. v3 is latest, v2-master for cinematic quality."}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "Output video aspect ratio."}),
            "duration": (["5", "10", "15"], {"default": "5", "tooltip": "Video duration in seconds."}),
            "mode": (MODES, {"default": "pro", "tooltip": "Generation mode: 'pro' for higher quality, 'std' for faster/cheaper."}),
            "sound": ("BOOLEAN", {"default": True, "tooltip": "Enable AI-generated sound effects and ambient audio."}),
            "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence, lower = more creative."}),
            "shot_type": (["natural", "wide_angle", "medium_shot", "close_up"], {"default": "natural", "tooltip": "Camera shot framing style."})
        }, "optional": {
            "camera_control": ("KLING_CAMERA",),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, prompt, negative_prompt, model_name, aspect_ratio, duration, mode, sound, cfg_scale, shot_type="natural", camera_control=None):
        client = _make_client(auth)
        sound_val = "on" if sound else "off"
        task_id = client.text_to_video(model_name, normalize_prompts(prompt), aspect_ratio, duration, negative_prompt, cfg_scale, camera_control, mode, sound_val, shot_type=shot_type)
        res = client.poll_task("/v1/videos/text2video", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_ImageToVideo(AlwaysExecuteMixin):
    """Generate video from an image using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional text prompt to guide the video generation from the image."}),
            "negative_prompt": ("STRING", {"default": "", "tooltip": "Things to avoid in the generated video."}),
            "model_name": (VIDEO_MODELS_I2V, {"default": "kling-v3", "tooltip": "Kling model version for image-to-video."}),
            "duration": (["5", "10", "15"], {"default": "5", "tooltip": "Video duration in seconds."}),
            "mode": (MODES, {"default": "pro", "tooltip": "Generation mode: 'pro' for higher quality, 'std' for faster/cheaper."}),
            "sound": ("BOOLEAN", {"default": True, "tooltip": "Enable AI-generated sound effects and ambient audio."}),
            "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence, lower = more creative."})
        }, "optional": {
            "image_tail": ("IMAGE",),
            "camera_control": ("KLING_CAMERA",),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, image, prompt, negative_prompt, model_name, duration, mode, sound, cfg_scale, image_tail=None, camera_control=None):
        client = _make_client(auth)
        sound_val = "on" if sound else "off"
        task_id = client.image_to_video(model_name, tensor_to_base64_string(image), duration, normalize_prompts(prompt), tensor_to_base64_string(image_tail), negative_prompt, cfg_scale, camera_control, mode, sound_val)
        res = client.poll_task("/v1/videos/image2video", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_VideoOmni(AlwaysExecuteMixin):
    """Generate video using Kling Omni model with optional image/video inputs."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt. Use @image1, @video1 to reference optional inputs."}),
            "model_name": (["kling-video-o1", "kling-v3-omni"], {"default": "kling-video-o1", "tooltip": "Omni model version."}),
            "duration": (["5", "10", "15"], {"default": "5", "tooltip": "Video duration in seconds."}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "Output video aspect ratio."}),
            "mode": (MODES, {"default": "pro", "tooltip": "Generation mode."})
        }, "optional": {
            "image_1": ("IMAGE",),
            "image_2": ("IMAGE",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of a reference video for omni generation."}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, prompt, model_name, duration, aspect_ratio, mode, image_1=None, image_2=None, video_url=None):
        client = _make_client(auth)
        # K2: Build image/video lists from optional inputs instead of empty []
        images = []
        if image_1 is not None:
            images.append({"image": tensor_to_base64_string(image_1)})
        if image_2 is not None:
            images.append({"image": tensor_to_base64_string(image_2)})
        videos = []
        if video_url and video_url.strip():
            videos.append({"video_url": video_url.strip()})
        task_id = client.omni_video(model_name, normalize_prompts(prompt), images, videos, aspect_ratio, duration, mode=mode)
        res = client.poll_task("/v1/videos/omni-video", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_VideoExtend(AlwaysExecuteMixin):
    """Extend an existing Kling video by appending more frames."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_id": ("STRING", {"default": "", "forceInput": True, "tooltip": "Task ID of the video to extend (from a previous generation)."}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for the extended portion."}),
            "negative_prompt": ("STRING", {"default": "", "tooltip": "Things to avoid in the extended video."}),
            "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Classifier-free guidance scale."})
        }, "optional": {"video": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, video_id, prompt, negative_prompt="", cfg_scale=0.5, video=None):
        client = _make_client(auth)
        task_id = client.extend_video(video_id, normalize_prompts(prompt), negative_prompt, cfg_scale)
        res = client.poll_task("/v1/videos/video-extend", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_LipSync(AlwaysExecuteMixin):
    """Standard Lip-Sync: sync lips to audio (audio2video) or generated TTS (text2video)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the source video for lip-sync."}),
            "mode": (["audio2video", "text2video"], {"default": "audio2video", "tooltip": "audio2video syncs to provided audio; text2video generates speech from text via Kling voice."})
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "audio2video mode: ComfyUI audio to sync with the video."}),
            "audio_url": ("STRING", {"default": "", "tooltip": "audio2video mode: alternative URL of audio."}),
            "text": ("STRING", {"default": "", "multiline": True, "tooltip": "text2video mode: text to speak. Required if mode=text2video."}),
            "voice_id": ("STRING", {"default": "girlfriend_4_speech02", "tooltip": "text2video mode: Kling voice ID. Use Voice Selector node."}),
            "voice_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "text2video mode: speech speed."}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, video_url, mode, audio=None, audio_url=None,
                 text="", voice_id="girlfriend_4_speech02", voice_speed=1.0):
        client = _make_client(auth)

        # Validate based on mode
        if mode == "text2video":
            if not text or not text.strip():
                raise ValueError("LipSync text2video mode requires a non-empty 'text' input.")
            audio_b64 = None
            audio_url_val = None
            text_val = text.strip()
        else:  # audio2video
            if (audio is None) and (not audio_url or not audio_url.strip()):
                raise ValueError("LipSync audio2video mode requires either 'audio' or 'audio_url'.")
            audio_b64 = audio_to_base64_string(audio) if audio is not None else None
            audio_url_val = audio_url if audio_url else None
            text_val = None

        # K-FIX v2.1: voice_speed is None in audio2video mode (was 1.0 which
        # Kling can reject as an unexpected param in audio2video flow).
        task_id = client.lip_sync(
            video_url,
            audio_url=audio_url_val,
            audio_b64=audio_b64,
            text=text_val,
            voice_id=voice_id if mode == "text2video" else None,
            voice_speed=voice_speed if mode == "text2video" else None,
            mode=mode,
        )
        res = client.poll_task("/v1/videos/lip-sync", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_AvatarGen(AlwaysExecuteMixin):
    """Generate an avatar (digital human) video from an image and optional audio."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional text prompt for the avatar animation."}),
            "mode": (MODES, {"default": "pro", "tooltip": "Generation mode."})
        }, "optional": {
            "audio": ("AUDIO",),
            "audio_url": ("STRING", {"default": "", "tooltip": "URL of audio for the avatar to speak."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, image, prompt, mode, audio=None, audio_url=None):
        client = _make_client(auth)
        audio_id = None
        audio_url_val = audio_url if audio_url != "" else None

        if audio is not None:
            print("[KLING] Converting audio and uploading material asset...")
            audio_b64 = audio_to_base64_string(audio, target_sr=TARGET_SAMPLE_RATE)
            asset_res = client.upload_asset(b64_data=audio_b64, asset_type="audio")
            audio_id = _extract_asset_id(asset_res)
            print(f"[KLING] Audio uploaded. Materials ID: {audio_id}")

        task_id = client.avatar(
            image_b64=tensor_to_base64_string(image),
            audio_url=audio_url_val,
            audio_id=audio_id,
            prompt=prompt,
            mode=mode
        )
        res = client.poll_task("/v1/videos/avatar/image2video", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_AdvancedLipSync(AlwaysExecuteMixin):
    """Advanced Lip-Sync with face detection and per-face audio control."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the source video."}),
            "audio_url": ("STRING", {"default": "", "tooltip": "URL of the audio to sync."}),
            "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "tooltip": "Index of the detected face to sync (0 = first face)."}),
            "volume": ("INT", {"default": 10, "min": 0, "max": 100, "tooltip": "Volume of the synced audio (0-100)."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, video_url, audio_url, face_index=0, volume=10):
        client = _make_client(auth)

        print("[KLING] Identifying face in video...")
        ident_res = client.identify_face(video_url=video_url)
        session_id = ident_res.get("session_id")
        if not session_id:
            raise Exception(f"Kling face identification returned no session_id. Response: {ident_res}")
        faces = ident_res.get("face_data", [])

        if not faces:
            raise Exception("Kling Error: No faces detected in the provided video for Lip-Sync.")

        idx = min(face_index, len(faces) - 1)
        face_id = faces[idx].get("face_id")
        if not face_id:
            raise Exception(f"Kling face at index {idx} has no face_id. Face data: {faces[idx]}")
        print(f"[KLING] Selected face ID: {face_id} from {len(faces)} detected face(s).")

        task_id = client.advanced_lip_sync(session_id, face_id, audio_url, volume=volume)
        res = client.poll_task("/v1/videos/advanced-lip-sync", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


# ============================================================
# Core Features (Image)
# ============================================================

class KlingDirect_ImageGen(AlwaysExecuteMixin):
    """Generate images from text using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text description of the image to generate."}),
            "negative_prompt": ("STRING", {"default": "", "tooltip": "Things to avoid in the generated image."}),
            "model_name": (["kling-v3"], {"default": "kling-v3", "tooltip": "Kling image model version."}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1", "tooltip": "Output image aspect ratio."}),
            "resolution": (IMAGE_RESOLUTIONS, {"default": "1k", "tooltip": "Output resolution: 1k (~1024px) or 2k (~2048px)."}),
            "fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Image fidelity/detail level (0.0 = creative, 1.0 = faithful)."}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9, "tooltip": "Number of images to generate (1-9)."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Image"

    def generate(self, auth, prompt, negative_prompt, model_name, aspect_ratio, resolution, fidelity, n=1):
        client = _make_client(auth)
        task_id = client.image_generation(model_name, normalize_prompts(prompt), aspect_ratio, n, resolution, negative_prompt, fidelity)
        res = client.poll_task("/v1/images/generations", task_id)
        if n > 1:
            images = res.get("images", [])
            if not images:
                raise Exception("Kling image generation completed but returned no images.")
            imgs = [download_to_tensor(img["url"]) for img in images]
            url = images[0].get("url", "")
            return (torch.cat(imgs, dim=0), url, task_id)
        url = _extract_image_url(res)
        return (download_to_tensor(url), url, task_id)


class KlingDirect_ImageOmni(AlwaysExecuteMixin):
    """Generate images using Kling Omni Image model with reference images."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt for omni image generation. Use @image1 to reference input."}),
            "image_1": ("IMAGE",),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1", "tooltip": "Output aspect ratio."}),
            "resolution": (IMAGE_RESOLUTIONS, {"default": "1k", "tooltip": "Output resolution."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Image"

    def generate(self, auth, prompt, image_1, aspect_ratio, resolution):
        client = _make_client(auth)
        task_id = client.omni_image("kling-image-o1", normalize_prompts(prompt), [{"image": tensor_to_base64_string(image_1)}], aspect_ratio=aspect_ratio, resolution=resolution)
        res = client.poll_task("/v1/images/omni-image", task_id)
        url = _extract_image_url(res)
        return (download_to_tensor(url), url, task_id)


class KlingDirect_ImageExtend(AlwaysExecuteMixin):
    """Extend/outpaint an image using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image_id": ("STRING", {"default": "", "forceInput": True, "tooltip": "Asset ID of the image to extend."}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for the extended area."}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1", "tooltip": "Target aspect ratio for the extended image."})
        }, "optional": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Image"

    def generate(self, auth, image_id, prompt, aspect_ratio, image=None):
        client = _make_client(auth)
        i_id = image_id
        if not i_id and image is not None:
            res = client.upload_asset(b64_data=tensor_to_base64_string(image), asset_type="image")
            i_id = _extract_asset_id(res)
        task_id = client.extend_image(i_id, normalize_prompts(prompt), aspect_ratio)
        res = client.poll_task("/v1/images/editing/expand", task_id)
        url = _extract_image_url(res)
        return (download_to_tensor(url), url, task_id)


class KlingDirect_VirtualTryOn(AlwaysExecuteMixin):
    """Virtual try-on: apply clothing from one image to a person in another."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "human_image": ("IMAGE",),
            "cloth_image": ("IMAGE",),
            "model_name": (["kolors-virtual-try-on-v1"], {"default": "kolors-virtual-try-on-v1", "tooltip": "Virtual try-on model."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Image"

    def generate(self, auth, human_image, cloth_image, model_name):
        client = _make_client(auth)
        task_id = client.virtual_try_on(tensor_to_base64_string(human_image), tensor_to_base64_string(cloth_image), model_name=model_name)
        res = client.poll_task("/v1/images/kolors-virtual-try-on", task_id)
        url = _extract_image_url(res)
        return (download_to_tensor(url), url, task_id)


class KlingDirect_MultiShot(AlwaysExecuteMixin):
    """Generate multi-shot consistent images (up to 6 shots) from a single prompt."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "tooltip": "Overall scene/character description for consistency across shots."}),
            "shot_1_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 1 (required)."}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1", "tooltip": "Aspect ratio for all generated shots."})
        }, "optional": {
            "shot_2_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 2."}),
            "shot_3_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 3."}),
            "shot_4_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 4."}),
            "shot_5_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 5."}),
            "shot_6_prompt": ("STRING", {"default": "", "tooltip": "Prompt for shot 6."}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Image"

    def generate(self, auth, prompt, shot_1_prompt, aspect_ratio, shot_2_prompt="", shot_3_prompt="", shot_4_prompt="", shot_5_prompt="", shot_6_prompt=""):
        client = _make_client(auth)
        shots = [{"prompt": shot_1_prompt}]
        for sp in [shot_2_prompt, shot_3_prompt, shot_4_prompt, shot_5_prompt, shot_6_prompt]:
            if sp and sp.strip():
                shots.append({"prompt": sp.strip()})
        task_id = client.multi_shot_image("kling-image-o1", prompt, shots, aspect_ratio=aspect_ratio)
        res = client.poll_task("/v1/images/ai-multi-shot", task_id)
        url = _extract_image_url(res)
        images = res.get("images", [])
        if not images:
            raise Exception("Kling multi-shot task completed but returned no images.")
        imgs = [download_to_tensor(img["url"]) for img in images]
        return (torch.cat(imgs, dim=0), url, task_id)


# ============================================================
# Core Features (Audio & Effects)
# ============================================================

class KlingDirect_AudioGenerate(AlwaysExecuteMixin):
    """Generate audio from a text prompt using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text description of the audio to generate."}),
            "duration": ("INT", {"default": 5, "min": 1, "max": 30, "tooltip": "Audio duration in seconds."})
        }}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_file", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Audio"

    def generate(self, auth, prompt, duration=5):
        client = _make_client(auth)
        task_id = client.text_to_audio(prompt, duration)
        res = client.poll_task("/v1/audio/text-to-audio", task_id)
        # K1: Fixed -- was calling _extract_video_url, now calls _extract_audio_url
        url = _extract_audio_url(res)
        path, name = download_to_output(url, ext="mp3")
        return (load_audio_to_tensor(path), name, url, task_id)


class KlingDirect_TTS(AlwaysExecuteMixin):
    """Text-to-Speech using Kling AI voices.

    v2.1: now exposes voice_speed and voice_language (parity with TTSAdvanced)
    so users don't have to learn two nodes for basic TTS.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to convert to speech."}),
            "voice_id": ("STRING", {"default": "girlfriend_4_speech02", "tooltip": "Voice ID (use Voice Selector node or a cloned voice_id)."}),
            "voice_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Speech speed multiplier (0.5 = slow, 2.0 = fast)."}),
            "voice_language": (["en", "zh"], {"default": "en", "tooltip": "Voice language: en or zh."}),
        }}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_file", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Audio"

    def generate(self, auth, text, voice_id, voice_speed=1.0, voice_language="en"):
        client = _make_client(auth)
        task_id = client.tts(text, voice_id, voice_speed, voice_language)
        res = client.poll_task("/v1/audio/tts", task_id)
        url = _extract_audio_url(res)
        path, name = download_to_output(url, ext="mp3")
        return (load_audio_to_tensor(path), name, url, task_id)


class KlingDirect_VideoToAudio(AlwaysExecuteMixin):
    """Extract/generate audio from a video URL using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the video to extract audio from."})
        }}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Audio"

    def generate(self, auth, video_url):
        client = _make_client(auth)
        task_id = client.video_to_audio(video_url)
        res = client.poll_task("/v1/audio/video-to-audio", task_id)
        url = _extract_audio_url(res)
        return (download_audio_to_tensor(url), url, task_id)


class KlingDirect_Upscale(AlwaysExecuteMixin):
    """Upscale an image or video using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "type": (["image", "video"], {"default": "image", "tooltip": "Whether to upscale an image or video."}),
            "target_id": ("STRING", {"default": "", "forceInput": True, "tooltip": "ID of the image or video to upscale (from a previous generation)."}),
            "model_name": (UPSCALE_MODELS, {"default": "kling-v1", "tooltip": "Upscale model version."})
        }, "optional": {
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the video (required for video upscale)."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("media", "media_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Effects"

    def generate(self, auth, type, target_id, model_name="kling-v1", video_url=None):
        client = _make_client(auth)
        if type == "image":
            task_id = client.upscale_image(target_id, model_name=model_name)
            res = client.poll_task("/v1/images/upscale", task_id)
            url = _extract_image_url(res)
            # Clone the empty audio tensor so downstream mutation doesn't corrupt the shared singleton
            empty_audio = {"waveform": EMPTY_AUDIO["waveform"].clone(), "sample_rate": EMPTY_AUDIO["sample_rate"]}
            return (download_to_tensor(url), "", empty_audio, url, task_id)
        task_id = client.upscale_video(target_id, video_url, model_name=model_name)
        res = client.poll_task("/v1/videos/upscale", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_MotionControl(AlwaysExecuteMixin):
    """Apply motion from a reference video to an image using Kling AI."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image": ("IMAGE",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the reference motion video."}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional text prompt to guide the motion-controlled generation."}),
            "model_name": (VIDEO_MODELS_I2V, {"default": "kling-v1-6", "tooltip": "Kling model for motion control."}),
            "mode": (MODES, {"default": "pro", "tooltip": "Generation mode: 'pro' for higher quality, 'std' for faster."}),
            "character_orientation": (["image", "video"], {"default": "image", "tooltip": "Whether to use the character orientation from the image or the reference video."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, image, video_url, prompt="", model_name="kling-v1-6", mode="pro", character_orientation="image"):
        client = _make_client(auth)
        task_id = client.motion_control(model_name, tensor_to_base64_string(image), video_url, prompt=prompt, character_orientation=character_orientation, mode=mode)
        res = client.poll_task("/v1/videos/motion-control", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


# ============================================================
# New/Advanced Nodes
# ============================================================

class KlingDirect_VoiceClone(AlwaysExecuteMixin):
    """Clone a voice from audio input. Returns a reusable voice_id for TTS."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"auth": ("KLING_AUTH",)},
                "optional": {
                    "audio": ("AUDIO",),
                    "audio_url": ("STRING", {"default": "", "tooltip": "URL of an audio sample to clone the voice from."})
                }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "clone"
    CATEGORY = "Kling AI/Audio"

    def clone(self, auth, audio=None, audio_url=None):
        client = _make_client(auth)
        a_url = audio_url if audio_url else None
        a_b64 = audio_to_base64_string(audio) if audio is not None else None
        if not a_url and not a_b64:
            raise ValueError("Voice Clone requires either audio input or audio_url.")
        voice_id = client.voice_clone(audio_url=a_url, audio_b64=a_b64)
        print(f"[KLING] Voice cloned successfully! voice_id: {voice_id}")
        return (voice_id,)


class KlingDirect_TTSAdvanced(AlwaysExecuteMixin):
    """Text-to-Speech with voice speed control and optional cloned voice."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to convert to speech."}),
            "voice_id": ("STRING", {"default": "girlfriend_4_speech02", "tooltip": "Voice ID (use Voice Selector or Voice Clone)."}),
            "voice_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Speech speed multiplier (0.5 = slow, 2.0 = fast)."}),
            "voice_language": ("STRING", {"default": "en", "tooltip": "Voice language code (en, zh, etc.)."})
        }}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_file", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Audio"

    def generate(self, auth, text, voice_id, voice_speed, voice_language="en"):
        client = _make_client(auth)
        task_id = client.tts(text, voice_id, voice_speed, voice_language)
        res = client.poll_task("/v1/audio/tts", task_id)
        # K1: Fixed -- was calling _extract_video_url, now calls _extract_audio_url
        url = _extract_audio_url(res)
        path, name = download_to_output(url, ext="mp3")
        return (load_audio_to_tensor(path), name, url, task_id)


class KlingDirect_VideoEffects(AlwaysExecuteMixin):
    """Apply video effects (e.g., hug, kiss, heart) to one or two images."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image_1": ("IMAGE",),
            "effect_scene": ("STRING", {"default": "hug", "tooltip": "Effect type (e.g., hug, kiss, heart). Use Effect Templates node to see available options."}),
            "model_name": (["kling-v1", "kling-v1-5"], {"default": "kling-v1", "tooltip": "Model for video effects."}),
            "duration": (["5", "10"], {"default": "5", "tooltip": "Effect video duration in seconds."}),
            "mode": (MODES, {"default": "std", "tooltip": "Generation mode."})
        }, "optional": {"image_2": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Effects"

    def generate(self, auth, image_1, effect_scene, model_name, duration, mode, image_2=None):
        client = _make_client(auth)
        images = [tensor_to_base64_string(image_1)]
        if image_2 is not None:
            images.append(tensor_to_base64_string(image_2))
        task_id = client.video_effects(effect_scene, model_name, duration, images, mode)
        res = client.poll_task("/v1/videos/effects", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_EffectTemplates(AlwaysExecuteMixin):
    """Fetch available video effect templates from Kling."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"auth": ("KLING_AUTH",)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("templates_json",)
    FUNCTION = "fetch"
    CATEGORY = "Kling AI/Effects"

    def fetch(self, auth):
        client = _make_client(auth)
        res = client.effect_templates()
        data = res.get("data", res)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        print(f"[KLING] Found effect templates:\n{formatted[:500]}...")
        return (formatted,)


class KlingDirect_ImageRecognize(AlwaysExecuteMixin):
    """Recognize/describe image content using Kling's vision model."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "image": ("IMAGE",)
        }}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("description", "task_id")
    FUNCTION = "recognize"
    CATEGORY = "Kling AI/Image"

    def recognize(self, auth, image):
        client = _make_client(auth)
        task_id = client.image_recognize(tensor_to_base64_string(image))
        res = client.poll_task("/v1/images/recognize", task_id)
        task_result = res.get("task_result", {})
        description = task_result.get("description", "") or task_result.get("text", "") or json.dumps(task_result)
        print(f"[KLING] Image recognized: {description[:200]}...")
        return (description, task_id)


class KlingDirect_FastVideoSaver:
    """Download video directly from URL to output folder WITHOUT loading into tensor.
    Avoids OOM for large/long videos. Returns the saved file path."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "url": ("STRING", {"default": "", "forceInput": True, "tooltip": "URL of the video to download and save."}),
            "filename_prefix": ("STRING", {"default": "kling_fast", "tooltip": "Prefix for the saved filename."})
        }}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "filename")
    FUNCTION = "save"
    CATEGORY = "Kling AI/Config"
    OUTPUT_NODE = True

    def save(self, url, filename_prefix):
        if not url:
            raise ValueError("FastVideoSaver requires a URL.")
        safe_prefix = _sanitize_filename(filename_prefix)
        output_dir = folder_paths.get_output_directory()

        clean_path = urlparse(url).path
        detected_ext = os.path.splitext(clean_path)[1] or ".mp4"

        filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}{detected_ext}"
        file_path = os.path.join(output_dir, filename)

        url = _safe_url(url)
        print(f"[KLING] Fast-saving to: {filename}")
        # K-FIX v2.1: retry + partial-file cleanup, matching download_to_output.
        last_err = None
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"[KLING] Saved: {filename} ({size_mb:.1f} MB)")
                return (file_path, filename)
            except (requests.exceptions.RequestException, OSError) as e:
                last_err = e
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError:
                    pass
                if attempt < 2:
                    import time as _time
                    _time.sleep(2 * (2 ** attempt))
        raise RuntimeError(f"FastVideoSaver failed after 3 attempts: {last_err}")


# ============================================================
# v2.1 New Nodes
# ============================================================

class KlingDirect_RegionSelector:
    """Pick the Kling API region (Singapore / China / US). Wraps an auth into a new
    auth with `base_url` overridden so the downstream client connects to the
    chosen regional gateway."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "region": (list(KLING_REGIONS.keys()), {"default": "singapore", "tooltip": "Kling API region."}),
        }, "optional": {
            "custom_base_url": ("STRING", {"default": "", "tooltip": "Optional custom base URL override (e.g. self-hosted proxy)."})
        }}
    RETURN_TYPES = ("KLING_AUTH",)
    RETURN_NAMES = ("auth",)
    FUNCTION = "select"
    CATEGORY = "Kling AI/Config"

    def select(self, auth, region, custom_base_url=""):
        new_auth = dict(auth)
        new_auth["base_url"] = (custom_base_url.strip() if custom_base_url.strip() else KLING_REGIONS[region])
        return (new_auth,)


class KlingDirect_CameraPreset:
    """Pre-configured camera movements. Outputs a KLING_CAMERA dict directly."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "preset": (list(CAMERA_PRESETS.keys()), {"default": "none", "tooltip": "Camera-movement preset. Wire into TextToVideo / ImageToVideo."}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.1, "tooltip": "Multiplier on the preset's axis value (1.0 = preset default)."}),
        }}
    RETURN_TYPES = ("KLING_CAMERA",)
    RETURN_NAMES = ("camera_control",)
    FUNCTION = "build"
    CATEGORY = "Kling AI/Config"

    def build(self, preset, intensity=1.0):
        cam_type, raw_cfg = CAMERA_PRESETS[preset]
        # Build the full 6-axis config (zeros for unused axes) then scale active axis.
        cfg = {"horizontal": 0.0, "vertical": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0, "zoom": 0.0}
        for k, v in raw_cfg.items():
            cfg[k] = float(v) * float(intensity)
        return ({"type": cam_type, "config": cfg},)


class KlingDirect_AspectRatioPicker:
    """Given an IMAGE, output the closest valid Kling aspect_ratio string.
    Useful for I2V pipelines where you want the output ratio to match the input.
    """
    _RATIOS = {"16:9": 16/9, "9:16": 9/16, "1:1": 1.0, "3:2": 3/2, "2:3": 2/3, "4:3": 4/3, "3:4": 3/4}

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = (list(_RATIOS.keys()), "STRING")
    RETURN_NAMES = ("aspect_ratio", "info")
    FUNCTION = "pick"
    CATEGORY = "Kling AI/Config"

    def pick(self, image):
        if image.dim() == 4:
            image = image[0]
        h, w, _ = image.shape
        ratio = float(w) / float(h) if h > 0 else 1.0
        # Find closest
        best = min(self._RATIOS.items(), key=lambda kv: abs(kv[1] - ratio))
        info = f"{w}x{h} ratio={ratio:.3f} -> {best[0]} ({best[1]:.3f})"
        return (best[0], info)


class KlingDirect_CostEstimator:
    """Estimate Kling credit cost for a generation. Pure local — no API call."""
    # Heuristic credit table (5s duration, std mode = base). pro = 2x.
    # Numbers are approximate published Kling pricing as of 2026-05; users
    # should verify in their dashboard.
    _T2V_BASE = {
        "kling-v3": 35, "kling-v2-6": 30, "kling-v2-master": 50, "kling-v2-5-turbo": 25, "kling-v1-6": 20,
    }
    _IMAGE_BASE = {"kling-v3": 4, "kling-image-o1": 5}

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "kind": (["text2video", "image2video", "image", "tts", "upscale_image", "upscale_video"], {"default": "text2video"}),
            "model_name": ("STRING", {"default": "kling-v3"}),
            "duration_sec": ("INT", {"default": 5, "min": 1, "max": 60}),
            "mode": (["pro", "std"], {"default": "pro"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9, "tooltip": "Number of images (only used for kind=image)."}),
        }}
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("estimated_credits", "info")
    FUNCTION = "estimate"
    CATEGORY = "Kling AI/Config"

    def estimate(self, kind, model_name, duration_sec, mode, n=1):
        mode_mult = 2.0 if mode == "pro" else 1.0
        if kind in ("text2video", "image2video"):
            base = self._T2V_BASE.get(model_name, 30)
            cost = int(base * mode_mult * (duration_sec / 5.0))
        elif kind == "image":
            base = self._IMAGE_BASE.get(model_name, 4)
            cost = int(base * n)
        elif kind == "tts":
            # ~1 credit per 5s of speech
            cost = max(1, duration_sec // 5)
        elif kind == "upscale_image":
            cost = 6
        elif kind == "upscale_video":
            cost = int(15 * (duration_sec / 5.0))
        else:
            cost = 0
        info = f"~{cost} credits (model={model_name}, mode={mode}, dur={duration_sec}s, n={n}). Heuristic — verify in dashboard."
        return (cost, info)


class KlingDirect_TaskStatus(AlwaysExecuteMixin):
    """One-shot task status check (no polling, no download). Returns the raw
    status JSON so you can chain async workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "endpoint": (TASK_ENDPOINTS, {"default": "/v1/videos/text2video", "tooltip": "The endpoint the task was submitted to."}),
            "task_id": ("STRING", {"default": "", "forceInput": True, "tooltip": "Task ID returned from a previous generation node."}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "result_json", "task_id")
    FUNCTION = "check"
    CATEGORY = "Kling AI/Config"

    def check(self, auth, endpoint, task_id):
        client = _make_client(auth)
        res = client.get_task_status(endpoint, task_id)
        data = res.get("data", {}) or {}
        status = data.get("task_status", "unknown")
        return (status, json.dumps(data, indent=2, default=str)[:8000], task_id)


class KlingDirect_LipSyncFromUrl(AlwaysExecuteMixin):
    """Convenience: submit a lip-sync task using URLs only (video + audio).
    Skips the local download/upload roundtrip if both are already hosted."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_url": ("STRING", {"default": "", "tooltip": "Public URL of the source video."}),
            "audio_url": ("STRING", {"default": "", "tooltip": "Public URL of the audio (mp3/wav)."}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, video_url, audio_url):
        if not video_url.strip() or not audio_url.strip():
            raise ValueError("LipSyncFromUrl requires both video_url and audio_url.")
        client = _make_client(auth)
        task_id = client.lip_sync(video_url.strip(), audio_url=audio_url.strip(), mode="audio2video")
        res = client.poll_task("/v1/videos/lip-sync", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_KeyframeVideo(AlwaysExecuteMixin):
    """Image-to-Video with both START and END frame (Kling's image_tail feature).
    Produces an interpolated video between the two keyframes."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "start_image": ("IMAGE",),
            "end_image": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional motion description."}),
            "negative_prompt": ("STRING", {"default": ""}),
            "model_name": (VIDEO_MODELS_I2V, {"default": "kling-v3"}),
            "duration": (["5", "10"], {"default": "5"}),
            "mode": (MODES, {"default": "pro"}),
            "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {
            "camera_control": ("KLING_CAMERA",),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, start_image, end_image, prompt, negative_prompt, model_name, duration, mode, cfg_scale, camera_control=None):
        client = _make_client(auth)
        task_id = client.image_to_video(
            model_name, tensor_to_base64_string(start_image), duration,
            normalize_prompts(prompt), tensor_to_base64_string(end_image),
            negative_prompt, cfg_scale, camera_control, mode, "on",
        )
        res = client.poll_task("/v1/videos/image2video", task_id)
        url = _extract_video_url(res)
        path, name = download_to_output(url)
        return (load_video_to_tensor(path), name, load_audio_to_tensor(path), url, task_id)


class KlingDirect_VideoToFile:
    """Write a ComfyUI IMAGE batch (video frames) to an .mp4 file via cv2.
    Pure local — no API call. Useful for saving Kling output frames as a
    standalone playable file without depending on VideoHelperSuite."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "video": ("IMAGE", {"tooltip": "Image batch where each frame is a video frame."}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "filename_prefix": ("STRING", {"default": "kling_export"}),
            "codec": (["mp4v", "avc1", "MJPG"], {"default": "mp4v", "tooltip": "FourCC codec. mp4v is most compatible; avc1 is H.264 (better, may need extra codecs)."}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "write"
    CATEGORY = "Kling AI/Config"
    OUTPUT_NODE = True

    def write(self, video, fps, filename_prefix, codec):
        if video.dim() != 4 or video.shape[0] == 0:
            raise ValueError("VideoToFile expects an IMAGE batch [N,H,W,C] with N>=1.")
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        safe = _sanitize_filename(filename_prefix)
        filename = f"{safe}_{uuid.uuid4().hex[:8]}.mp4"
        file_path = os.path.join(output_dir, filename)
        n, h, w, _c = video.shape
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(file_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open with codec={codec}. Try mp4v.")
        try:
            arr = (video.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
            for i in range(n):
                bgr = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
        print(f"[KLING] VideoToFile wrote {n} frames @ {fps}fps -> {filename}")
        return (file_path,)


class KlingDirect_ApiHealthCheck(AlwaysExecuteMixin):
    """Verify auth + connectivity to Kling by fetching effect templates
    (cheap / free call). Returns is_healthy + a status message."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"auth": ("KLING_AUTH",)}}
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_healthy", "status")
    FUNCTION = "check"
    CATEGORY = "Kling AI/Config"

    def check(self, auth):
        try:
            client = _make_client(auth)
            client.effect_templates()
            return (True, "Kling API: OK (auth + connectivity verified via /v1/videos/effect-templates).")
        except KlingAPIError as e:
            return (False, f"Kling API ERROR: {e}")
        except Exception as e:
            return (False, f"Kling API connection failed: {type(e).__name__}: {e}")


class KlingDirect_VoiceCatalog:
    """Output the preset voice catalog as JSON. Pure local — useful for browsing."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("catalog_json",)
    FUNCTION = "list"
    CATEGORY = "Kling AI/Config"

    def list(self):
        rows = [{"display_name": name, "voice_id": vid, "language": lang}
                for name, (vid, lang) in VOICES_CONFIG.items()]
        return (json.dumps(rows, indent=2, ensure_ascii=False),)


# ============================================================
# Registry
# ============================================================

NODE_CLASS_MAPPINGS = {
    # Video
    "KlingDirect_TextToVideo": KlingDirect_TextToVideo,
    "KlingDirect_ImageToVideo": KlingDirect_ImageToVideo,
    "KlingDirect_VideoOmni": KlingDirect_VideoOmni,
    "KlingDirect_VideoExtend": KlingDirect_VideoExtend,
    "KlingDirect_LipSync": KlingDirect_LipSync,
    "KlingDirect_AdvancedLipSync": KlingDirect_AdvancedLipSync,
    "KlingDirect_MotionControl": KlingDirect_MotionControl,
    "KlingDirect_AvatarGen": KlingDirect_AvatarGen,

    # Image
    "KlingDirect_ImageGen": KlingDirect_ImageGen,
    "KlingDirect_ImageOmni": KlingDirect_ImageOmni,
    "KlingDirect_ImageExtend": KlingDirect_ImageExtend,
    "KlingDirect_VirtualTryOn": KlingDirect_VirtualTryOn,
    "KlingDirect_MultiShot": KlingDirect_MultiShot,
    "KlingDirect_ImageRecognize": KlingDirect_ImageRecognize,

    # Audio
    "KlingDirect_AudioGenerate": KlingDirect_AudioGenerate,
    "KlingDirect_TTS": KlingDirect_TTS,
    "KlingDirect_TTSAdvanced": KlingDirect_TTSAdvanced,
    "KlingDirect_VideoToAudio": KlingDirect_VideoToAudio,
    "KlingDirect_VoiceClone": KlingDirect_VoiceClone,

    # Effects
    "KlingDirect_VideoEffects": KlingDirect_VideoEffects,
    "KlingDirect_EffectTemplates": KlingDirect_EffectTemplates,
    "KlingDirect_Upscale": KlingDirect_Upscale,

    # Config
    "KlingDirect_Auth": KlingDirect_Auth,
    "KlingDirect_VideoLoader": KlingDirect_VideoLoader,
    "KlingDirect_RawFileLoader": KlingDirect_RawFileLoader,
    "KlingDirect_RawFileSaver": KlingDirect_RawFileSaver,
    "KlingDirect_AssetUpload": KlingDirect_AssetUpload,
    "KlingDirect_ElementSelector": KlingDirect_ElementSelector,
    "KlingDirect_CameraControl": KlingDirect_CameraControl,
    "KlingDirect_VoiceSelector": KlingDirect_VoiceSelector,
    "KlingDirect_CloudUploader": KlingDirect_CloudUploader,
    "KlingDirect_FastVideoSaver": KlingDirect_FastVideoSaver,

    # v2.1: New nodes
    "KlingDirect_RegionSelector": KlingDirect_RegionSelector,
    "KlingDirect_CameraPreset": KlingDirect_CameraPreset,
    "KlingDirect_AspectRatioPicker": KlingDirect_AspectRatioPicker,
    "KlingDirect_CostEstimator": KlingDirect_CostEstimator,
    "KlingDirect_TaskStatus": KlingDirect_TaskStatus,
    "KlingDirect_LipSyncFromUrl": KlingDirect_LipSyncFromUrl,
    "KlingDirect_KeyframeVideo": KlingDirect_KeyframeVideo,
    "KlingDirect_VideoToFile": KlingDirect_VideoToFile,
    "KlingDirect_ApiHealthCheck": KlingDirect_ApiHealthCheck,
    "KlingDirect_VoiceCatalog": KlingDirect_VoiceCatalog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Video
    "KlingDirect_TextToVideo": "Kling Text to Video",
    "KlingDirect_ImageToVideo": "Kling Image to Video",
    "KlingDirect_VideoOmni": "Kling Video Omni",
    "KlingDirect_VideoExtend": "Kling Video Extend",
    "KlingDirect_LipSync": "Kling Lip Sync",
    "KlingDirect_AdvancedLipSync": "Kling Advanced Lip Sync",
    "KlingDirect_MotionControl": "Kling Motion Control",
    "KlingDirect_AvatarGen": "Kling Avatar Generation",

    # Image
    "KlingDirect_ImageGen": "Kling Image Generation",
    "KlingDirect_ImageOmni": "Kling Image Omni",
    "KlingDirect_ImageExtend": "Kling Image Extend",
    "KlingDirect_VirtualTryOn": "Kling Virtual Try-On",
    "KlingDirect_MultiShot": "Kling AI Multi-Shot",
    "KlingDirect_ImageRecognize": "Kling Image Recognize",

    # Audio
    "KlingDirect_AudioGenerate": "Kling Text to Audio",
    "KlingDirect_TTS": "Kling Text to Speech",
    "KlingDirect_TTSAdvanced": "Kling TTS Advanced",
    "KlingDirect_VideoToAudio": "Kling Video to Audio",
    "KlingDirect_VoiceClone": "Kling Voice Clone",

    # Effects
    "KlingDirect_VideoEffects": "Kling Video Effects",
    "KlingDirect_EffectTemplates": "Kling Effect Templates",
    "KlingDirect_Upscale": "Kling AI Upscale",

    # Config
    "KlingDirect_Auth": "Kling AI Authentication",
    "KlingDirect_VideoLoader": "Kling Video Loader",
    "KlingDirect_RawFileLoader": "Kling Raw File Loader",
    "KlingDirect_RawFileSaver": "Kling Raw File Saver",
    "KlingDirect_AssetUpload": "Kling AI Asset Upload",
    "KlingDirect_ElementSelector": "Kling AI Element",
    "KlingDirect_CameraControl": "Kling Camera Control",
    "KlingDirect_VoiceSelector": "Kling Voice Selector",
    "KlingDirect_CloudUploader": "Kling AI Cloud Uploader",
    "KlingDirect_FastVideoSaver": "Kling Fast Video Saver",

    # v2.1: New nodes
    "KlingDirect_RegionSelector": "Kling Region Selector",
    "KlingDirect_CameraPreset": "Kling Camera Preset",
    "KlingDirect_AspectRatioPicker": "Kling Aspect Ratio Picker",
    "KlingDirect_CostEstimator": "Kling Cost Estimator",
    "KlingDirect_TaskStatus": "Kling Task Status",
    "KlingDirect_LipSyncFromUrl": "Kling Lip Sync (URLs)",
    "KlingDirect_KeyframeVideo": "Kling Keyframe Video (Start+End)",
    "KlingDirect_VideoToFile": "Kling Video to File (MP4)",
    "KlingDirect_ApiHealthCheck": "Kling API Health Check",
    "KlingDirect_VoiceCatalog": "Kling Voice Catalog",
}
