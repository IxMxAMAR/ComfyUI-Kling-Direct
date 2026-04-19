"""Kling AI nodes for ComfyUI-AI-Suite.

Ported from ComfyUI-Kling-Direct with bug fixes, new features, and UI improvements.
"""

import json
import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
import uuid
import re
import wave
import logging
import cv2
import shutil
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional

try:
    import folder_paths
except ImportError:
    folder_paths = None

from .kling_client import KlingClient, get_client

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

VIDEO_MODELS = ["kling-v3", "kling-v2-1", "kling-v2-5-turbo", "kling-v2-6", "kling-v2-master", "kling-v1-6"]
VIDEO_MODELS_I2V = ["kling-v3", "kling-v2-1", "kling-v2-6", "kling-v2-master", "kling-v1-6"]
UPSCALE_MODELS = ["kling-v1", "kling-v2-1", "kling-v3"]


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


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in Windows filenames."""
    return SAFE_FILENAME_RE.sub("_", name)


def _make_client(auth: dict) -> KlingClient:
    """Create or retrieve a cached KlingClient from an auth dict."""
    return get_client(auth["access_key"], auth["secret_key"], debug=auth.get("debug", False))


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

def tensor_to_base64_string(image: torch.Tensor) -> str:
    if image is None:
        return None
    if image.dim() == 4:
        image = image[0]
    h, w, _ = image.shape
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        logger.warning(f"Kling Warning: Image resolution ({w}x{h}) is below recommended {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}.")
    i = (255. * image).cpu().numpy().astype(np.uint8)
    img = Image.fromarray(i)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def audio_to_base64_string(audio: Dict[str, Any], target_sr: int = TARGET_SAMPLE_RATE) -> str:
    if audio is None or "waveform" not in audio:
        return None
    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate", 44100)

    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    w = waveform.cpu().numpy()

    duration = w.shape[-1] / sample_rate
    if duration < MIN_AUDIO_DURATION:
        logger.warning(f"Kling Warning: Audio duration ({duration:.2f}s) is too short. Looping to reach {MIN_AUDIO_DURATION}s minimum.")
        repeats = int(np.ceil(MIN_AUDIO_DURATION / duration))
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

    # Dynamic Resampling
    if sample_rate != target_sr:
        try:
            from scipy.interpolate import interp1d
            times = np.arange(w.shape[-1]) / sample_rate
            new_times = np.arange(int(w.shape[-1] * target_sr / sample_rate)) / target_sr
            new_w = []
            for channel in range(w.shape[0]):
                f = interp1d(times, w[channel], kind='linear', fill_value="extrapolate")
                new_w.append(f(new_times))
            w = np.array(new_w)
            sample_rate = target_sr
        except ImportError:
            new_samples = int(w.shape[-1] * target_sr / sample_rate)
            indices = np.linspace(0, w.shape[-1] - 1, new_samples).astype(np.int64)
            w = w[:, indices]
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


def download_to_output(url: str, ext: str = "mp4") -> tuple:
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)

    clean_path = urlparse(url).path
    detected_ext = os.path.splitext(clean_path)[1]
    filename_ext = ext if not detected_ext else detected_ext.lstrip('.')

    filename = f"kling_{uuid.uuid4().hex}.{filename_ext}"
    file_path = os.path.join(output_dir, filename)

    # Print URL first so it's recoverable if download fails
    print(f"[KLING] Download URL: {url}")

    response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
    print(f"[KLING] Saved to output: {filename}")
    return file_path, filename

def load_video_to_tensor(video_path: str) -> torch.Tensor:
    """Load video frames into a tensor with automatic subsampling for large videos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"[KLING] Could not open video file: {os.path.basename(video_path)}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    step = 1
    if total_frames > MAX_VIDEO_FRAMES:
        step = max(1, total_frames // MAX_VIDEO_FRAMES)
        logger.info(f"[KLING] Video has {total_frames} frames -- subsampling every {step}th frame (keeping ~{total_frames // step} frames) to avoid OOM.")

    frames = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_idx += 1
        cap.release()

        if not frames:
            raise Exception(f"[KLING] Video '{os.path.basename(video_path)}' contains no readable frames.")

        try:
            stack = np.stack(frames, axis=0)
            return torch.from_numpy(stack).float() / 255.0
        except (MemoryError, RuntimeError) as mem_err:
            logger.warning(f"[KLING] MEMORY ERROR: Video at {video_path} too large ({len(frames)} frames). Error: {mem_err}")
            logger.warning("[KLING] Returning single-frame placeholder. Full video file is saved in output folder.")
            first = frames[0] if frames else cv2.cvtColor(cv2.imread(video_path), cv2.COLOR_BGR2RGB)
            return torch.from_numpy(np.array(first)[None]).float() / 255.0

    except Exception as e:
        cap.release()
        if "MEMORY ERROR" in str(e) or isinstance(e, MemoryError):
            raise
        logger.error(f"[KLING] Error loading video '{video_path}': {e}")
        raise Exception(f"[KLING] Failed to load video '{os.path.basename(video_path)}': {e}")

def download_to_tensor(url: str) -> torch.Tensor:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None,]

def load_audio_to_tensor(file_path: str) -> Dict[str, Any]:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except ImportError:
        logger.warning("[KLING] torchaudio not installed -- returning silent audio placeholder.")
        return dict(EMPTY_AUDIO)
    except Exception as e:
        logger.warning(f"[KLING] Could not extract audio from '{os.path.basename(file_path)}': {e}. Returning silent placeholder.")
        return dict(EMPTY_AUDIO)

def download_audio_to_tensor(url: str) -> Dict[str, Any]:
    path, name = download_to_output(url, ext="mp3")
    return load_audio_to_tensor(path)

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


def upload_to_catbox(file_content: bytes, filename: str, mime_type: str) -> str:
    """Uploads to catbox.moe with retry."""
    import time as _time
    url = "https://catbox.moe/user/api.php"
    for attempt in range(3):
        try:
            data = {"reqtype": "fileupload"}
            files = {"fileToUpload": (filename, file_content, mime_type)}
            response = requests.post(url, data=data, files=files, timeout=120)
            response.raise_for_status()
            url_res = response.text.strip()
            if not url_res.startswith("http"):
                raise Exception(f"Catbox error: {url_res}")
            print(f"[KLING] Catbox upload complete: {url_res}")
            return url_res
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"[KLING] Catbox upload failed ({e}), retrying in {wait}s... ({attempt + 1}/3)")
                _time.sleep(wait)
            else:
                logger.error(f"Catbox Upload Failed after 3 attempts: {e}")
                raise Exception(f"Catbox Upload Failed: {e}")

def upload_to_tmpfiles(file_content: bytes, filename: str, mime_type: str) -> str:
    """Uploads to tmpfiles.org with retry."""
    import time as _time
    url = "https://tmpfiles.org/api/v1/upload"
    for attempt in range(3):
        try:
            files = {"file": (filename, file_content, mime_type)}
            response = requests.post(url, files=files, timeout=120)
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
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"[KLING] Tmpfiles upload failed ({e}), retrying in {wait}s... ({attempt + 1}/3)")
                _time.sleep(wait)
            else:
                logger.error(f"Tmpfiles Upload Failed after 3 attempts: {e}")
                raise Exception(f"Tmpfiles Upload Failed: {e}")


def upload_to_litterbox(file_content: bytes, filename: str, mime_type: str, retention: str = "1h") -> str:
    """Uploads to litterbox.catbox.moe (catbox's temporary bucket, more reliable than tmpfiles).
    retention: '1h', '12h', '24h', '72h'
    """
    import time as _time
    url = "https://litterbox.catbox.moe/resources/internals/api.php"
    for attempt in range(3):
        try:
            data = {"reqtype": "fileupload", "time": retention}
            files = {"fileToUpload": (filename, file_content, mime_type)}
            response = requests.post(url, data=data, files=files, timeout=180)
            response.raise_for_status()
            url_res = response.text.strip()
            if not url_res.startswith("http"):
                raise Exception(f"Litterbox error: {url_res}")
            print(f"[KLING] Litterbox upload complete: {url_res}")
            return url_res
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"[KLING] Litterbox upload failed ({e}), retrying in {wait}s... ({attempt + 1}/3)")
                _time.sleep(wait)
            else:
                raise Exception(f"Litterbox Upload Failed: {e}")


def upload_to_0x0(file_content: bytes, filename: str, mime_type: str) -> str:
    """Uploads to 0x0.st (reliable, permanent, no account, 512MB max)."""
    import time as _time
    url = "https://0x0.st"
    for attempt in range(3):
        try:
            files = {"file": (filename, file_content, mime_type)}
            # 0x0.st requires a User-Agent
            headers = {"User-Agent": "ComfyUI-API-Toolkit/1.0"}
            response = requests.post(url, files=files, headers=headers, timeout=180)
            response.raise_for_status()
            url_res = response.text.strip()
            if not url_res.startswith("http"):
                raise Exception(f"0x0.st error: {url_res}")
            print(f"[KLING] 0x0.st upload complete: {url_res}")
            return url_res
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"[KLING] 0x0.st upload failed ({e}), retrying in {wait}s... ({attempt + 1}/3)")
                _time.sleep(wait)
            else:
                raise Exception(f"0x0.st Upload Failed: {e}")


def upload_to_uguu(file_content: bytes, filename: str, mime_type: str) -> str:
    """Uploads to uguu.se (simple, reliable, 24h retention, 128MB max)."""
    import time as _time
    url = "https://uguu.se/upload"
    for attempt in range(3):
        try:
            files = {"files[]": (filename, file_content, mime_type)}
            response = requests.post(url, files=files, timeout=180)
            response.raise_for_status()
            data = response.json()
            files_list = data.get("files", [])
            if files_list and files_list[0].get("url"):
                url_res = files_list[0]["url"]
                print(f"[KLING] Uguu upload complete: {url_res}")
                return url_res
            raise Exception(f"Uguu invalid response: {data}")
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"[KLING] Uguu upload failed ({e}), retrying in {wait}s... ({attempt + 1}/3)")
                _time.sleep(wait)
            else:
                raise Exception(f"Uguu Upload Failed: {e}")


# Provider registry -- order matters for the "auto" fallback chain
_UPLOAD_PROVIDERS = {
    "catbox": upload_to_catbox,
    "litterbox_1h": lambda c, f, m: upload_to_litterbox(c, f, m, "1h"),
    "litterbox_24h": lambda c, f, m: upload_to_litterbox(c, f, m, "24h"),
    "litterbox_72h": lambda c, f, m: upload_to_litterbox(c, f, m, "72h"),
    "0x0": upload_to_0x0,
    "uguu": upload_to_uguu,
    "tmpfiles": upload_to_tmpfiles,
}

# Fallback chain for "auto" mode -- most reliable first
_AUTO_FALLBACK_ORDER = ["catbox", "litterbox_1h", "0x0", "uguu", "tmpfiles"]


def upload_to_cloud(file_content: bytes, filename: str, mime_type: str, provider: str = "catbox") -> str:
    """Upload to cloud with automatic fallback.

    provider: one of the keys in _UPLOAD_PROVIDERS, or 'auto' to try the fallback chain.
    """
    if provider == "auto":
        last_err = None
        for p in _AUTO_FALLBACK_ORDER:
            try:
                return _UPLOAD_PROVIDERS[p](file_content, filename, mime_type)
            except Exception as e:
                last_err = f"{p}: {e}"
                print(f"[KLING] {p} failed, trying next...")
        raise Exception(f"All cloud providers failed. Last error: {last_err}")

    # Single provider with fallback to the rest if it fails
    if provider not in _UPLOAD_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(_UPLOAD_PROVIDERS.keys())}")

    try:
        return _UPLOAD_PROVIDERS[provider](file_content, filename, mime_type)
    except Exception as primary_err:
        # Try the other providers as fallback
        for fallback in _AUTO_FALLBACK_ORDER:
            if fallback == provider:
                continue
            try:
                print(f"[KLING] {provider} failed, trying {fallback}...")
                return _UPLOAD_PROVIDERS[fallback](file_content, filename, mime_type)
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
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "provider": (
                ["auto", "catbox", "litterbox_1h", "litterbox_24h", "litterbox_72h", "0x0", "uguu", "tmpfiles"],
                {"default": "auto",
                 "tooltip": "Cloud host. 'auto' tries catbox -> litterbox -> 0x0 -> uguu -> tmpfiles (recommended). "
                            "catbox = permanent. litterbox = 1/24/72h temp. 0x0 = permanent. uguu = 24h. tmpfiles = unreliable."}
            ),
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

    def upload(self, provider, audio=None, image=None, file_path="",
               preserve_audio_quality=True, audio_format="wav"):
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
    """Standard Lip-Sync: sync lips to audio or text."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "video_url": ("STRING", {"default": "", "tooltip": "URL of the source video for lip-sync."}),
            "mode": (["audio2video", "text2video"], {"default": "audio2video", "tooltip": "audio2video syncs to audio; text2video generates speech."})
        }, "optional": {
            "audio": ("AUDIO",),
            "audio_url": ("STRING", {"default": "", "tooltip": "URL of audio to sync with the video."})
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "audio", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Video"

    def generate(self, auth, video_url, mode, audio=None, audio_url=None):
        client = _make_client(auth)
        audio_b64 = audio_to_base64_string(audio) if audio is not None else None
        task_id = client.lip_sync(video_url, audio_url=audio_url, audio_b64=audio_b64, mode=mode)
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
            "model_name": (["kling-v3", "kling-v2-1"], {"default": "kling-v3", "tooltip": "Kling image model version."}),
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
    """Text-to-Speech using Kling AI voices."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "text": ("STRING", {"default": "", "tooltip": "Text to convert to speech."}),
            "voice_id": ("STRING", {"default": "female_1", "tooltip": "Voice ID (use Voice Selector node or a cloned voice_id)."})
        }}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_file", "url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Kling AI/Audio"

    def generate(self, auth, text, voice_id):
        client = _make_client(auth)
        task_id = client.tts(text, voice_id, 1.0, "en")
        res = client.poll_task("/v1/audio/tts", task_id)
        # K1: Fixed -- was calling _extract_video_url, now calls _extract_audio_url
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
            return (download_to_tensor(url), "", dict(EMPTY_AUDIO), url, task_id)
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

        print(f"[KLING] Fast-saving to: {filename}")
        response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[KLING] Saved: {filename} ({size_mb:.1f} MB)")
        return (file_path, filename)


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
}
