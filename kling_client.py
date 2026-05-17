"""Kling AI API client with JWT auth, retry logic, and task polling.

v2.1.0 hardening (2026-05-17):
- JWT secret never leaks via exception tracebacks (wrapped in KlingAPIError).
- _strip_none now recurses into lists of dicts (Kling's image_list/video_list).
- task_id is URL-quoted before path interpolation (no path traversal).
- poll_task uses time.monotonic() (NTP-safe) and chunked sleeps so ComfyUI
  cancellation is respected within ~1s.
- 1106 (Task Failed) moved to permanent errors -- retrying it is pointless.
- Retry-After honors HTTP-date format and 300s cap (was 60s).
- LRU-bounded client cache with thread lock + __del__ closing the session.
- JWT nbf widened to -60s for clock-skew tolerance.
- Polling backoff extended for long tasks (15s after 2min, 30s after 5min)
  with jitter.
- mimetypes used for upload_asset (now supports jpg/jpeg/mp3/m4a/etc).
- Token cached on instance for ~25 min (saves HMAC cycles + reduces churn).
"""

import hmac
import hashlib
import mimetypes
import random
import threading
import time
import base64
import json
import requests
import logging
import os
import io
from collections import OrderedDict
from email.utils import parsedate_to_datetime
from typing import Optional, Dict, Any, List
from urllib.parse import quote as _url_quote

logger = logging.getLogger(__name__)

# --- Client Cache (bounded LRU + thread-safe) ---
_CLIENT_CACHE_MAX = 8
_client_cache: "OrderedDict[tuple, KlingClient]" = OrderedDict()
_client_cache_lock = threading.Lock()


def get_client(access_key: str, secret_key: str, debug: bool = False,
               base_url: str = "https://api-singapore.klingai.com") -> "KlingClient":
    """Return a cached KlingClient for the given credentials, creating one if needed.

    Thread-safe (ComfyUI may invoke nodes concurrently). LRU-bounded to prevent
    unbounded growth when a workflow cycles through many credentials.
    """
    cache_key = (access_key, secret_key, base_url)
    with _client_cache_lock:
        client = _client_cache.get(cache_key)
        if client is None:
            client = KlingClient(access_key, secret_key, base_url=base_url, debug=debug)
            _client_cache[cache_key] = client
            # Evict oldest if over capacity. Close session to release sockets.
            while len(_client_cache) > _CLIENT_CACHE_MAX:
                _, old = _client_cache.popitem(last=False)
                try:
                    old.close()
                except Exception:
                    pass
        else:
            # Move to MRU position
            _client_cache.move_to_end(cache_key)
        client.debug = debug
    return client


# --- Custom error type ---

class KlingAPIError(RuntimeError):
    """Kling API error. NEVER carries the original `requests` exception object
    (which holds the Authorization header with the signed JWT). Tracebacks
    from this exception are safe to share publicly."""

    def __init__(self, message: str, code: Optional[int] = None,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


# --- Kling Error Code Mapping ---
KLING_ERROR_MAP = {
    # HTTP-level
    401: "Unauthorized: Invalid AccessKey or SecretKey. Double-check your Kling AI Authentication node.",
    403: "Forbidden: You may have run out of credits or don't have permission for this model.",
    429: "Too Many Requests: Kling API rate limit hit. Slow down or check your balance/tier.",
    # Auth-level (1000 range)
    1000: "Invalid Parameter: One of your inputs (prompt length, image size, etc.) is invalid.",
    1001: "Invalid Token: Authentication token is bad. Regenerate your Kling access/secret keys.",
    1002: "Invalid API Key: Your AccessKey is not recognized. Check your Kling dev console.",
    1003: "Authorization Not Active: Your Kling account hasn't been activated for API access yet. "
          "Go to https://app.klingai.com/global/dev and complete API activation/KYC. "
          "New accounts may need approval before API calls work.",
    1004: "Authorization Expired: Your API access has expired. Renew it in your Kling dev console.",
    # Content/resource-level (1100 range)
    1100: "Invalid Video Duration: Requested duration is not supported for this model.",
    1101: "Invalid Image: Image is too large, has an unsupported aspect ratio, or failed content check.",
    1102: "Account Balance Not Enough: Top up your Kling credits at https://app.klingai.com/global/",
    1103: "Account Frozen: Your Kling account has been suspended. Contact Kling support.",
    1104: "Resource Exhausted: You've hit a resource cap (concurrent tasks, daily limit, etc.).",
    1105: "Task Not Found: The task_id you're polling doesn't exist or has expired (72 hours).",
    1106: "Task Failed: The generation task failed on Kling's side. Try a different prompt/input.",
    1107: "Invalid Audio: The provided audio file is too long, wrong format, or unreadable.",
    1108: "Invalid Video: The provided video is too large, too long, or wrong format.",
    # Server-level (1200 range)
    1200: "Server Busy: Kling's systems are overloaded. Try again in ~60 seconds.",
    1201: "Internal Error: Kling server hiccup. Your task may still complete in the queue.",
    1202: "Gateway Timeout: Kling took too long to respond. Try again shortly.",
    # Content policy (1300 range)
    1301: "IP Banned: Your IP has been blocked. Contact Kling if this is unexpected.",
    1302: "Content Policy Violation: Your prompt or image was flagged by Kling's safety filter.",
    1303: "Copyright Violation: Detected content is protected/copyrighted.",
}

# Non-retryable Kling error codes -- retrying won't help.
# K-FIX v2.1: 1106 (Task Failed) moved here -- retrying just re-queries a known-failed task.
_PERMANENT_ERROR_CODES = {
    401, 403,
    1000, 1001, 1002, 1003, 1004,
    1100, 1101, 1102, 1103, 1105, 1106, 1107, 1108,
    1301, 1302, 1303,
}
# Retryable Kling error codes -- transient server issues.
_TRANSIENT_ERROR_CODES = {1104, 1200, 1201, 1202}

REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 120

# JWT settings
_JWT_TTL_SECONDS = 1800           # 30 min token lifetime (per Kling docs)
_JWT_NBF_SKEW = 60                # widened from 5s to 60s for clock-skew tolerance
_JWT_REUSE_SECONDS = 1500         # reuse cached token until 5 min before exp


def _strip_none(d):
    """Recursively remove keys with None values from a dict / dicts inside lists,
    so Kling's strict JSON parser doesn't reject the payload. v2.1 now recurses
    into list elements (Kling's image_list / video_list / shots / multi_prompt)."""
    if d is None:
        return None
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, dict):
                out[k] = _strip_none(v)
            elif isinstance(v, list):
                out[k] = [_strip_none(i) if isinstance(i, (dict, list)) else i for i in v if i is not None]
            else:
                out[k] = v
        return out
    if isinstance(d, list):
        return [_strip_none(i) if isinstance(i, (dict, list)) else i for i in d if i is not None]
    return d


def _parse_retry_after(value: str, fallback: float) -> float:
    """Parse a Retry-After header. May be int seconds OR RFC 7231 HTTP-date.
    Returns seconds to wait. Capped at 300s (was 60s pre-2.1)."""
    if not value:
        return fallback
    try:
        return min(float(value), 300.0)
    except (ValueError, TypeError):
        pass
    # HTTP-date format
    try:
        target = parsedate_to_datetime(value)
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc) if target.tzinfo else _dt.datetime.utcnow()
        delta = (target - now).total_seconds()
        return min(max(delta, 0.0), 300.0)
    except Exception:
        return fallback


class KlingClient:
    def __init__(self, access_key: str, secret_key: str,
                 base_url: str = "https://api-singapore.klingai.com",
                 debug: bool = False):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.debug = debug
        # Persistent session for connection pooling -- reuses TCP connections across polling loops
        self._session = requests.Session()
        # Token cache (refresh well before exp)
        self._cached_token = None
        self._cached_token_exp = 0
        self._token_lock = threading.Lock()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        try:
            self._session.close()
        except Exception:
            pass

    def __del__(self):
        # Best-effort cleanup; may run during interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass

    def _generate_token(self) -> str:
        """Generates an HS256 JWT token for authentication.

        v2.1: Cached on the instance for ~25 min so we don't re-HMAC on every
        poll. Cleared whenever the cache is older than _JWT_REUSE_SECONDS.
        """
        now = int(time.time())
        with self._token_lock:
            if self._cached_token and now < self._cached_token_exp - 300:
                return self._cached_token

            def b64_encode(d: Dict[str, Any]) -> str:
                return base64.urlsafe_b64encode(
                    json.dumps(d, separators=(',', ':')).encode('utf-8')
                ).decode('utf-8').rstrip('=')

            header_enc = b64_encode({"alg": "HS256", "typ": "JWT"})
            exp = now + _JWT_TTL_SECONDS
            payload_enc = b64_encode({
                "iss": self.access_key,
                "exp": exp,
                "nbf": now - _JWT_NBF_SKEW,
            })

            msg = f"{header_enc}.{payload_enc}".encode('utf-8')
            signature = hmac.new(
                self.secret_key.encode('utf-8'), msg, hashlib.sha256
            ).digest()

            token = f"{header_enc}.{payload_enc}.{base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')}"
            self._cached_token = token
            self._cached_token_exp = exp
            return token

    def _get_headers(self) -> Dict[str, str]:
        """Base headers for all requests."""
        return {
            "Authorization": f"Bearer {self._generate_token()}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None,
                 retries: int = 3) -> Dict[str, Any]:
        """Performs an authenticated HTTP request with retry, error parsing, and
        SECRET-SAFE exception handling.

        v2.1 SECURITY: Network exceptions are caught and re-raised as
        KlingAPIError so the original `requests.exceptions.RequestException`
        (which embeds the `Request` object including the Authorization header
        with the signed JWT) never reaches the user's traceback log.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = self._get_headers()

        # Strip None values from payload to keep requests clean (now recurses into lists).
        clean_data = _strip_none(data) if data else data

        if self.debug and clean_data:
            log_data = clean_data.copy() if isinstance(clean_data, dict) else clean_data
            # Mask any large strings (b64 images/audio etc) -- not just known keys.
            if isinstance(log_data, dict):
                for key, val in list(log_data.items()):
                    if isinstance(val, str) and len(val) > 100:
                        log_data[key] = f"{val[:20]}... [{len(val)} chars]"
            print(f"[KLING DEBUG] Request to {url}:\n{json.dumps(log_data, indent=2, default=str)}")

        for attempt in range(retries):
            try:
                if method.upper() == "POST":
                    response = self._session.post(url, headers=headers, json=clean_data, timeout=REQUEST_TIMEOUT)
                else:
                    response = self._session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

                # Attempt to parse JSON even if status is not 200 (Kling puts errors in body)
                try:
                    res_json = response.json()
                except (ValueError, Exception) as parse_err:
                    logger.debug(f"[KLING] Could not parse JSON response: {parse_err}")
                    res_json = {}

                if self.debug:
                    print(f"[KLING DEBUG] Response Status: {response.status_code}")
                    if res_json:
                        print(f"[KLING DEBUG] Body: {json.dumps(res_json, indent=2)}")
                    else:
                        print(f"[KLING DEBUG] Body: {response.text[:200]}")

                # Check for Kling-specific error codes in the body FIRST
                code = res_json.get("code")
                if code is not None and code != 0:
                    msg = res_json.get("message", "Unknown Error")
                    hint = KLING_ERROR_MAP.get(code, "Check your Kling AI account/quota.")

                    # Permanent errors -- retrying won't help
                    if code in _PERMANENT_ERROR_CODES:
                        raise KlingAPIError(f"Kling API Error {code}: {msg}\n[Hint]: {hint}",
                                            code=code, status_code=response.status_code)

                    # Transient errors -- retry
                    if code in _TRANSIENT_ERROR_CODES and attempt < retries - 1:
                        wait = 60 if code == 1200 else (attempt + 1) * 5
                        print(f"[KLING] Transient error {code}. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(wait)
                        continue

                    raise KlingAPIError(f"Kling API Error {code}: {msg}\n[Hint]: {hint}",
                                        code=code, status_code=response.status_code)

                # Handle standard HTTP errors (rate limits, server issues)
                if response.status_code in (429, 502, 503, 504):
                    if attempt < retries - 1:
                        retry_after = response.headers.get("Retry-After")
                        wait = _parse_retry_after(retry_after, (attempt + 1) * 5)
                        print(f"[KLING] Status {response.status_code}. Retrying in {wait:.0f}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(wait)
                        continue

                # raise_for_status would attach the request object; catch and rewrap to
                # avoid leaking the Authorization header in tracebacks.
                if response.status_code >= 400:
                    raise KlingAPIError(
                        f"Kling HTTP {response.status_code} at {path}: "
                        f"{(response.text or '')[:200]}",
                        status_code=response.status_code,
                    )
                return res_json

            except KlingAPIError:
                raise
            except requests.exceptions.RequestException as e:
                # SECURITY: do NOT re-raise the original exception. It carries
                # `e.request.headers["Authorization"]` which contains the signed
                # JWT. Stringify the message only.
                err_msg = type(e).__name__
                if attempt < retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"[KLING] Connection error: {err_msg}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"Kling Request Failed after {retries} attempts: {err_msg}")
                # raise from None to detach the original cause (and its headers)
                raise KlingAPIError(
                    f"Kling network error after {retries} attempts: {err_msg}"
                ) from None

        raise KlingAPIError("Max retries exceeded for Kling API request.")

    def _create_task(self, endpoint: str, request_data: Dict[str, Any]) -> str:
        """Submits a generation task and returns the task_id."""
        print(f"[KLING] Submitting task to {endpoint}...")
        res = self._request("POST", endpoint, request_data)
        data = res.get("data")
        if not data or "task_id" not in data:
            raise KlingAPIError(f"Kling API did not return a task_id. Response: {res}")
        task_id = data["task_id"]
        print(f"[KLING] Task submitted successfully. Task ID: {task_id}")
        return task_id

    def poll_task(self, endpoint: str, task_id: str, timeout: int = 1200) -> Dict[str, Any]:
        """Polls a task with adaptive intervals + jitter + cancellation support.

        v2.1 changes:
        - time.monotonic() (NTP-safe; was time.time()).
        - task_id URL-encoded (path injection safe).
        - Extended backoff: 15s after 2min, 30s after 5min.
        - Random jitter prevents thundering herds.
        - Chunked sleep with ComfyUI interrupt check every ~1s -- workflow
          cancellation propagates within 1s instead of waiting full interval.
        """
        # K-FIX v2.1: URL-encode untrusted task_id to prevent path injection.
        safe_task_id = _url_quote(str(task_id), safe="")
        start_time = time.monotonic()
        poll_count = 0
        print(f"[KLING] Starting status polling for task {task_id} (Timeout: {timeout}s)...")

        # Try to import ComfyUI's interrupt check; fall back to no-op if absent (tests).
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except Exception:
            def throw_exception_if_processing_interrupted():
                return None

        # ComfyUI progress bar for UI feedback (optional).
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        while True:
            throw_exception_if_processing_interrupted()
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise KlingAPIError(f"Polling timed out after {timeout} seconds for task {task_id}.")

            res = self._request("GET", f"{endpoint}/{safe_task_id}")
            data = res.get("data")
            if not data:
                logger.warning(f"[KLING] Polling returned no data for task {task_id}.")
                self._sleep_with_interrupt(5, throw_exception_if_processing_interrupted)
                poll_count += 1
                continue

            status = data.get("task_status", "unknown")
            progress = data.get("task_status_msg", "") or status
            print(f"[KLING] [{int(elapsed)}s] Status: [{status}] {progress}")

            # Best-effort progress hint (Kling doesn't report % so we estimate by elapsed/timeout)
            if pbar is not None:
                try:
                    pct = min(99, int((elapsed / timeout) * 100))
                    pbar.update_absolute(pct)
                except Exception:
                    pass

            if status in ("succeed", "succeeded"):
                print(f"[KLING] Task {task_id} completed in {int(elapsed)}s!")
                if pbar is not None:
                    try:
                        pbar.update_absolute(100)
                    except Exception:
                        pass
                return data
            if status in ("fail", "failed"):
                reason = data.get("task_status_msg", "Unknown Reason")
                print(f"[KLING] Task {task_id} failed after {int(elapsed)}s: {reason}")
                raise KlingAPIError(f"Kling Task Failed: {reason}")

            # v2.1: Extended backoff tiers + jitter for long video tasks.
            poll_count += 1
            if elapsed < 30:
                interval = 3
            elif elapsed < 120:
                interval = 5
            elif elapsed < 300:
                interval = 15
            else:
                interval = 30
            interval += random.uniform(0, 2)  # jitter
            self._sleep_with_interrupt(interval, throw_exception_if_processing_interrupted)

    @staticmethod
    def _sleep_with_interrupt(seconds: float, interrupt_check) -> None:
        """Sleep in 1-second chunks, polling the interrupt flag between chunks
        so ComfyUI cancellation propagates within ~1s."""
        end = time.monotonic() + seconds
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            interrupt_check()
            time.sleep(min(1.0, remaining))

    def text_to_video(self, model_name: str, prompt: str, aspect_ratio: str, duration: str, negative_prompt: str = "", cfg_scale: float = 0.5, camera_control: Optional[Dict[str, Any]] = None, mode: str = "pro", sound: str = "on", shot_type: str = "natural") -> str:
        data = {
            "model_name": model_name, "prompt": prompt, "negative_prompt": negative_prompt or None,
            "aspect_ratio": aspect_ratio, "duration": duration, "cfg_scale": cfg_scale,
            "camera_control": camera_control, "mode": mode, "sound": sound, "shot_type": shot_type
        }
        return self._create_task("/v1/videos/text2video", data)

    def image_to_video(self, model_name: str, image_b64: str, duration: str, prompt: str = "", image_tail_b64: Optional[str] = None, negative_prompt: str = "", cfg_scale: float = 0.5, camera_control: Optional[Dict[str, Any]] = None, mode: str = "pro", sound: str = "on") -> str:
        data = {
            "model_name": model_name or "kling-v1", "image": image_b64, "image_tail": image_tail_b64,
            "prompt": prompt or None, "negative_prompt": negative_prompt or None, "duration": duration,
            "cfg_scale": cfg_scale, "camera_control": camera_control, "mode": mode, "sound": sound
        }
        return self._create_task("/v1/videos/image2video", data)

    def omni_video(self, model_name: str, prompt: str, images: List[Dict[str, str]], videos: List[Dict[str, Any]], aspect_ratio: str, duration: str, mode: str = "pro", multi_prompt: Optional[List[Dict[str, Any]]] = None, shot_type: Optional[str] = None, sound: str = "on") -> str:
        data = {
            "model_name": model_name, "prompt": prompt, "image_list": images or None, "video_list": videos or None,
            "aspect_ratio": aspect_ratio, "duration": duration, "mode": mode, "sound": sound,
            "multi_prompt": multi_prompt, "shot_type": shot_type, "multi_shot": True if multi_prompt else False
        }
        return self._create_task("/v1/videos/omni-video", data)

    def extend_video(self, video_id: str, prompt: str, negative_prompt: str = "", cfg_scale: float = 0.5) -> str:
        data = {"video_id": video_id, "prompt": prompt, "negative_prompt": negative_prompt or None, "cfg_scale": cfg_scale}
        return self._create_task("/v1/videos/video-extend", data)

    def lip_sync(self, video_url: str, audio_url: Optional[str] = None, audio_b64: Optional[str] = None, text: Optional[str] = None, voice_id: Optional[str] = None, voice_speed: Optional[float] = None, mode: str = "audio2video") -> str:
        """Legacy/Standard Lip-Sync. Per Kling docs, voice_id/voice_speed are
        only valid in text2video mode -- callers should pass None for audio2video."""
        data = {"input": {"video_url": video_url, "mode": mode}}
        if audio_url:
            data["input"]["audio_url"] = audio_url
            data["input"]["audio_type"] = "url"
        if audio_b64:
            data["input"]["audio"] = audio_b64
            data["input"]["audio_type"] = "base64"
        if text:
            data["input"]["text"] = text
        if voice_id:
            data["input"]["voice_id"] = voice_id
        if voice_speed is not None:
            data["input"]["voice_speed"] = voice_speed
        return self._create_task("/v1/videos/lip-sync", data)

    def identify_face(self, video_url: str = None, video_id: str = None) -> Dict[str, Any]:
        """Identifies faces in a video for Advanced Lip-Sync. Returns session_id and face_data."""
        data = {}
        if video_url:
            data["video_url"] = video_url
        if video_id:
            data["video_id"] = video_id
        res = self._request("POST", "/v1/videos/identify-face", data)
        res_data = res.get("data")
        if not res_data:
            raise KlingAPIError(f"Kling face identification returned no data. Response: {res}")
        return res_data

    def advanced_lip_sync(self, session_id: str, face_id: str, audio_url: str, volume: int = 10, original_audio_volume: int = 0) -> str:
        """Submits an Advanced Lip-Sync task using a session_id and face_id."""
        data = {
            "session_id": session_id,
            "face_choose": [
                {
                    "face_id": face_id,
                    "sound_file": audio_url,
                    "sound_volume": volume,
                    "original_audio_volume": original_audio_volume
                }
            ]
        }
        return self._create_task("/v1/videos/advanced-lip-sync", data)

    def avatar(self, image_b64: str, audio_url: Optional[str] = None, audio_id: Optional[str] = None, prompt: str = "", mode: str = "pro") -> str:
        """Kling Avatar (Digital Human) Generation."""
        data = {
            "image": image_b64,
            "prompt": prompt or None,
            "mode": mode
        }
        if audio_url:
            data["sound_file"] = audio_url
        if audio_id:
            data["audio_id"] = audio_id
        return self._create_task("v1/videos/avatar/image2video", data)

    def video_effects(self, effect_scene: str, model_name: str, duration: str, images: List[str], mode: str = "std") -> str:
        data = {
            "effect_scene": effect_scene,
            "input": {
                "model_name": model_name,
                "mode": mode,
                "duration": duration
            }
        }
        if len(images) == 1:
            data["input"]["image"] = images[0]
        else:
            data["input"]["images"] = images
        return self._create_task("/v1/videos/effects", data)

    def text_to_audio(self, prompt: str, duration: int) -> str:
        data = {"prompt": prompt, "duration": duration}
        return self._create_task("/v1/audio/text-to-audio", data)

    def video_to_audio(self, video_url: str) -> str:
        data = {"video_url": video_url}
        return self._create_task("/v1/audio/video-to-audio", data)

    def tts(self, text: str, voice_id: str, voice_speed: float, voice_language: str = "en") -> str:
        data = {"text": text, "voice_id": voice_id, "voice_speed": voice_speed, "voice_language": voice_language}
        return self._create_task("/v1/audio/tts", data)

    def voice_clone(self, audio_url: Optional[str] = None, audio_b64: Optional[str] = None) -> str:
        """Clones a voice from audio and returns a reusable voice_id."""
        data = {}
        if audio_url:
            data["audio_url"] = audio_url
        if audio_b64:
            data["audio"] = audio_b64
        res = self._request("POST", "/v1/audio/voice-clone", data)
        res_data = res.get("data")
        if not res_data or "voice_id" not in res_data:
            raise KlingAPIError(f"Kling voice clone returned no voice_id. Response: {res}")
        return res_data["voice_id"]

    def image_generation(self, model_name: str, prompt: str, aspect_ratio: str, n: int, resolution: str = "1k", negative_prompt: str = "", fidelity: float = 0.5) -> str:
        data = {"model_name": model_name, "prompt": prompt, "negative_prompt": negative_prompt or None, "aspect_ratio": aspect_ratio, "n": n, "resolution": resolution.lower(), "fidelity": fidelity}
        return self._create_task("/v1/images/generations", data)

    def virtual_try_on(self, human_image_b64: str, cloth_image_b64: str, model_name: str = "kolors-virtual-try-on-v1") -> str:
        data = {"model_name": model_name, "human_image": human_image_b64, "cloth_image": cloth_image_b64}
        return self._create_task("/v1/images/kolors-virtual-try-on", data)

    def motion_control(self, model_name: str, image_b64: str, video_url: str, prompt: str = "", character_orientation: str = "image", mode: str = "pro") -> str:
        data = {
            "model_name": model_name, "image": image_b64, "video_url": video_url,
            "prompt": prompt or None, "character_orientation": character_orientation, "mode": mode
        }
        return self._create_task("/v1/videos/motion-control", data)

    def omni_image(self, model_name: str, prompt: str, images: List[Dict[str, str]], elements: Optional[List[Dict[str, Any]]] = None, aspect_ratio: str = "1:1", resolution: str = "1k", n: int = 1) -> str:
        data = {
            "model_name": model_name, "prompt": prompt, "image_list": images or None,
            "element_list": elements or None, "aspect_ratio": aspect_ratio, "resolution": resolution, "n": n
        }
        return self._create_task("/v1/images/omni-image", data)

    def extend_image(self, image_id: str, prompt: str = "", aspect_ratio: str = "1:1") -> str:
        data = {"image_id": image_id, "prompt": prompt or None, "aspect_ratio": aspect_ratio}
        return self._create_task("/v1/images/editing/expand", data)

    def multi_shot_image(self, model_name: str, prompt: str, shots: List[Dict[str, Any]], aspect_ratio: str = "1:1") -> str:
        data = {"model_name": model_name, "prompt": prompt, "shots": shots, "aspect_ratio": aspect_ratio}
        return self._create_task("/v1/images/ai-multi-shot", data)

    def image_recognize(self, image_b64: str) -> str:
        data = {"image": image_b64}
        return self._create_task("/v1/images/recognize", data)

    def effect_templates(self) -> Dict[str, Any]:
        """Fetches available effect templates."""
        return self._request("GET", "/v1/videos/effect-templates")

    def get_task_status(self, endpoint: str, task_id: str) -> Dict[str, Any]:
        """One-shot task status check (no polling). Used by Task Status node."""
        safe_task_id = _url_quote(str(task_id), safe="")
        return self._request("GET", f"{endpoint}/{safe_task_id}")

    def account_balance(self) -> Dict[str, Any]:
        """Fetches account balance / remaining credits. Endpoint may vary by region.
        Returns the raw `data` dict (caller should display whatever fields are present)."""
        # Kling exposes balance via /v1/account/costs ; we surface it raw.
        res = self._request("GET", "/v1/account/costs")
        return res.get("data", {})

    def upload_asset(self, file_path: str = None, b64_data: str = None, asset_type: str = "audio") -> Dict[str, Any]:
        """Uploads a local file or base64 data to Kling materials.

        v2.1: uses mimetypes for richer format detection (jpg, jpeg, mp3, m4a, etc).
        """
        url = f"{self.base_url}/v1/materials"
        headers = self._get_headers()
        headers.pop("Content-Type", None)

        # Per-type fallback defaults
        fallback_mime = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}.get(asset_type, "application/octet-stream")
        fallback_ext = {"image": "png", "video": "mp4", "audio": "wav"}.get(asset_type, "bin")

        multipart_data = [("type", (None, asset_type))]
        file_content = None

        try:
            if file_path:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Kling upload file not found: {file_path}")
                # Use mimetypes to support jpg/jpeg/mp3/m4a/etc.
                guessed, _ = mimetypes.guess_type(file_path)
                mime_type = guessed or fallback_mime
                file_content = open(file_path, "rb")
                multipart_data.append(("file", (os.path.basename(file_path), file_content, mime_type)))
            elif b64_data:
                multipart_data.append(("file", (f"upload.{fallback_ext}", io.BytesIO(base64.b64decode(b64_data)), fallback_mime)))
            else:
                raise ValueError("upload_asset requires either file_path or b64_data.")

            if self.debug:
                print(f"[KLING DEBUG] Uploading material to {url}...")
                print(f"[KLING DEBUG] asset_type: {asset_type}")

            try:
                response = self._session.post(url, headers=headers, files=multipart_data, timeout=UPLOAD_TIMEOUT)
            except requests.exceptions.RequestException as e:
                # SECURITY: scrub the underlying exception so the Authorization
                # header in e.request never appears in tracebacks.
                raise KlingAPIError(f"Kling upload network error: {type(e).__name__}") from None

            try:
                res_json = response.json()
            except (ValueError, Exception) as parse_err:
                logger.debug(f"[KLING] Could not parse upload response JSON: {parse_err}")
                res_json = {}

            if response.status_code != 200:
                if self.debug:
                    print(f"[KLING DEBUG] Asset Upload Error: {response.status_code}")
                code = res_json.get("code")
                if code is not None and code != 0:
                    msg = res_json.get("message", "Unknown Error")
                    hint = KLING_ERROR_MAP.get(code, "")
                    raise KlingAPIError(f"Kling Asset Upload Error {code}: {msg} {hint}", code=code, status_code=response.status_code)
                raise KlingAPIError(f"Kling upload HTTP {response.status_code}: {(response.text or '')[:200]}",
                                    status_code=response.status_code)

            return res_json.get("data", {})
        finally:
            if file_content is not None:
                try:
                    file_content.close()
                except Exception:
                    pass

    def upscale_image(self, image_id: str, model_name: str = "kling-v1") -> str:
        """Upscales an image."""
        data = {"image_id": image_id, "model_name": model_name}
        return self._create_task("/v1/images/upscale", data)

    def upscale_video(self, video_id: str, video_url: str = None, model_name: str = "kling-v1") -> str:
        """Upscales a video."""
        data = {"video_id": video_id, "video_url": video_url, "model_name": model_name}
        return self._create_task("/v1/videos/upscale", data)
