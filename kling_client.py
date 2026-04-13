"""Kling AI API client with JWT auth, retry logic, and task polling."""

import hmac
import hashlib
import time
import base64
import json
import requests
import logging
import os
import io
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# --- Client Cache ---
_client_cache: Dict[tuple, "KlingClient"] = {}


def get_client(access_key: str, secret_key: str, debug: bool = False) -> "KlingClient":
    """Return a cached KlingClient for the given credentials, creating one if needed."""
    cache_key = (access_key, secret_key)
    client = _client_cache.get(cache_key)
    if client is None:
        client = KlingClient(access_key, secret_key, debug=debug)
        _client_cache[cache_key] = client
    # Always update debug flag to match latest request
    client.debug = debug
    return client


# --- Kling Error Code Mapping ---
KLING_ERROR_MAP = {
    401: "Unauthorized: Invalid AccessKey or SecretKey. Please check your Kling AI Authentication node.",
    403: "Forbidden: You may have run out of credits or don't have permission for this model.",
    429: "Too Many Requests: Kling API rate limit hit or account issue. Try slowing down or check your balance.",
    1000: "Invalid Parameter: One of your inputs (prompt length, image size) is invalid.",
    1101: "Invalid Image: The provided image is too large or has an unsupported aspect ratio.",
    1102: "Account Balance Concern: Often means 'Account balance not enough'. Check your Kling credits.",
    1107: "Invalid Audio: The provided audio file is problematic.",
    1200: "Server Busy: Kling's internal system is overloaded. Try again in 60 seconds.",
    1201: "Internal Error: Kling server had a hiccup. Your task might still be in the queue.",
}

# Non-retryable Kling error codes -- retrying won't help
_PERMANENT_ERROR_CODES = {401, 403, 1000, 1101, 1102, 1107}
# Retryable Kling error codes -- transient server issues
_TRANSIENT_ERROR_CODES = {1200, 1201}

REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 120


def _strip_none(d: dict) -> dict:
    """Recursively remove keys with None values from a dict to keep payloads clean."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = _strip_none(v)
        else:
            out[k] = v
    return out


class KlingClient:
    def __init__(self, access_key: str, secret_key: str, base_url: str = "https://api-singapore.klingai.com", debug: bool = False):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.debug = debug
        # Persistent session for connection pooling -- reuses TCP connections across polling loops
        self._session = requests.Session()

    def _generate_token(self) -> str:
        """Generates an HS256 JWT token for authentication."""
        def b64_encode(d: Dict[str, Any]) -> str:
            return base64.urlsafe_b64encode(json.dumps(d, separators=(',', ':')).encode('utf-8')).decode('utf-8').rstrip('=')

        header_enc = b64_encode({"alg": "HS256", "typ": "JWT"})
        payload_enc = b64_encode({
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        })

        msg = f"{header_enc}.{payload_enc}".encode('utf-8')
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            msg,
            hashlib.sha256
        ).digest()

        token = f"{header_enc}.{payload_enc}.{base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')}"
        return token

    def _get_headers(self) -> Dict[str, str]:
        """Base headers for all requests. Token is regenerated each call to stay fresh."""
        return {
            "Authorization": f"Bearer {self._generate_token()}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None, retries: int = 3) -> Dict[str, Any]:
        """Performs an authenticated HTTP request with enhanced QOS, Debug, and Error Parsing."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = self._get_headers()

        # Strip None values from payload to keep requests clean
        clean_data = _strip_none(data) if data else data

        if self.debug and clean_data:
            log_data = clean_data.copy()
            for key in ("image", "audio", "human_image", "cloth_image", "image_tail"):
                if key in log_data and isinstance(log_data[key], str) and len(log_data[key]) > 30:
                    log_data[key] = log_data[key][:20] + "..."
            print(f"[KLING DEBUG] Request to {url}:\n{json.dumps(log_data, indent=2)}")

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
                        raise Exception(f"Kling API Error {code}: {msg}\n[Hint]: {hint}")

                    # Transient errors -- retry with 60s backoff for error 1200
                    if code in _TRANSIENT_ERROR_CODES and attempt < retries - 1:
                        wait = 60 if code == 1200 else (attempt + 1) * 5
                        print(f"[KLING] Transient error {code}. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(wait)
                        continue

                    raise Exception(f"Kling API Error {code}: {msg}\n[Hint]: {hint}")

                # Handle standard HTTP errors (rate limits, server issues)
                if response.status_code in [429, 502, 503, 504]:
                    if attempt < retries - 1:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = min(int(retry_after), 60)
                            except ValueError:
                                wait = (attempt + 1) * 5
                        else:
                            wait = (attempt + 1) * 5
                        print(f"[KLING] Status {response.status_code}. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(wait)
                        continue

                response.raise_for_status()
                return res_json

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"[KLING] Connection error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"Kling Request Failed after {retries} attempts: {e}")
                raise e
            except Exception as e:
                # Re-raise parsed Kling errors immediately
                raise e

        raise Exception("Max retries exceeded for Kling API request.")

    def _create_task(self, endpoint: str, request_data: Dict[str, Any]) -> str:
        """Submits a generation task and returns the task_id."""
        print(f"[KLING] Submitting task to {endpoint}...")
        res = self._request("POST", endpoint, request_data)
        data = res.get("data")
        if not data or "task_id" not in data:
            raise Exception(f"Kling API did not return a task_id. Response: {res}")
        task_id = data["task_id"]
        print(f"[KLING] Task submitted successfully. Task ID: {task_id}")
        return task_id

    def poll_task(self, endpoint: str, task_id: str, timeout: int = 1200) -> Dict[str, Any]:
        """Polls a task with adaptive intervals and elapsed time display."""
        start_time = time.time()
        poll_count = 0
        print(f"[KLING] Starting status polling for task {task_id} (Timeout: {timeout}s)...")

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise Exception(f"Polling timed out after {timeout} seconds.")

            res = self._request("GET", f"{endpoint}/{task_id}")
            data = res.get("data")
            if not data:
                logger.warning(f"[KLING] Polling returned no data for task {task_id}. Raw response: {res}")
                time.sleep(5)
                poll_count += 1
                continue

            status = data.get("task_status", "unknown")
            progress = data.get("task_status_msg", "") or status
            print(f"[KLING] [{int(elapsed)}s] Status: [{status}] {progress}")

            if status in ["succeed", "succeeded"]:
                print(f"[KLING] Task {task_id} completed in {int(elapsed)}s!")
                return data
            elif status in ["fail", "failed"]:
                reason = data.get("task_status_msg", "Unknown Reason")
                print(f"[KLING] Task {task_id} failed after {int(elapsed)}s: {reason}")
                raise Exception(f"Kling Task Failed: {reason}")

            # Adaptive interval: 3s for first 30s, 5s up to 2min, 8s after that
            poll_count += 1
            if elapsed < 30:
                interval = 3
            elif elapsed < 120:
                interval = 5
            else:
                interval = 8
            time.sleep(interval)

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

    def lip_sync(self, video_url: str, audio_url: Optional[str] = None, audio_b64: Optional[str] = None, text: Optional[str] = None, voice_id: Optional[str] = None, voice_speed: float = 1.0, mode: str = "audio2video") -> str:
        """Legacy/Standard Lip-Sync."""
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
        if voice_speed:
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
            raise Exception(f"Kling face identification returned no data. Response: {res}")
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
            raise Exception(f"Kling voice clone returned no voice_id. Response: {res}")
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

    def upload_asset(self, file_path: str = None, b64_data: str = None, asset_type: str = "audio") -> Dict[str, Any]:
        """Uploads a local file or base64 data to Kling materials (Singapore)."""
        url = f"{self.base_url}/v1/materials"
        headers = self._get_headers()
        headers.pop("Content-Type", None)

        mime_map = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}
        ext_map = {"image": "png", "video": "mp4", "audio": "wav"}

        mime_type = mime_map.get(asset_type, "audio/wav")
        ext = ext_map.get(asset_type, "wav")

        multipart_data = [
            ("type", (None, asset_type))
        ]

        file_content = None
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Kling upload file not found: {file_path}")
            file_content = open(file_path, "rb")
            multipart_data.append(("file", (os.path.basename(file_path), file_content, mime_type)))
        elif b64_data:
            multipart_data.append(("file", (f"upload.{ext}", io.BytesIO(base64.b64decode(b64_data)), mime_type)))
        else:
            raise ValueError("upload_asset requires either file_path or b64_data.")

        try:
            if self.debug:
                print(f"[KLING DEBUG] Uploading material to {url}...")
                print(f"[KLING DEBUG] asset_type: {asset_type}")

            response = self._session.post(url, headers=headers, files=multipart_data, timeout=UPLOAD_TIMEOUT)

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
                    raise Exception(f"Kling Asset Upload Error {code}: {msg} {hint}")
                elif self.debug:
                    print(f"[KLING DEBUG] Response Body: {response.text[:200]}")

            response.raise_for_status()
            return res_json.get("data", {})
        finally:
            if file_content:
                file_content.close()

    def upscale_image(self, image_id: str, model_name: str = "kling-v1") -> str:
        """Upscales an image."""
        data = {"image_id": image_id, "model_name": model_name}
        return self._create_task("/v1/images/upscale", data)

    def upscale_video(self, video_id: str, video_url: str = None, model_name: str = "kling-v1") -> str:
        """Upscales a video."""
        data = {"video_id": video_id, "video_url": video_url, "model_name": model_name}
        return self._create_task("/v1/videos/upscale", data)
