"""Conversion roundtrips, EMPTY_AUDIO mutation safety, cv2 cleanup, and
new-node behavior. All tests are pure local — no Kling API calls."""

import io
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

import kling_nodes as kn
from kling_nodes import (
    tensor_to_base64_string,
    audio_to_base64_string,
    EMPTY_AUDIO,
    load_audio_to_tensor,
    _empty_audio,
    _sanitize_filename,
    _safe_url,
    download_to_output,
    KlingDirect_CameraPreset,
    KlingDirect_AspectRatioPicker,
    KlingDirect_CostEstimator,
    KlingDirect_VoiceCatalog,
    KlingDirect_RegionSelector,
    KlingDirect_VideoToFile,
)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def test_tensor_to_base64_returns_string():
    img = torch.rand((1, 320, 320, 3))
    b64 = tensor_to_base64_string(img)
    assert isinstance(b64, str)
    assert len(b64) > 100


def test_tensor_to_base64_handles_rgba():
    """K-FIX: tensor_to_base64_string must NOT crash on a 4-channel input
    (some ComfyUI nodes emit RGBA)."""
    img = torch.rand((1, 320, 320, 4))
    b64 = tensor_to_base64_string(img)
    assert isinstance(b64, str)


def test_tensor_to_base64_returns_none_on_none():
    assert tensor_to_base64_string(None) is None


def test_audio_to_base64_empty_waveform_raises_value_error():
    """K-FIX REGRESSION: empty waveform used to crash with ZeroDivisionError
    inside `repeats = ceil(MIN_AUDIO_DURATION / duration)`."""
    audio = {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 44100}
    with pytest.raises(ValueError, match="empty"):
        audio_to_base64_string(audio)


def test_audio_to_base64_none_sample_rate_raises_value_error():
    """K-FIX REGRESSION: sample_rate=None used to cause a TypeError, not a
    clear user-facing error."""
    audio = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": None}
    with pytest.raises(ValueError, match="invalid sample_rate"):
        audio_to_base64_string(audio)


def test_audio_to_base64_negative_sample_rate_raises():
    audio = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": -1}
    with pytest.raises(ValueError, match="invalid sample_rate"):
        audio_to_base64_string(audio)


def test_audio_to_base64_normal_case_produces_b64_string():
    # 2 sec of silence at 44100 Hz, stereo
    audio = {"waveform": torch.zeros((1, 2, 88200)), "sample_rate": 44100}
    b64 = audio_to_base64_string(audio)
    assert isinstance(b64, str)
    assert len(b64) > 100


# --------------------------------------------------------------------------
# EMPTY_AUDIO mutation safety
# --------------------------------------------------------------------------

def test_empty_audio_is_cloned():
    """K-FIX: mutating one _empty_audio() result must not affect the next call,
    nor the module-level EMPTY_AUDIO singleton."""
    a = _empty_audio()
    b = _empty_audio()
    a["waveform"][...] = 42.0
    assert torch.all(b["waveform"] == 0), "EMPTY_AUDIO was shared, not cloned!"
    assert torch.all(EMPTY_AUDIO["waveform"] == 0), "Module-level EMPTY_AUDIO was mutated!"


def test_load_audio_to_tensor_handles_bad_file_with_clone():
    """When torchaudio fails, load_audio_to_tensor returns a fresh empty audio
    (not a shared reference)."""
    a = load_audio_to_tensor("/nonexistent/path/zzz.mp3")
    b = load_audio_to_tensor("/nonexistent/path/zzz.mp3")
    a["waveform"][...] = 7.0
    assert torch.all(b["waveform"] == 0)


# --------------------------------------------------------------------------
# Filename / URL safety
# --------------------------------------------------------------------------

def test_sanitize_filename_strips_bad_chars():
    name = 'evil<>:"/\\|?*name'
    out = _sanitize_filename(name)
    for bad in "<>:\"/\\|?*":
        assert bad not in out


def test_sanitize_filename_truncates_to_max_len():
    name = "x" * 500
    assert len(_sanitize_filename(name, max_len=100)) == 100


def test_safe_url_rejects_file_scheme():
    with pytest.raises(ValueError, match="non-http"):
        _safe_url("file:///etc/passwd")


def test_safe_url_rejects_gopher():
    with pytest.raises(ValueError, match="non-http"):
        _safe_url("gopher://internal/data")


def test_safe_url_rejects_loopback_resolved_host():
    # Use a hostname that resolves to 127.0.0.1
    with pytest.raises(ValueError, match="internal/loopback"):
        _safe_url("http://localhost/data")


def test_safe_url_accepts_https():
    # example.com resolves to public IPs
    url = _safe_url("https://example.com/path/to/video.mp4")
    assert url == "https://example.com/path/to/video.mp4"


# --------------------------------------------------------------------------
# cv2 cleanup on error path
# --------------------------------------------------------------------------

def test_load_video_to_tensor_releases_cap_on_failed_open():
    """K-FIX: cap.release() must run even when cap.isOpened()==False."""
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=fake_cap):
        with pytest.raises(Exception, match="Could not open"):
            kn.load_video_to_tensor("/path/does/not/exist.mp4")
    fake_cap.release.assert_called()


# --------------------------------------------------------------------------
# download_to_output retries and cleans up
# --------------------------------------------------------------------------

def test_download_to_output_retries_and_cleans_up(tmp_path):
    """K-FIX: download_to_output must retry transient errors and not leave
    partial files on disk."""
    import requests as _req
    # Direct output dir to tmp_path
    output_dir = str(tmp_path)
    with patch.object(kn, "folder_paths") as fp:
        fp.get_output_directory.return_value = output_dir
        # Simulate 3 connection errors -> RuntimeError raised
        with patch("requests.get", side_effect=_req.exceptions.ConnectionError("boom")), \
             patch("kling_nodes._safe_url", side_effect=lambda u: u), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Download failed"):
                download_to_output("https://example.com/v.mp4")
    # No partial files left in output dir
    assert os.listdir(output_dir) == []


# --------------------------------------------------------------------------
# New v2.1 node behavior
# --------------------------------------------------------------------------

def test_camera_preset_outputs_klingcamera_dict():
    cam, = KlingDirect_CameraPreset().build("orbit_left", intensity=1.0)
    assert isinstance(cam, dict)
    assert cam["type"] == "horizontal"
    assert cam["config"]["horizontal"] == -7.0


def test_camera_preset_intensity_scales_values():
    cam, = KlingDirect_CameraPreset().build("zoom_in", intensity=0.5)
    assert cam["config"]["zoom"] == 7.0 * 0.5


def test_camera_preset_none_emits_simple():
    cam, = KlingDirect_CameraPreset().build("none", intensity=1.0)
    assert cam["type"] == "simple"


def test_aspect_ratio_picker_landscape():
    img = torch.zeros((1, 720, 1280, 3))  # 16:9
    ratio, info = KlingDirect_AspectRatioPicker().pick(img)
    assert ratio == "16:9"


def test_aspect_ratio_picker_portrait():
    img = torch.zeros((1, 1280, 720, 3))  # 9:16
    ratio, info = KlingDirect_AspectRatioPicker().pick(img)
    assert ratio == "9:16"


def test_aspect_ratio_picker_square():
    img = torch.zeros((1, 512, 512, 3))  # 1:1
    ratio, info = KlingDirect_AspectRatioPicker().pick(img)
    assert ratio == "1:1"


def test_cost_estimator_t2v_pro_is_double_std():
    pro, _ = KlingDirect_CostEstimator().estimate("text2video", "kling-v3", 5, "pro")
    std, _ = KlingDirect_CostEstimator().estimate("text2video", "kling-v3", 5, "std")
    assert pro == 2 * std


def test_cost_estimator_image_scales_with_n():
    one, _ = KlingDirect_CostEstimator().estimate("image", "kling-v3", 5, "pro", n=1)
    four, _ = KlingDirect_CostEstimator().estimate("image", "kling-v3", 5, "pro", n=4)
    assert four == 4 * one


def test_voice_catalog_returns_valid_json():
    import json as _json
    out, = KlingDirect_VoiceCatalog().list()
    rows = _json.loads(out)
    assert len(rows) > 5
    assert all({"display_name", "voice_id", "language"} <= set(r) for r in rows)


def test_region_selector_overrides_base_url():
    new_auth, = KlingDirect_RegionSelector().select({"access_key": "ak", "secret_key": "sk"}, "china")
    assert new_auth["base_url"] == "https://api.klingai.com"


def test_region_selector_custom_url_wins():
    new_auth, = KlingDirect_RegionSelector().select(
        {"access_key": "ak", "secret_key": "sk"}, "singapore", custom_base_url="https://my-proxy.example/v1")
    assert new_auth["base_url"] == "https://my-proxy.example/v1"


def test_video_to_file_writes_mp4(tmp_path):
    """VideoToFile produces a non-empty mp4 on disk."""
    video = torch.rand((5, 64, 64, 3))  # 5 frames
    with patch.object(kn, "folder_paths") as fp:
        fp.get_output_directory.return_value = str(tmp_path)
        (file_path,) = KlingDirect_VideoToFile().write(video, fps=12, filename_prefix="t", codec="mp4v")
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


# --------------------------------------------------------------------------
# Cloud uploader consent
# --------------------------------------------------------------------------

def test_cloud_uploader_requires_consent():
    """K-FIX SECURITY: must refuse upload unless the user ticks the consent box."""
    node = kn.KlingDirect_CloudUploader()
    with pytest.raises(ValueError, match="i_understand_uploads_are_public"):
        node.upload("catbox", i_understand_uploads_are_public=False, file_path="/tmp/x")
