"""Polling loop, retry semantics, and ComfyUI cancellation tests.

NO real Kling calls — every network method is mocked.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from kling_client import KlingClient, KlingAPIError, _PERMANENT_ERROR_CODES, _TRANSIENT_ERROR_CODES


def test_1106_is_permanent_not_retried():
    """REGRESSION: Code 1106 (Task Failed) used to be in _TRANSIENT_ERROR_CODES,
    causing useless re-queries against a known-failed task."""
    assert 1106 in _PERMANENT_ERROR_CODES
    assert 1106 not in _TRANSIENT_ERROR_CODES


def test_poll_task_terminates_on_succeed():
    c = KlingClient("ak", "sk")
    # First call returns processing, second returns succeed.
    calls = [
        {"data": {"task_status": "processing", "task_status_msg": "queued"}},
        {"data": {"task_status": "succeed", "task_result": {"videos": [{"url": "https://example.com/v.mp4"}]}}},
    ]
    with patch.object(c, "_request", side_effect=calls), \
         patch("time.sleep"):
        data = c.poll_task("/v1/videos/text2video", "abc123", timeout=60)
    assert data["task_status"] == "succeed"


def test_poll_task_raises_on_fail():
    c = KlingClient("ak", "sk")
    with patch.object(c, "_request", return_value={"data": {"task_status": "failed", "task_status_msg": "OOM"}}), \
         patch("time.sleep"):
        with pytest.raises(KlingAPIError) as ei:
            c.poll_task("/v1/videos/text2video", "abc123", timeout=60)
    assert "OOM" in str(ei.value)


def test_poll_task_url_encodes_task_id():
    """K-FIX SECURITY: task_id is URL-quoted to prevent path traversal."""
    c = KlingClient("ak", "sk")
    captured = {}

    def fake_request(method, path, *a, **kw):
        captured["path"] = path
        return {"data": {"task_status": "succeed", "task_result": {"videos": [{"url": "x"}]}}}

    with patch.object(c, "_request", side_effect=fake_request), patch("time.sleep"):
        c.poll_task("/v1/videos/text2video", "../../malicious", timeout=10)
    # The "../" must be percent-encoded.
    assert "%2F" in captured["path"] or "%2E" in captured["path"]
    assert "../" not in captured["path"]


def test_poll_task_times_out():
    c = KlingClient("ak", "sk")
    with patch.object(c, "_request", return_value={"data": {"task_status": "processing"}}), \
         patch("time.sleep"):
        # Force monotonic to leap forward each call so we exit on timeout fast.
        t = [0.0]
        def mono():
            t[0] += 100
            return t[0]
        with patch("kling_client.time.monotonic", side_effect=mono):
            with pytest.raises(KlingAPIError) as ei:
                c.poll_task("/v1/videos/text2video", "abc", timeout=1)
    assert "timed out" in str(ei.value).lower()


def test_poll_task_uses_monotonic_clock():
    """REGRESSION: must call time.monotonic(), never time.time(), so NTP syncs
    don't cause premature/infinite polling."""
    c = KlingClient("ak", "sk")
    with patch.object(c, "_request", return_value={"data": {"task_status": "succeed", "task_result": {"videos": [{"url": "x"}]}}}), \
         patch("time.sleep"), \
         patch("kling_client.time.monotonic", wraps=time.monotonic) as mono:
        c.poll_task("/v1/videos/text2video", "abc", timeout=60)
    assert mono.called, "poll_task must use time.monotonic()"


def test_transient_http_error_is_retried():
    c = KlingClient("ak", "sk")

    bad = MagicMock(status_code=503, headers={}, text="Service Unavailable")
    bad.json.return_value = {}
    good = MagicMock(status_code=200)
    good.json.return_value = {"code": 0, "data": {"task_id": "abc"}}
    seq = [bad, good]

    def fake_post(*a, **kw):
        return seq.pop(0)

    with patch.object(c._session, "post", side_effect=fake_post), patch("time.sleep"):
        res = c._request("POST", "/v1/videos/text2video", {"prompt": "x"}, retries=3)
    assert res["data"]["task_id"] == "abc"


def test_permanent_kling_error_does_not_retry():
    c = KlingClient("ak", "sk")
    bad = MagicMock(status_code=200, headers={}, text="")
    bad.json.return_value = {"code": 1001, "message": "Invalid Token"}

    with patch.object(c._session, "post", return_value=bad) as post, patch("time.sleep"):
        with pytest.raises(KlingAPIError) as ei:
            c._request("POST", "/v1/videos/text2video", {"prompt": "x"}, retries=3)
    assert post.call_count == 1
    assert "1001" in str(ei.value)


def test_comfyui_interrupt_propagates_within_one_second():
    """K-FIX: poll_task must check the interrupt flag via _sleep_with_interrupt.
    Cancellation must propagate quickly (next sleep chunk) rather than waiting
    for the full 8-30s polling interval."""
    c = KlingClient("ak", "sk")

    # When the patched _sleep_with_interrupt raises, the poll loop must
    # propagate the exception immediately.
    def raise_interrupt(self_or_cls, secs, check):
        raise InterruptedError("user cancelled")

    with patch.object(c, "_request", return_value={"data": {"task_status": "processing"}}), \
         patch("time.sleep"):
        with patch.object(KlingClient, "_sleep_with_interrupt", raise_interrupt):
            with pytest.raises(InterruptedError):
                c.poll_task("/v1/videos/text2video", "abc", timeout=60)
