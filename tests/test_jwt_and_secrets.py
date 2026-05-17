"""JWT generation correctness and secret-leakage regression tests."""

import base64
import hashlib
import hmac
import json
import time
import traceback
from unittest.mock import patch, MagicMock

import pytest
import requests

import kling_client
from kling_client import KlingClient, KlingAPIError, _strip_none, _parse_retry_after


SECRET = "super-secret-do-not-leak-1234567890abcdef"
ACCESS = "AK-public-access-key-do-not-leak-9876"


def _decode_jwt_parts(token: str):
    h, p, s = token.split(".")
    def _decode(b64):
        # add padding
        padded = b64 + "=" * (-len(b64) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    return _decode(h), _decode(p), s


def test_jwt_structure_is_valid_hs256():
    c = KlingClient(ACCESS, SECRET)
    tok = c._generate_token()
    parts = tok.split(".")
    assert len(parts) == 3
    header, payload, sig_b64 = _decode_jwt_parts(tok)
    assert header == {"alg": "HS256", "typ": "JWT"}
    assert payload["iss"] == ACCESS
    assert "exp" in payload and "nbf" in payload
    # exp must be in the future
    assert payload["exp"] > int(time.time())
    # nbf must be in the past (clock-skew tolerance)
    assert payload["nbf"] < int(time.time()) + 1


def test_jwt_signature_matches_known_hs256():
    """Bit-exact HMAC verification: regenerate the signature from raw secret
    and check it matches what _generate_token produced."""
    c = KlingClient(ACCESS, SECRET)
    tok = c._generate_token()
    h_b64, p_b64, s_b64 = tok.split(".")
    msg = f"{h_b64}.{p_b64}".encode("utf-8")
    expected_sig = hmac.new(SECRET.encode(), msg, hashlib.sha256).digest()
    expected_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
    assert s_b64 == expected_b64


def test_jwt_is_cached_across_calls():
    """The token is cached on the instance; back-to-back calls return the same string."""
    c = KlingClient(ACCESS, SECRET)
    a = c._generate_token()
    b = c._generate_token()
    assert a == b


def test_secret_not_in_network_exception_traceback():
    """REGRESSION: a requests.RequestException must NOT bubble up with the
    Authorization header (which embeds the signed JWT) anywhere in its repr
    or its traceback locals.

    Pre-fix the requests library would attach the original Request object
    (carrying headers) to the exception; the user's console log would then
    contain the JWT (and effectively their identity for 30 mins).
    """
    c = KlingClient(ACCESS, SECRET)

    def boom(*a, **kw):
        # Build a Request-like object that carries the secret header — this is
        # what `requests` does in real life.
        e = requests.exceptions.ConnectionError("simulated network drop")
        fake_req = MagicMock()
        fake_req.headers = {"Authorization": f"Bearer {c._generate_token()}"}
        e.request = fake_req
        raise e

    with patch.object(c._session, "post", side_effect=boom), patch.object(c._session, "get", side_effect=boom):
        with pytest.raises(KlingAPIError) as excinfo:
            c._request("POST", "/v1/videos/text2video", {"prompt": "x"}, retries=1)

    # Stringify the exception and its full chain — secret must NOT appear.
    full_text = "".join(traceback.format_exception(
        type(excinfo.value), excinfo.value, excinfo.value.__traceback__))
    assert SECRET not in full_text, "Secret leaked through exception chain!"
    # The signed JWT (Bearer ...) also must NOT appear.
    tok = c._generate_token()
    assert tok not in full_text, "Signed JWT leaked through exception chain!"


def test_secret_not_in_debug_logs(capsys):
    """Debug-mode request prints must mask large strings (b64 images/audio)
    and must NEVER print the Authorization header."""
    c = KlingClient(ACCESS, SECRET, debug=True)

    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"code": 0, "data": {"task_id": "ok"}}
    with patch.object(c._session, "post", return_value=fake):
        c._request("POST", "/v1/videos/text2video", {"prompt": "x", "image": "A" * 5000})

    out = capsys.readouterr().out
    assert SECRET not in out
    assert "Authorization" not in out
    # The huge b64 image string must have been truncated.
    assert "A" * 1000 not in out


def test_strip_none_recurses_into_lists():
    """K-FIX: _strip_none must handle Kling's image_list/video_list (lists of dicts)."""
    payload = {
        "prompt": "hi",
        "image_list": [{"image": "abc", "weight": None}, {"image": "def"}, None],
        "ignore_me": None,
    }
    out = _strip_none(payload)
    assert "ignore_me" not in out
    assert len(out["image_list"]) == 2
    assert "weight" not in out["image_list"][0]  # None stripped
    assert out["image_list"][1]["image"] == "def"


def test_parse_retry_after_seconds():
    assert _parse_retry_after("30", fallback=5) == 30.0


def test_parse_retry_after_caps_at_300():
    assert _parse_retry_after("9999", fallback=5) == 300.0


def test_parse_retry_after_http_date_format():
    # Far-future HTTP-date should clamp at 300s, not crash
    val = _parse_retry_after("Wed, 01 Jan 2099 00:00:00 GMT", fallback=5)
    assert val == 300.0


def test_parse_retry_after_falls_back_on_garbage():
    assert _parse_retry_after("not-a-number", fallback=7) == 7
