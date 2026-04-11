"""Standardized API error classes and response parsing for all services."""

import json


class APIError(RuntimeError):
    """Base API error with service name, HTTP status code, and detail message."""

    def __init__(self, service: str, status_code: int, detail: str):
        self.service = service
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{service}] HTTP {status_code}: {detail}")


class APITransientError(APIError):
    """Retryable API error (e.g. 429, 500, 502, 503, 504)."""
    pass


class APIPermanentError(APIError):
    """Non-retryable API error (e.g. 400, 401, 403, 404)."""
    pass


class APIQuotaError(APIPermanentError):
    """Credits or quota exhausted."""
    pass


_MAX_DETAIL_LEN = 300


def parse_error_response(service: str, status_code: int, body: str) -> APIError:
    """Parse an error response body and return the appropriate APIError subclass.

    Supports:
    - ElevenLabs format: {"detail": {"status": "...", "message": "..."}}
    - Kling format: {"code": N, "message": "..."}
    - Plain text fallback
    """
    detail = ""

    try:
        data = json.loads(body)

        # ElevenLabs format: {"detail": {"status": "...", "message": "..."}}
        if isinstance(data, dict) and "detail" in data:
            d = data["detail"]
            if isinstance(d, dict) and "message" in d:
                detail = d["message"]
            elif isinstance(d, str):
                detail = d
            else:
                detail = str(d)

        # Kling format: {"code": N, "message": "..."}
        elif isinstance(data, dict) and "message" in data:
            detail = data["message"]

        else:
            detail = str(data)

    except (json.JSONDecodeError, ValueError):
        detail = body

    # Truncate long details
    if len(detail) > _MAX_DETAIL_LEN:
        detail = detail[:_MAX_DETAIL_LEN] + "..."

    # Classify the error
    if status_code == 429 or status_code >= 500:
        return APITransientError(service, status_code, detail)
    elif status_code in (402, 403) and any(
        kw in detail.lower() for kw in ("quota", "credit", "limit", "exhausted")
    ):
        return APIQuotaError(service, status_code, detail)
    else:
        return APIPermanentError(service, status_code, detail)
