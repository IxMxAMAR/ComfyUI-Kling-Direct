"""HTTP request helpers with exponential-backoff retry logic."""

import io
import time
from typing import Optional

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout
from requests.exceptions import ChunkedEncodingError as ReqChunkedEncodingError
from requests.exceptions import RequestException as ReqRequestException

from .errors import APITransientError, APIPermanentError, parse_error_response


def api_request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: float = 60,
    transient_codes: tuple = (429, 500, 502, 503, 504),
    service_name: str = "API",
    **kwargs,
) -> requests.Response:
    """Make an HTTP request with exponential backoff retry on transient failures.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        url: Request URL
        session: Optional requests.Session to reuse connections
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        timeout: Request timeout in seconds
        transient_codes: HTTP status codes to retry on
        service_name: Service name for error messages
        **kwargs: Additional arguments passed to requests

    Returns:
        requests.Response on success

    Raises:
        APITransientError: After exhausting retries on transient errors
        APIPermanentError: Immediately on non-retryable errors
    """
    requester = session or requests
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requester.request(method, url, timeout=timeout, **kwargs)

            if response.status_code < 400:
                return response

            # Parse the error
            body = response.text
            error = parse_error_response(service_name, response.status_code, body)

            if response.status_code in transient_codes:
                last_error = error
                if attempt < max_retries:
                    # Honor Retry-After header
                    delay = _get_retry_delay(response, base_delay, attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise last_error
            else:
                # Non-transient error -- raise immediately
                raise error

        except (
            ReqConnectionError,
            ReqTimeout,
            ReqChunkedEncodingError,
        ) as exc:
            last_error = APITransientError(
                service_name, 0, f"Connection error: {exc}"
            )
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                raise last_error from exc

    # Should not reach here, but just in case
    raise last_error  # type: ignore[misc]


def _get_retry_delay(
    response: requests.Response, base_delay: float, attempt: int
) -> float:
    """Calculate retry delay, honoring Retry-After header if present."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return base_delay * (2 ** attempt)


def download_file(
    url: str,
    *,
    retries: int = 3,
    timeout: float = 120,
    chunk_size: int = 8192,
) -> bytes:
    """Download a file with retry logic, returns bytes.

    Args:
        url: URL to download from
        retries: Number of retry attempts
        timeout: Request timeout in seconds
        chunk_size: Size of chunks for streaming download

    Returns:
        File contents as bytes
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    buffer.write(chunk)
            return buffer.getvalue()

        except (
            ReqRequestException,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (2 ** attempt))
                continue

    raise APITransientError(
        "download", 0, f"Download failed after {retries + 1} attempts: {last_error}"
    )
