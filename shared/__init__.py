"""Shared utilities for ComfyUI-AI-Suite.

Re-exports everything so services can do:
    from ...shared import tensor_to_base64, AlwaysExecuteMixin, api_request_with_retry
"""

from .errors import (
    APIError,
    APITransientError,
    APIPermanentError,
    APIQuotaError,
    parse_error_response,
)

from .retry import (
    api_request_with_retry,
    download_file,
)

from .node_utils import (
    AlwaysExecuteMixin,
)

from .auth import (
    BaseAPIKeyNode,
    DualKeyAPIKeyNode,
)

from .conversions import (
    tensor_to_pil,
    pil_to_tensor,
    tensor_to_base64,
    tensor_to_jpeg_bytes,
    mask_to_jpeg_bytes,
    bytes_to_tensor,
    audio_to_comfy,
    comfy_to_audio_bytes,
)

__all__ = [
    # errors
    "APIError",
    "APITransientError",
    "APIPermanentError",
    "APIQuotaError",
    "parse_error_response",
    # retry
    "api_request_with_retry",
    "download_file",
    # node_utils
    "AlwaysExecuteMixin",
    # auth
    "BaseAPIKeyNode",
    "DualKeyAPIKeyNode",
    # conversions
    "tensor_to_pil",
    "pil_to_tensor",
    "tensor_to_base64",
    "tensor_to_jpeg_bytes",
    "mask_to_jpeg_bytes",
    "bytes_to_tensor",
    "audio_to_comfy",
    "comfy_to_audio_bytes",
]
