"""Image, audio, and tensor conversion utilities for ComfyUI nodes."""

import base64
import io
import struct
from typing import Optional

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI image tensor [B,H,W,C] float32 to a PIL Image (first from batch).

    Args:
        tensor: Image tensor of shape [B, H, W, C] with values in [0, 1]

    Returns:
        PIL Image in RGB mode
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first from batch
    # Clamp and convert to uint8
    arr = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a ComfyUI image tensor [1,H,W,C] float32.

    Args:
        image: PIL Image (will be converted to RGB if needed)

    Returns:
        Tensor of shape [1, H, W, C] with values in [0, 1]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_base64(tensor: Optional[torch.Tensor], fmt: str = "PNG") -> Optional[str]:
    """Convert a ComfyUI image tensor to a base64-encoded string.

    Args:
        tensor: Image tensor [B,H,W,C] or None
        fmt: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 string or None if tensor is None
    """
    if tensor is None:
        return None
    img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_jpeg_bytes(tensor: torch.Tensor, quality: int = 95) -> bytes:
    """Convert a ComfyUI image tensor to JPEG bytes.

    Args:
        tensor: Image tensor [B,H,W,C]
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded bytes
    """
    img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Convert image bytes (any PIL-supported format) to a ComfyUI tensor [1,H,W,C].

    Args:
        data: Raw image bytes

    Returns:
        Tensor of shape [1, H, W, C] with values in [0, 1]
    """
    img = Image.open(io.BytesIO(data))
    return pil_to_tensor(img)


def audio_to_comfy(audio_bytes: bytes, output_format: str = "wav") -> dict:
    """Convert raw audio bytes to ComfyUI AUDIO dict format.

    Tries soundfile first, then falls back to torchaudio.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, FLAC, OGG, etc.)
        output_format: Hint about the source format

    Returns:
        ComfyUI AUDIO dict with "waveform" [1, channels, samples] and "sample_rate"
    """
    # Try soundfile first
    try:
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        # soundfile returns (samples, channels) or (samples,) for mono
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]  # (samples, 1)
        # Convert to [1, channels, samples]
        waveform = torch.from_numpy(audio_data.T).unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        pass

    # Fallback to torchaudio
    try:
        import torchaudio

        buf = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buf, format=output_format)
        # torchaudio returns (channels, samples), we need (1, channels, samples)
        waveform = waveform.unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode audio. Install soundfile or torchaudio. Error: {e}"
        )


def comfy_to_audio_bytes(audio: dict, fmt: str = "wav") -> bytes:
    """Convert ComfyUI AUDIO dict to WAV bytes.

    Args:
        audio: ComfyUI AUDIO dict with "waveform" and "sample_rate"
        fmt: Output format (currently only "wav" supported)

    Returns:
        WAV-encoded bytes
    """
    waveform = audio["waveform"]  # [1, channels, samples]
    sample_rate = audio["sample_rate"]

    # Remove batch dimension
    if waveform.dim() == 3:
        waveform = waveform[0]  # [channels, samples]

    # Try soundfile first
    try:
        import soundfile as sf

        buf = io.BytesIO()
        # soundfile expects (samples, channels)
        data = waveform.cpu().numpy().T
        sf.write(buf, data, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception:
        pass

    # Fallback: manual WAV encoding
    data = waveform.cpu().numpy()
    channels = data.shape[0]
    samples = data.shape[1]

    # Convert float32 to int16
    int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    # Interleave channels
    interleaved = np.empty(channels * samples, dtype=np.int16)
    for ch in range(channels):
        interleaved[ch::channels] = int_data[ch]

    raw_bytes = interleaved.tobytes()
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2

    buf = io.BytesIO()
    # RIFF header
    data_size = len(raw_bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, 16))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_bytes)

    return buf.getvalue()
