"""API key authentication base classes for ComfyUI nodes."""

import os


class BaseAPIKeyNode:
    """Single API key authentication node with env-var fallback.

    Subclasses should set:
        ENV_VAR_NAME: str  - environment variable name (e.g. "ELEVENLABS_API_KEY")
        SERVICE_NAME: str  - display name (e.g. "ElevenLabs")
    """

    ENV_VAR_NAME: str = ""
    SERVICE_NAME: str = "API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": (
                        f"Your {cls.SERVICE_NAME} API key. "
                        f"Leave blank to use {cls.ENV_VAR_NAME} environment variable."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "provide_key"
    CATEGORY = "Kling AI/Auth"

    def provide_key(self, api_key: str = "") -> tuple:
        """Return the API key from input or environment variable."""
        key = api_key.strip() if api_key else ""
        if not key:
            key = os.environ.get(self.ENV_VAR_NAME, "").strip()
        if not key:
            raise ValueError(
                f"No {self.SERVICE_NAME} API key provided. "
                f"Enter it in the node or set the {self.ENV_VAR_NAME} environment variable."
            )
        return (key,)


class DualKeyAPIKeyNode:
    """Dual API key authentication node (access_key + secret_key) with env-var fallback.

    Subclasses should set:
        ENV_VAR_ACCESS: str  - env var for access key (e.g. "KLING_ACCESS_KEY")
        ENV_VAR_SECRET: str  - env var for secret key (e.g. "KLING_SECRET_KEY")
        SERVICE_NAME: str    - display name (e.g. "Kling AI")
    """

    ENV_VAR_ACCESS: str = ""
    ENV_VAR_SECRET: str = ""
    SERVICE_NAME: str = "API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "access_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": (
                        f"Your {cls.SERVICE_NAME} access key. "
                        f"Leave blank to use {cls.ENV_VAR_ACCESS} environment variable."
                    ),
                }),
                "secret_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": (
                        f"Your {cls.SERVICE_NAME} secret key. "
                        f"Leave blank to use {cls.ENV_VAR_SECRET} environment variable."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("access_key", "secret_key")
    FUNCTION = "provide_keys"
    CATEGORY = "Kling AI/Auth"

    def provide_keys(self, access_key: str = "", secret_key: str = "") -> tuple:
        """Return the access and secret keys from input or environment variables."""
        ak = access_key.strip() if access_key else ""
        sk = secret_key.strip() if secret_key else ""

        if not ak:
            ak = os.environ.get(self.ENV_VAR_ACCESS, "").strip()
        if not sk:
            sk = os.environ.get(self.ENV_VAR_SECRET, "").strip()

        if not ak:
            raise ValueError(
                f"No {self.SERVICE_NAME} access key provided. "
                f"Enter it in the node or set the {self.ENV_VAR_ACCESS} environment variable."
            )
        if not sk:
            raise ValueError(
                f"No {self.SERVICE_NAME} secret key provided. "
                f"Enter it in the node or set the {self.ENV_VAR_SECRET} environment variable."
            )

        return (ak, sk)
