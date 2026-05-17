"""Pytest configuration for ComfyUI-Kling-Direct tests.

Sets up the path and mocks `folder_paths` so the kling_nodes module loads
cleanly without a live ComfyUI environment. Every test runs in this
context — NO real Kling API calls are ever made.
"""

import os
import sys
from unittest.mock import MagicMock

# Resolve the package directory and add its parent so `import kling_client` /
# `import kling_nodes` work as if loaded by ComfyUI.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Mock folder_paths so kling_nodes can import without a real ComfyUI.
_mock_folder_paths = MagicMock()
_mock_folder_paths.get_output_directory.return_value = os.path.join(_HERE, "_test_output")
_mock_folder_paths.get_input_directory.return_value = os.path.join(_HERE, "_test_input")
_mock_folder_paths.get_annotated_filepath.side_effect = lambda x: x
os.makedirs(_mock_folder_paths.get_output_directory.return_value, exist_ok=True)
os.makedirs(_mock_folder_paths.get_input_directory.return_value, exist_ok=True)
sys.modules["folder_paths"] = _mock_folder_paths
