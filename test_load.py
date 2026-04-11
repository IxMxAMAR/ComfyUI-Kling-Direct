import sys
import os
from unittest.mock import MagicMock

# Mock ComfyUI modules BEFORE importing kling_nodes
mock_folder_paths = MagicMock()
mock_folder_paths.get_output_directory.return_value = "output"
mock_folder_paths.get_input_directory.return_value = "input"
mock_folder_paths.get_annotated_filepath.side_effect = lambda x: x
sys.modules["folder_paths"] = mock_folder_paths

# Add relevant paths
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

print(f"--- SIMULATING KLING NODE LOAD (MOCKED) ---")
try:
    import kling_nodes
    print(f"SUCCESS: Loaded {len(kling_nodes.NODE_CLASS_MAPPINGS)} nodes.")
    for name in ["KlingAvatarNode", "KlingVideoExtendNode", "KlingVirtualTryOnNode", "KlingImageGenerationNode"]:
        if name in kling_nodes.NODE_CLASS_MAPPINGS:
            print(f"REGISTERED: {name}")
        else:
            print(f"MISSING: {name}")
except Exception as e:
    import traceback
    traceback.print_exc()
