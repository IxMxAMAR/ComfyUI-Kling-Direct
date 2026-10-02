"""Request/response shapes against the Kling API reference. Every HTTP call is
mocked -- no real Kling API calls are made."""

from unittest.mock import patch, MagicMock

import pytest
import torch

import kling_nodes as kn
import kling_client
from kling_client import KlingClient, KlingAPIError


def _client():
    c = KlingClient("ak", "sk")
    c._create_task = MagicMock(return_value="task-1")
    return c


def _payload(c):
    return c._create_task.call_args[0][1]


def _endpoint(c):
    return c._create_task.call_args[0][0]


def _fake_response(body, status=200):
    r = MagicMock(status_code=status, headers={}, text="")
    r.json.return_value = body
    return r


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def test_api_key_is_sent_as_bearer_instead_of_jwt():
    c = KlingClient("ak", "sk", api_key="KEY123")
    assert c._get_headers()["Authorization"] == "Bearer KEY123"


def test_without_api_key_a_jwt_is_sent():
    token = KlingClient("ak", "sk")._get_headers()["Authorization"].split(" ", 1)[1]
    assert token.count(".") == 2


def test_client_cache_separates_api_keys():
    a = kling_client.get_client("ak", "sk", api_key="A")
    b = kling_client.get_client("ak", "sk", api_key="B")
    assert a is not b and b.api_key == "B"


def test_auth_node_accepts_api_key_without_access_secret(monkeypatch):
    for var in ("KLING_ACCESS_KEY", "KLING_SECRET_KEY", "KLING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    (auth,) = kn.KlingDirect_Auth().execute(api_key=" KEY ")
    assert auth["api_key"] == "KEY"


def test_auth_node_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("KLING_API_KEY", "ENVKEY")
    (auth,) = kn.KlingDirect_Auth().execute(access_key="ak", secret_key="sk")
    assert auth["api_key"] == "ENVKEY" and auth["access_key"] == "ak"


def test_auth_node_without_any_key_still_raises(monkeypatch):
    for var in ("KLING_ACCESS_KEY", "KLING_SECRET_KEY", "KLING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="access key"):
        kn.KlingDirect_Auth().execute()


def test_auth_node_api_key_input_is_last():
    inputs = kn.KlingDirect_Auth.INPUT_TYPES()
    assert list(inputs["optional"]) == ["api_key"]
    assert list(inputs["required"]) == ["access_key", "secret_key", "debug"]


def test_make_client_passes_api_key():
    c = kn._make_client({"access_key": "ak", "secret_key": "sk", "api_key": "KEY9"})
    assert c.api_key == "KEY9"


# --------------------------------------------------------------------------
# Error codes
# --------------------------------------------------------------------------

def test_invalid_parameter_error_is_not_retried():
    c = KlingClient("ak", "sk")
    bad = _fake_response({"code": 1201, "message": "bad value"})
    with patch.object(c._session, "post", return_value=bad) as post, patch("time.sleep"):
        with pytest.raises(KlingAPIError, match="1201"):
            c._request("POST", "/v1/videos/text2video", {"prompt": "x"}, retries=3)
    assert post.call_count == 1


def test_concurrency_limit_error_is_retried():
    c = KlingClient("ak", "sk")
    busy = _fake_response({"code": 1303, "message": "parallel task over resource pack limit"}, status=429)
    ok = _fake_response({"code": 0, "data": {"task_id": "t"}})
    with patch.object(c._session, "post", side_effect=[busy, ok]) as post, patch("time.sleep"):
        res = c._request("POST", "/v1/videos/text2video", {"prompt": "x"}, retries=3)
    assert res["data"]["task_id"] == "t" and post.call_count == 2


# --------------------------------------------------------------------------
# Video payloads
# --------------------------------------------------------------------------

def test_text_to_video_single_shot_omits_multi_shot_fields():
    c = _client()
    c.text_to_video("kling-v3", "p", "16:9", "5")
    data = _payload(c)
    assert data["duration"] == "5" and data["mode"] == "pro"
    assert not {"multi_shot", "shot_type", "multi_prompt"} & {k for k, v in data.items() if v is not None}


def test_text_to_video_customize_multi_shot_payload():
    c = _client()
    shots = [{"index": 1, "prompt": "a", "duration": "3"}, {"index": 2, "prompt": "b", "duration": "2"}]
    c.text_to_video("kling-v3", "p", "16:9", "5", shot_type="customize", multi_prompt=shots)
    data = _payload(c)
    assert data["multi_shot"] is True and data["shot_type"] == "customize" and data["multi_prompt"] == shots


def test_image_to_video_multi_shot_payload():
    c = _client()
    shots = [{"index": 1, "prompt": "a", "duration": "5"}]
    c.image_to_video("kling-v3", "b64", "5", shot_type="customize", multi_prompt=shots, mode="4k")
    data = _payload(c)
    assert data["multi_shot"] is True and data["multi_prompt"] == shots and data["mode"] == "4k"


def test_omni_video_payload_has_elements_and_multi_shot_flag():
    c = _client()
    c.omni_video("kling-v3-omni", "p", [{"image_url": "x"}], [], "16:9", "5", shot_type="intelligence", elements=[{"element_id": 7}])
    data = _payload(c)
    assert data["multi_shot"] is True and data["shot_type"] == "intelligence"
    assert data["element_list"] == [{"element_id": 7}] and data["image_list"] == [{"image_url": "x"}]


def test_omni_video_without_shots_sends_multi_shot_false():
    c = _client()
    c.omni_video("kling-v3-omni", "p", [], [], "16:9", "5")
    assert _payload(c)["multi_shot"] is False


def test_motion_control_uses_image_url_key():
    c = _client()
    c.motion_control("kling-v2-6", "b64", "https://v/x.mp4", keep_original_sound="no")
    data = _payload(c)
    assert data["image_url"] == "b64" and "image" not in data and data["keep_original_sound"] == "no"
    assert _endpoint(c) == "/v1/videos/motion-control"


def test_advanced_lip_sync_payload_has_sound_times():
    c = _client()
    c.advanced_lip_sync("sess", "face", "https://a/x.mp3", 0, 4000, 500, volume=1.5)
    face = _payload(c)["face_choose"][0]
    assert face["sound_start_time"] == 0 and face["sound_end_time"] == 4000 and face["sound_insert_time"] == 500
    assert face["sound_volume"] == 1.5 and face["sound_file"] == "https://a/x.mp3"


def test_avatar_sends_base64_audio_as_sound_file():
    c = _client()
    c.avatar("img", audio_b64="AUDIO", prompt="hi")
    data = _payload(c)
    assert _endpoint(c) == "/v1/videos/avatar/image2video"
    assert data["sound_file"] == "AUDIO" and "audio_id" not in {k for k, v in data.items() if v is not None}


def test_video_effects_single_and_dual_image_payloads():
    c = _client()
    c.video_effects("pet_dance", ["a"])
    assert _payload(c) == {"effect_scene": "pet_dance", "input": {"image": "a"}}
    c.video_effects("hug_pro", ["a", "b"])
    assert _payload(c) == {"effect_scene": "hug_pro", "input": {"images": ["a", "b"]}}


# --------------------------------------------------------------------------
# Image / audio / voice / element payloads
# --------------------------------------------------------------------------

def test_image_generation_does_not_send_fidelity():
    c = _client()
    c.image_generation("kling-v2-1", "p", "21:9", 2, "2K")
    data = _payload(c)
    assert "fidelity" not in data and data["resolution"] == "2k" and data["aspect_ratio"] == "21:9"


def test_extend_image_payload():
    c = _client()
    c.extend_image("b64", 0.0, 0.0, 0.25, 0.25, prompt="more sky")
    data = _payload(c)
    assert data["image"] == "b64" and data["left_expansion_ratio"] == 0.25 and data["up_expansion_ratio"] == 0.0
    assert "image_id" not in data and "aspect_ratio" not in data
    assert _endpoint(c) == "/v1/images/editing/expand"


def test_image_recognize_is_synchronous_on_documented_endpoint():
    c = KlingClient("ak", "sk")
    body = {"code": 0, "data": {"task_result": {"images": [{"type": "face_seg", "is_contain": True, "url": "u"}]}}}
    with patch.object(c, "_request", return_value=body) as req:
        data = c.image_recognize("b64")
    assert req.call_args[0][:2] == ("POST", "/v1/videos/image-recognize")
    assert data["task_result"]["images"][0]["type"] == "face_seg"


def test_video_to_audio_payload_drops_unset_prompts():
    c = _client()
    c.video_to_audio("https://v/x.mp4", None, "calm piano", True)
    data = _payload(c)
    assert data["bgm_prompt"] == "calm piano" and data["asmr_mode"] is True


def test_tts_is_synchronous_and_returns_task_result():
    c = KlingClient("ak", "sk")
    body = {"code": 0, "data": {"task_id": "t", "task_result": {"audios": [{"id": "a", "url": "https://x/a.mp3"}]}}}
    with patch.object(c, "_request", return_value=body) as req:
        data = c.tts("hello", "voice", 1.0, "en")
    assert req.call_args[0][:2] == ("POST", "/v1/audio/tts")
    assert kn._extract_audio_url(data) == "https://x/a.mp3"


def test_create_voice_payload():
    c = _client()
    assert c.create_voice("mine", "https://x/v.mp3") == "task-1"
    assert _endpoint(c) == "/v1/general/custom-voices"
    assert _payload(c) == {"voice_name": "mine", "voice_url": "https://x/v.mp3"}


def test_create_element_payload():
    c = _client()
    c.create_element("hero", "a hero", "front", ["r1", "r2"])
    assert _endpoint(c) == "/v1/general/advanced-custom-elements"
    data = _payload(c)
    assert data["reference_type"] == "image_refer"
    assert data["element_image_list"] == {"frontal_image": "front", "refer_images": [{"image_url": "r1"}, {"image_url": "r2"}]}


def test_subject_completion_payload():
    c = _client()
    c.subject_completion("front")
    assert _endpoint(c) == "/v1/general/ai-multi-shot"
    assert _payload(c) == {"element_frontal_image": "front"}


def test_account_costs_queries_documented_endpoint_with_time_range():
    c = KlingClient("ak", "sk")
    with patch.object(c, "_request", return_value={"code": 0, "data": {"resource_pack_subscribe_infos": []}}) as req:
        c.account_costs()
    method, path = req.call_args[0][:2]
    assert method == "GET" and path.startswith("/account/costs?start_time=") and "&end_time=" in path


# --------------------------------------------------------------------------
# Kling 3.0 Turbo / unified /tasks API
# --------------------------------------------------------------------------

def test_turbo_text_to_video_payload():
    c = _client()
    c.turbo_text_to_video("a fox", "1080p", "9:16", 7)
    assert _endpoint(c) == "/text-to-video/kling-3.0-turbo"
    assert _payload(c) == {"prompt": "a fox", "settings": {"resolution": "1080p", "aspect_ratio": "9:16", "duration": 7}}


def test_turbo_image_to_video_payload_with_and_without_prompt():
    c = _client()
    c.turbo_image_to_video("b64", "pan left", "720p", 5)
    assert _endpoint(c) == "/image-to-video/kling-3.0-turbo"
    assert _payload(c)["contents"] == [{"type": "prompt", "text": "pan left"}, {"type": "first_frame", "url": "b64"}]
    c.turbo_image_to_video("b64", "", "720p", 5)
    assert _payload(c)["contents"] == [{"type": "first_frame", "url": "b64"}]


def test_create_task_accepts_unified_api_id():
    c = KlingClient("ak", "sk")
    with patch.object(c, "_request", return_value={"code": 0, "data": {"id": "u-1", "status": "submitted"}}):
        assert c._create_task("/text-to-video/kling-3.0-turbo", {}) == "u-1"


def test_poll_task_normalizes_unified_tasks_response():
    c = KlingClient("ak", "sk")
    seq = [
        {"data": [{"id": "u-1", "status": "processing"}]},
        {"data": [{"id": "u-1", "status": "succeeded", "outputs": [{"type": "video", "id": "v", "url": "https://x/v.mp4"}]}]},
    ]
    with patch.object(c, "_request", side_effect=seq) as req, patch("time.sleep"):
        res = c.poll_task("/tasks", "u-1", timeout=60)
    assert req.call_args[0][1] == "/tasks?task_ids=u-1"
    assert kn._extract_video_url(res) == "https://x/v.mp4"


def test_poll_task_unified_failure_reports_message():
    c = KlingClient("ak", "sk")
    body = {"data": [{"id": "u-1", "status": "failed", "message": "content blocked"}]}
    with patch.object(c, "_request", return_value=body), patch("time.sleep"):
        with pytest.raises(KlingAPIError, match="content blocked"):
            c.poll_task("/tasks", "u-1", timeout=60)


def test_get_task_status_supports_unified_tasks():
    c = KlingClient("ak", "sk")
    body = {"data": [{"id": "u-1", "status": "processing"}]}
    with patch.object(c, "_request", return_value=body):
        assert c.get_task_status("/tasks", "u-1")["data"]["task_status"] == "processing"


# --------------------------------------------------------------------------
# Node helpers
# --------------------------------------------------------------------------

def test_extract_image_url_reads_task_result_images():
    res = {"task_result": {"images": [{"index": 0, "url": "https://x/0.png"}, {"index": 1, "url": "https://x/1.png"}]}}
    assert kn._extract_image_url(res) == "https://x/0.png"
    assert kn._extract_image_url(res, 1) == "https://x/1.png"


def test_extract_image_url_without_images_raises():
    with pytest.raises(Exception, match="no images"):
        kn._extract_image_url({"task_result": {}})


def test_extract_audio_url_prefers_mp3():
    res = {"task_result": {"audios": [{"url_mp3": "https://x/a.mp3", "url_wav": "https://x/a.wav"}]}}
    assert kn._extract_audio_url(res) == "https://x/a.mp3"


def test_extract_audio_url_falls_back_to_url():
    assert kn._extract_audio_url({"task_result": {"audios": [{"url": "https://x/t.mp3"}]}}) == "https://x/t.mp3"


def test_extract_audio_url_without_audios_raises():
    with pytest.raises(Exception, match="no audios"):
        kn._extract_audio_url({"task_result": {}})


def test_build_multi_shot_from_shot_list():
    shot_type, shots = kn._build_multi_shot("3|a dog runs\n\n2| it stops ", "natural")
    assert shot_type == "customize"
    assert shots == [{"index": 1, "prompt": "a dog runs", "duration": "3"}, {"index": 2, "prompt": "it stops", "duration": "2"}]


def test_build_multi_shot_intelligence_and_none():
    assert kn._build_multi_shot("", "intelligence") == ("intelligence", None)
    assert kn._build_multi_shot("", "natural") == (None, None)
    assert kn._build_multi_shot("", "") == (None, None)


def test_build_multi_shot_rejects_malformed_line():
    with pytest.raises(ValueError, match="seconds\\|prompt"):
        kn._build_multi_shot("a dog runs")


def test_element_list_parsing():
    assert kn._element_list("12, 34  56") == [{"element_id": 12}, {"element_id": 34}, {"element_id": 56}]
    assert kn._element_list("") == []


def test_expansion_ratios_widen_and_heighten():
    up, down, left, right = kn._expansion_ratios(1000, 1000, "2:1")
    assert (up, down) == (0.0, 0.0) and left == right == pytest.approx(0.5)
    up, down, left, right = kn._expansion_ratios(1000, 1000, "1:2")
    assert (left, right) == (0.0, 0.0) and up == down == pytest.approx(0.5)


def test_camera_axis_type_becomes_simple_with_single_axis():
    (cam,) = kn.KlingDirect_CameraControl().execute(type="zoom", horizontal=1.0, vertical=2.0, pan=3.0, tilt=0.0, roll=0.0, zoom=4.0)
    assert cam == {"type": "simple", "config": {"horizontal": 0.0, "vertical": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0, "zoom": 4.0}}


def test_camera_predefined_move_has_no_config():
    (cam,) = kn.KlingDirect_CameraControl().execute(type="down_back", horizontal=0.0, vertical=0.0, pan=0.0, tilt=0.0, roll=0.0, zoom=0.0)
    assert cam == {"type": "down_back"}


def test_camera_simple_keeps_all_axes():
    (cam,) = kn.KlingDirect_CameraControl().execute(type="simple", horizontal=5.0, vertical=0.0, pan=0.0, tilt=0.0, roll=0.0, zoom=0.0)
    assert cam["type"] == "simple" and cam["config"]["horizontal"] == 5.0


def test_all_camera_presets_emit_documented_type():
    for name in kn.CAMERA_PRESETS:
        (cam,) = kn.KlingDirect_CameraPreset().build(name)
        assert cam["type"] == "simple"


def test_china_region_uses_beijing_gateway():
    assert kn.KLING_REGIONS["china"] == "https://api-beijing.klingai.com"


# --------------------------------------------------------------------------
# Node interface compatibility: new inputs are appended, defaults unchanged
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls, new_optional", [
    (kn.KlingDirect_TextToVideo, ["camera_control", "shot_list"]),
    (kn.KlingDirect_ImageToVideo, ["image_tail", "camera_control", "shot_list"]),
    (kn.KlingDirect_VideoOmni, ["image_1", "image_2", "video_url", "sound", "element_ids", "shot_list", "video_refer_type", "keep_original_sound"]),
    (kn.KlingDirect_AdvancedLipSync, ["sound_start_time", "sound_end_time", "sound_insert_time"]),
    (kn.KlingDirect_ImageOmni, ["model_name", "element_ids"]),
    (kn.KlingDirect_ImageExtend, ["image", "up_expansion_ratio", "down_expansion_ratio", "left_expansion_ratio", "right_expansion_ratio"]),
    (kn.KlingDirect_VideoToAudio, ["sound_effect_prompt", "bgm_prompt", "asmr_mode"]),
    (kn.KlingDirect_MotionControl, ["keep_original_sound"]),
    (kn.KlingDirect_VoiceClone, ["audio", "audio_url", "voice_name"]),
])
def test_new_inputs_are_optional_and_last(cls, new_optional):
    assert list(cls.INPUT_TYPES()["optional"]) == new_optional


def test_text_to_video_defaults_unchanged():
    req = kn.KlingDirect_TextToVideo.INPUT_TYPES()["required"]
    assert req["duration"][1]["default"] == "5" and req["duration"][0] == kn.VIDEO_DURATIONS
    assert req["mode"][1]["default"] == "pro" and "4k" in req["mode"][0]
    assert req["shot_type"][1]["default"] == "natural"


def test_video_durations_cover_documented_range():
    assert kn.VIDEO_DURATIONS == [str(d) for d in range(3, 16)]


def test_motion_control_models_match_documented_enum():
    req = kn.KlingDirect_MotionControl.INPUT_TYPES()["required"]
    assert req["model_name"][0] == ["kling-v2-6", "kling-v3"] and req["model_name"][1]["default"] == "kling-v2-6"


def test_image_aspect_ratio_choices_include_21_9_and_auto():
    assert "21:9" in kn.KlingDirect_ImageGen.INPUT_TYPES()["required"]["aspect_ratio"][0]
    omni = kn.KlingDirect_ImageOmni.INPUT_TYPES()["required"]
    assert "auto" in omni["aspect_ratio"][0] and "4k" in omni["resolution"][0]


def test_new_models_and_choices_are_exposed():
    assert "kling-v2-1" in kn.KlingDirect_ImageGen.INPUT_TYPES()["required"]["model_name"][0]
    assert "kolors-virtual-try-on-v1-5" in kn.KlingDirect_VirtualTryOn.INPUT_TYPES()["required"]["model_name"][0]
    assert "kling-v2-5-turbo" in kn.KlingDirect_ImageToVideo.INPUT_TYPES()["required"]["model_name"][0]
    assert "intelligence" in kn.KlingDirect_TextToVideo.INPUT_TYPES()["required"]["shot_type"][0]


def test_new_nodes_are_registered_with_prefix():
    new = ["KlingDirect_TurboTextToVideo", "KlingDirect_TurboImageToVideo", "KlingDirect_CreateElement", "KlingDirect_SubjectAngles"]
    for key in new:
        assert key in kn.NODE_CLASS_MAPPINGS and key in kn.NODE_DISPLAY_NAME_MAPPINGS


# --------------------------------------------------------------------------
# Node execution with a mocked client
# --------------------------------------------------------------------------

AUTH = {"access_key": "ak", "secret_key": "sk"}
IMG = torch.rand((1, 320, 320, 3))


@pytest.fixture
def client():
    c = MagicMock()
    with patch.object(kn, "_make_client", return_value=c):
        yield c


@pytest.fixture
def no_video_io():
    with patch.object(kn, "download_to_output", return_value=("/tmp/v.mp4", "v.mp4")), \
         patch.object(kn, "load_video_to_tensor", return_value=torch.zeros((1, 8, 8, 3))), \
         patch.object(kn, "load_audio_to_tensor", return_value=kn._empty_audio()):
        yield


VIDEO_RES = {"task_result": {"videos": [{"url": "https://x/v.mp4"}]}}


def test_text_to_video_node_sends_storyboard(client, no_video_io):
    client.text_to_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_TextToVideo().generate(AUTH, "p", "", "kling-v3", "16:9", "5", "pro", True, 0.5, shot_list="3|a\n2|b")
    kwargs = client.text_to_video.call_args.kwargs
    assert kwargs["shot_type"] == "customize" and len(kwargs["multi_prompt"]) == 2


def test_text_to_video_node_ignores_legacy_shot_types(client, no_video_io):
    client.text_to_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_TextToVideo().generate(AUTH, "p", "", "kling-v3", "16:9", "5", "pro", True, 0.5, shot_type="close_up")
    kwargs = client.text_to_video.call_args.kwargs
    assert kwargs["shot_type"] is None and kwargs["multi_prompt"] is None


def test_text_to_video_node_intelligence_shot_type(client, no_video_io):
    client.text_to_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_TextToVideo().generate(AUTH, "p", "", "kling-v3", "16:9", "5", "pro", True, 0.5, shot_type="intelligence")
    assert client.text_to_video.call_args.kwargs["shot_type"] == "intelligence"


def test_video_omni_node_uses_image_url_and_element_ids(client, no_video_io):
    client.omni_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_VideoOmni().generate(AUTH, "p", "kling-v3-omni", "5", "16:9", "pro", image_1=IMG, element_ids="11,22")
    args, kwargs = client.omni_video.call_args
    images = args[2]
    assert list(images[0]) == ["image_url"]
    assert kwargs["elements"] == [{"element_id": 11}, {"element_id": 22}] and kwargs["sound"] == "on"


def test_video_omni_node_turns_sound_off_for_reference_video(client, no_video_io):
    client.omni_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_VideoOmni().generate(AUTH, "p", "kling-video-o1", "5", "16:9", "pro", video_url="https://v/x.mp4",
                                        video_refer_type="feature", keep_original_sound="yes")
    args, kwargs = client.omni_video.call_args
    assert args[3] == [{"video_url": "https://v/x.mp4", "refer_type": "feature", "keep_original_sound": "yes"}]
    assert kwargs["sound"] == "off"


def test_video_omni_node_default_keep_sound_is_omitted(client, no_video_io):
    client.omni_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_VideoOmni().generate(AUTH, "p", "kling-video-o1", "5", "16:9", "pro", video_url="https://v/x.mp4")
    assert client.omni_video.call_args[0][3][0]["keep_original_sound"] is None


def test_motion_control_node_passes_keep_original_sound(client, no_video_io):
    client.motion_control.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_MotionControl().generate(AUTH, IMG, "https://v/x.mp4", keep_original_sound="no")
    assert client.motion_control.call_args.kwargs["keep_original_sound"] == "no"
    kn.KlingDirect_MotionControl().generate(AUTH, IMG, "https://v/x.mp4")
    assert client.motion_control.call_args.kwargs["keep_original_sound"] is None


def test_avatar_node_sends_audio_inline_without_materials_upload(client, no_video_io):
    client.avatar.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    audio = {"waveform": torch.zeros((1, 1, 32000)), "sample_rate": 16000}
    kn.KlingDirect_AvatarGen().generate(AUTH, IMG, "", "pro", audio=audio)
    assert client.avatar.call_args.kwargs["audio_b64"]
    client.upload_asset.assert_not_called()


def test_advanced_lip_sync_node_measures_audio_and_uses_face_start(client, no_video_io, tmp_path):
    client.identify_face.return_value = {"session_id": "s", "face_data": [{"face_id": "f", "start_time": 1500, "end_time": 9000}]}
    client.advanced_lip_sync.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    clip = {"waveform": torch.zeros((1, 1, 16000 * 5)), "sample_rate": 16000}
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")

    def fake_download(url, ext="mp4", **kwargs):
        return (str(audio), "a.mp3") if ext == "mp3" else ("/tmp/v.mp4", "v.mp4")

    with patch.object(kn, "download_to_output", side_effect=fake_download), \
         patch.object(kn, "load_audio_to_tensor", return_value=clip):
        kn.KlingDirect_AdvancedLipSync().generate(AUTH, "https://v/x.mp4", "https://a/x.mp3", volume=10)
    args, kwargs = client.advanced_lip_sync.call_args
    assert args[3:6] == (0, 5000, 1500) and kwargs["volume"] == 1.0
    assert not audio.exists()


def test_advanced_lip_sync_node_uses_explicit_times_and_caps_volume(client, no_video_io):
    client.identify_face.return_value = {"session_id": "s", "face_data": [{"face_id": "f", "start_time": 1500}]}
    client.advanced_lip_sync.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_AdvancedLipSync().generate(AUTH, "https://v/x.mp4", "https://a/x.mp3", volume=100,
                                              sound_start_time=200, sound_end_time=4000, sound_insert_time=300)
    args, kwargs = client.advanced_lip_sync.call_args
    assert args[3:6] == (200, 4000, 300) and kwargs["volume"] == 2.0


def test_advanced_lip_sync_node_rejects_unmeasurable_audio(client, tmp_path):
    client.identify_face.return_value = {"session_id": "s", "face_data": [{"face_id": "f"}]}
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")
    with patch.object(kn, "download_to_output", return_value=(str(audio), "a.mp3")), \
         patch.object(kn, "load_audio_to_tensor", return_value=kn._empty_audio()):
        with pytest.raises(ValueError, match="sound_end_time"):
            kn.KlingDirect_AdvancedLipSync().generate(AUTH, "https://v/x.mp4", "https://a/x.mp3")
    assert not audio.exists()


def test_image_gen_node_downloads_every_image(client):
    client.image_generation.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}, {"url": "https://x/1.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))) as dl:
        out, url, _ = kn.KlingDirect_ImageGen().generate(AUTH, "p", "", "kling-v3", "1:1", "1k", 0.5, n=2)
    assert out.shape[0] == 2 and url == "https://x/0.png" and dl.call_count == 2


def test_image_omni_node_selects_model_and_elements(client):
    client.omni_image.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))):
        kn.KlingDirect_ImageOmni().generate(AUTH, "p", IMG, "auto", "4k", model_name="kling-v3-omni", element_ids="5")
    args, kwargs = client.omni_image.call_args
    assert args[0] == "kling-v3-omni" and kwargs["elements"] == [{"element_id": 5}] and kwargs["resolution"] == "4k"


def test_image_omni_node_default_model_unchanged(client):
    client.omni_image.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))):
        kn.KlingDirect_ImageOmni().generate(AUTH, "p", IMG, "1:1", "1k")
    assert client.omni_image.call_args[0][0] == "kling-image-o1"


def test_image_extend_node_derives_ratios_from_aspect_ratio(client):
    client.extend_image.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}]}}
    img = torch.rand((1, 400, 400, 3))
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))):
        kn.KlingDirect_ImageExtend().generate(AUTH, "", "sky", "2:1", image=img)
    args, kwargs = client.extend_image.call_args
    assert args[1:5] == (0.0, 0.0, pytest.approx(0.5), pytest.approx(0.5)) and kwargs["prompt"] == "sky"


def test_image_extend_node_explicit_ratios_win(client):
    client.extend_image.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))):
        kn.KlingDirect_ImageExtend().generate(AUTH, "https://i/x.png", "", "2:1", up_expansion_ratio=0.3)
    args, _ = client.extend_image.call_args
    assert args[0] == "https://i/x.png" and args[1:5] == (0.3, 0.0, 0.0, 0.0)


def test_image_extend_node_needs_a_source_image(client):
    with pytest.raises(ValueError, match="image"):
        kn.KlingDirect_ImageExtend().generate(AUTH, "", "", "2:1")


def test_text_to_audio_node_downloads_mp3(client):
    client.text_to_audio.return_value = "t"
    client.poll_task.return_value = {"task_result": {"audios": [{"url_mp3": "https://x/a.mp3", "url_wav": "https://x/a.wav"}]}}
    with patch.object(kn, "download_to_output", return_value=("/tmp/a.mp3", "a.mp3")) as dl, \
         patch.object(kn, "load_audio_to_tensor", return_value=kn._empty_audio()):
        _, _, url, _ = kn.KlingDirect_AudioGenerate().generate(AUTH, "rain", 5)
    assert url == "https://x/a.mp3" and dl.call_args[0][0] == "https://x/a.mp3"


def test_tts_nodes_use_synchronous_result(client):
    client.tts.return_value = {"task_id": "tt", "task_result": {"audios": [{"url": "https://x/t.mp3"}]}}
    with patch.object(kn, "download_to_output", return_value=("/tmp/t.mp3", "t.mp3")), \
         patch.object(kn, "load_audio_to_tensor", return_value=kn._empty_audio()):
        _, _, url, task_id = kn.KlingDirect_TTS().generate(AUTH, "hi", "voice")
        assert (url, task_id) == ("https://x/t.mp3", "tt")
        _, _, url, task_id = kn.KlingDirect_TTSAdvanced().generate(AUTH, "hi", "voice", 1.0)
        assert (url, task_id) == ("https://x/t.mp3", "tt")
    client.poll_task.assert_not_called()


def test_video_to_audio_node_forwards_prompts(client):
    client.video_to_audio.return_value = "t"
    client.poll_task.return_value = {"task_result": {"audios": [{"url_mp3": "https://x/a.mp3"}]}}
    with patch.object(kn, "download_audio_to_tensor", return_value=kn._empty_audio()):
        kn.KlingDirect_VideoToAudio().generate(AUTH, "https://v/x.mp4", sound_effect_prompt="rain", bgm_prompt="", asmr_mode=True)
    assert client.video_to_audio.call_args[0] == ("https://v/x.mp4", "rain", None, True)


def test_video_to_audio_node_defaults_send_nothing_extra(client):
    client.video_to_audio.return_value = "t"
    client.poll_task.return_value = {"task_result": {"audios": [{"url_mp3": "https://x/a.mp3"}]}}
    with patch.object(kn, "download_audio_to_tensor", return_value=kn._empty_audio()):
        kn.KlingDirect_VideoToAudio().generate(AUTH, "https://v/x.mp4")
    assert client.video_to_audio.call_args[0] == ("https://v/x.mp4", None, None, None)


def test_voice_clone_node_creates_custom_voice_from_url(client):
    client.create_voice.return_value = "t"
    client.poll_task.return_value = {"task_result": {"voices": [{"voice_id": "v-9", "voice_name": "mine"}]}}
    (voice_id,) = kn.KlingDirect_VoiceClone().clone(AUTH, audio_url=" https://x/v.mp3 ", voice_name="mine")
    assert voice_id == "v-9"
    assert client.create_voice.call_args[0] == ("mine", "https://x/v.mp3")
    assert client.poll_task.call_args[0][0] == "/v1/general/custom-voices"


def test_voice_clone_node_requires_a_url(client):
    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    with pytest.raises(ValueError, match="audio_url"):
        kn.KlingDirect_VoiceClone().clone(AUTH, audio=audio)


def test_video_effects_node_sends_only_images(client, no_video_io):
    client.video_effects.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_VideoEffects().generate(AUTH, IMG, "hug_pro", "kling-v1", "5", "std", image_2=IMG)
    args = client.video_effects.call_args[0]
    assert args[0] == "hug_pro" and len(args[1]) == 2


def test_video_effects_default_scene_is_not_a_discontinued_effect():
    assert kn.KlingDirect_VideoEffects.INPUT_TYPES()["required"]["effect_scene"][1]["default"] == "hug_pro"


def test_image_recognize_node_returns_segments_as_json(client):
    client.image_recognize.return_value = {"task_result": {"images": [{"type": "face_seg", "is_contain": True, "url": "u"}]}}
    description, task_id = kn.KlingDirect_ImageRecognize().recognize(AUTH, IMG)
    assert "face_seg" in description and task_id == ""


def test_health_check_uses_account_costs(client):
    client.account_costs.return_value = {}
    ok, status = kn.KlingDirect_ApiHealthCheck().check(AUTH)
    assert ok and "/account/costs" in status
    client.account_costs.side_effect = KlingAPIError("nope")
    ok, status = kn.KlingDirect_ApiHealthCheck().check(AUTH)
    assert not ok and "nope" in status


def test_turbo_text_to_video_node(client, no_video_io):
    client.turbo_text_to_video.return_value = "u-1"
    client.poll_task.return_value = VIDEO_RES
    out = kn.KlingDirect_TurboTextToVideo().generate({**AUTH, "api_key": "K"}, "a fox", "1080p", "9:16", 6)
    assert client.turbo_text_to_video.call_args[0] == ("a fox", "1080p", "9:16", 6)
    assert client.poll_task.call_args[0] == ("/tasks", "u-1")
    assert out[3] == "https://x/v.mp4" and out[4] == "u-1"


def test_turbo_image_to_video_node(client, no_video_io):
    client.turbo_image_to_video.return_value = "u-2"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_TurboImageToVideo().generate({**AUTH, "api_key": "K"}, IMG, "zoom in", "720p", 4)
    args = client.turbo_image_to_video.call_args[0]
    assert args[1:] == ("zoom in", "720p", 4) and isinstance(args[0], str)


def test_turbo_nodes_require_api_key(client):
    with pytest.raises(ValueError, match="API key"):
        kn.KlingDirect_TurboTextToVideo().generate(AUTH, "p", "720p", "16:9", 5)
    with pytest.raises(ValueError, match="API key"):
        kn.KlingDirect_TurboImageToVideo().generate(AUTH, IMG, "", "720p", 5)
    client.turbo_text_to_video.assert_not_called()


def test_create_element_node_returns_element_id(client):
    client.create_element.return_value = "t"
    client.poll_task.return_value = {"task_result": {"elements": [{"element_id": 829836802793406551}]}}
    element_id, task_id = kn.KlingDirect_CreateElement().create(AUTH, "hero", "a hero", IMG, IMG, refer_image_3=IMG)
    assert element_id == "829836802793406551" and task_id == "t"
    args = client.create_element.call_args[0]
    assert args[:2] == ("hero", "a hero") and len(args[3]) == 2


def test_subject_angles_node_collects_all_angle_urls(client):
    client.subject_completion.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"index": 0, "url_1": "https://x/1.png", "url_2": "https://x/2.png", "url_3": "https://x/3.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))) as dl:
        images, url, task_id = kn.KlingDirect_SubjectAngles().generate(AUTH, IMG)
    assert images.shape[0] == 3 and url == "https://x/1.png" and dl.call_count == 3
    assert client.poll_task.call_args[0][0] == "/v1/general/ai-multi-shot"


# --------------------------------------------------------------------------
# Multi-image to video, reference to image
# --------------------------------------------------------------------------

def test_multi_image_to_video_payload():
    c = _client()
    c.multi_image_to_video("two friends", ["a", "b"], "blur", "pro", "10", "9:16")
    data = _payload(c)
    assert _endpoint(c) == "/v1/videos/multi-image2video"
    assert data["model_name"] == "kling-v1-6" and data["image_list"] == [{"image": "a"}, {"image": "b"}]
    assert data["mode"] == "pro" and data["duration"] == "10" and data["aspect_ratio"] == "9:16"


def test_reference_to_image_payload():
    c = _client()
    c.reference_to_image("a hero", ["s1", "s2"], "scene", None, 3, "21:9")
    data = _payload(c)
    assert _endpoint(c) == "/v1/images/multi-image2image"
    assert data["model_name"] == "kling-v2-1" and data["subject_image_list"] == [{"subject_image": "s1"}, {"subject_image": "s2"}]
    assert data["scene_image"] == "scene" and data["style_image"] is None and data["n"] == 3


def test_multi_image_to_video_node(client, no_video_io):
    client.multi_image_to_video.return_value = "t"
    client.poll_task.return_value = VIDEO_RES
    kn.KlingDirect_MultiImageToVideo().generate(AUTH, "p", IMG, "", "std", "5", "16:9", image_3=IMG)
    args = client.multi_image_to_video.call_args[0]
    assert len(args[1]) == 2 and args[3:] == ("std", "5", "16:9")
    assert client.poll_task.call_args[0][0] == "/v1/videos/multi-image2video"


def test_reference_to_image_node(client):
    client.reference_to_image.return_value = "t"
    client.poll_task.return_value = {"task_result": {"images": [{"url": "https://x/0.png"}, {"url": "https://x/1.png"}]}}
    with patch.object(kn, "download_to_tensor", return_value=torch.zeros((1, 4, 4, 3))):
        out, url, _ = kn.KlingDirect_ReferenceToImage().generate(AUTH, "p", IMG, "1:1", 2, scene_image=IMG)
    args = client.reference_to_image.call_args[0]
    assert out.shape[0] == 2 and url == "https://x/0.png"
    assert len(args[1]) == 1 and isinstance(args[2], str) and args[3] is None and args[4:] == (2, "1:1")


def test_multi_image_nodes_are_registered():
    for key in ("KlingDirect_MultiImageToVideo", "KlingDirect_ReferenceToImage"):
        assert key in kn.NODE_CLASS_MAPPINGS and key in kn.NODE_DISPLAY_NAME_MAPPINGS
