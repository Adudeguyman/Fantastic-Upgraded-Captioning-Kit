import unittest
import os
import shutil
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

from captioning_kit.llm_captioning import (
    AutoCaptionError,
    CaptioningSettings,
    ModelAssets,
    ModelJsonError,
    build_llama_server_command,
    bbox_target_indices,
    bbox_target_indices_with_reasons,
    bbox_xyxy_to_yxyx,
    chat_text,
    chat_vision,
    ensure_model_assets,
    ensure_server_running,
    extract_json,
    format_prompt,
    generate_json_from_image,
    generate_json_refinement,
    json_system_prompt,
    load_model_profiles,
    load_prompts,
    parse_batch_bboxes,
    parse_batch_bboxes_with_reasons,
    parse_json_with_repair,
    request_user_prompt,
    runtime_config_for_task,
    safe_repo_dir,
    server_host_port,
    server_model_ids,
    should_try_bbox,
    strip_thinking_output,
    write_default_prompts,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content, finish_reason="stop", model="fake-model"):
        self.choices = [FakeChoice(content, finish_reason)]
        self.model = model


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response left.")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = self


class LlmCaptioningTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        parsed = extract_json('```json\n{"high_level_description":"sign"}\n```')
        self.assertEqual(parsed["high_level_description"], "sign")

    def test_repairs_malformed_json_once(self):
        progress_messages = []
        with patch(
            "captioning_kit.llm_captioning.chat_text",
            return_value='{"high_level_description":"sign"}',
        ) as chat:
            parsed = parse_json_with_repair(
                CaptioningSettings(caption_model="repair-model"),
                "caption",
                '{"high_level_description":"sign"',
                "a caption object",
                max_tokens=100,
                progress=progress_messages.append,
            )

        self.assertEqual(parsed["high_level_description"], "sign")
        self.assertEqual(chat.call_count, 1)
        self.assertIn("retrying", progress_messages[0])
        self.assertIn("succeeded", progress_messages[-1])

    def test_reports_original_and_repair_json_failures(self):
        with patch("captioning_kit.llm_captioning.chat_text", return_value='{"still":"broken"'):
            with self.assertRaises(ModelJsonError) as raised:
                parse_json_with_repair(
                    CaptioningSettings(caption_model="repair-model"),
                    "caption",
                    '{"high_level_description":"sign"',
                    "a caption object",
                    max_tokens=100,
                )

        self.assertIn("repair retry failed", str(raised.exception))
        self.assertIn("high_level_description", raised.exception.raw_output)
        self.assertIn("still", raised.exception.repair_output)

    def test_converts_bbox_coordinates(self):
        self.assertEqual(bbox_xyxy_to_yxyx([200, 100, 400, 300]), [100, 200, 300, 400])
        self.assertIsNone(bbox_xyxy_to_yxyx([200, 100, 200, 300]))

    def test_parses_batch_bbox_response(self):
        parsed = parse_batch_bboxes('{"bboxes":{"0":[10,20,30,40],"1":null}}')
        self.assertEqual(parsed["0"], [20, 10, 40, 30])
        self.assertIsNone(parsed["1"])

    def test_parses_batch_bbox_skip_reasons(self):
        parsed, reasons = parse_batch_bboxes_with_reasons(
            '{"bboxes":{"0":[10,20,30,40],"1":null,"2":[10,20,10,40]}}'
        )

        self.assertEqual(parsed["0"], [20, 10, 40, 30])
        self.assertIsNone(parsed["1"])
        self.assertIsNone(parsed["2"])
        self.assertEqual(reasons["1"], "model returned null")
        self.assertEqual(reasons["2"], "model returned invalid bbox")

    def test_bbox_filter_keeps_concrete_element_with_patterned_clothing(self):
        self.assertTrue(
            should_try_bbox(
                {
                    "type": "obj",
                    "desc": (
                        "A young woman sitting on a white toilet, wearing white panties "
                        "with a small heart pattern and a white headband."
                    ),
                }
            )
        )
        self.assertFalse(should_try_bbox({"type": "obj", "desc": "A repeating background pattern."}))

    def test_bbox_target_filter_is_opt_in(self):
        elements = [
            {"type": "obj", "desc": "A repeating background pattern."},
            {"type": "obj", "desc": "A woman wearing a floral pattern dress."},
            {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "A chair."},
        ]

        self.assertEqual(bbox_target_indices(elements, CaptioningSettings()), [0, 1, 2])
        self.assertEqual(
            bbox_target_indices(elements, CaptioningSettings(filter_bbox_targets=True, overwrite_bboxes=False)),
            [1],
        )

    def test_bbox_target_reasons(self):
        elements = [
            {"type": "obj", "desc": "A repeating background pattern."},
            {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "A chair."},
            {"type": "misc", "desc": "A label."},
        ]

        indices, reasons = bbox_target_indices_with_reasons(
            elements,
            CaptioningSettings(filter_bbox_targets=True, overwrite_bboxes=False),
        )

        self.assertEqual(indices, [])
        self.assertEqual(reasons[0], "filtered as vague/ambient")
        self.assertEqual(reasons[1], "existing bbox kept")
        self.assertEqual(reasons[2], "not an obj/text element")

    def test_legacy_caption_model_for_bboxes_flag_is_ignored(self):
        settings = CaptioningSettings(use_caption_model_for_bboxes=True, caption_model="shared-model")
        self.assertFalse(settings.use_caption_model_for_bboxes)
        self.assertNotEqual(runtime_config_for_task(settings, "bbox").api_model, "shared-model")

    def test_default_profile_is_downloadable_local_model(self):
        config = runtime_config_for_task(CaptioningSettings(), "caption")
        self.assertEqual(config.hf_repo, "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF")
        self.assertEqual(config.model_filename, "Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf")

    def test_loads_profiles_from_json_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "id": "local-caption",
                                "label": "Local Caption",
                                "tasks": ["caption"],
                                "kind": "local",
                                "api_model": "local-caption",
                                "mmproj_repo": "other/projector-repo",
                                "local_model_path": "C:/models/model.gguf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profiles = load_model_profiles(path)

        self.assertEqual(profiles["caption"][0].id, "local-caption")
        self.assertEqual(profiles["caption"][0].mmproj_repo, "other/projector-repo")
        self.assertNotEqual(profiles["bbox"][0].id, "local-caption")
        self.assertEqual(profiles["caption"][-2].id, "custom-hf")
        self.assertEqual(profiles["caption"][-1].id, "custom-local")

    def test_loads_partial_prompt_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "prompts"
            folder.mkdir()
            (folder / "bbox_system.txt").write_text("custom bbox system", encoding="utf-8")
            prompts = load_prompts(folder)

        self.assertEqual(prompts["bbox_system"], "custom bbox system")
        self.assertIn("{targets_json}", prompts["bbox_user"])
        self.assertIn("{instructions}", prompts["json_refine_user"])
        self.assertIn("plain_caption_system", prompts)

    def test_writes_default_prompt_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "prompts"
            written = write_default_prompts(folder)

            self.assertTrue((written / "bbox_user.txt").exists())
            self.assertTrue((written / "text_to_json_user.txt").exists())
            self.assertTrue((written / "json_refine_user.txt").exists())

    def test_prompt_placeholder_errors_are_actionable(self):
        with self.assertRaises(AutoCaptionError):
            format_prompt("{missing}", present="x")

    def test_json_directive_is_system_side(self):
        prompts = load_prompts(Path("__missing_prompts_folder__"))
        settings = CaptioningSettings(creative_json=True)
        system = json_system_prompt(prompts, "text_to_json_system", settings)

        self.assertIn("Expansion policy", system)
        self.assertNotIn("{directive}", prompts["text_to_json_user"])
        self.assertNotIn("{directive}", prompts["image_to_json_user"])

    def test_json_refinement_requires_instructions(self):
        with self.assertRaises(AutoCaptionError):
            generate_json_refinement(
                CaptioningSettings(caption_model="vision-model"),
                Path("sample.png"),
                {"high_level_description": "A sign"},
                "",
                "",
            )

    def test_json_refinement_uses_image_context_and_preserves_missing_bboxes(self):
        raw = json.dumps(
            {
                "high_level_description": "A woman seated beside a window.",
                "style_description": {
                    "aesthetics": "natural",
                    "lighting": "window light",
                    "photo": "portrait lens",
                    "medium": "photograph",
                },
                "compositional_deconstruction": {
                    "background": "room",
                    "elements": [{"type": "obj", "desc": "A woman wearing a red jacket, seated in profile."}],
                },
            }
        )
        caption = {
            "high_level_description": "A woman.",
            "style_description": {
                "aesthetics": "natural",
                "lighting": "soft",
                "photo": "",
                "medium": "photograph",
            },
            "compositional_deconstruction": {
                "background": "room",
                "elements": [{"type": "obj", "bbox": [100, 200, 500, 700], "desc": "A woman."}],
            },
        }

        with patch("captioning_kit.llm_captioning.chat_vision", return_value=raw) as chat:
            refined = generate_json_refinement(
                CaptioningSettings(caption_model="vision-model"),
                Path("sample.png"),
                caption,
                "original sidecar caption",
                "Add clothing and pose details to people.",
            )

        self.assertEqual(
            refined["compositional_deconstruction"]["elements"][0]["bbox"],
            [100, 200, 500, 700],
        )
        request = chat.call_args.kwargs["user"]
        self.assertIn("Add clothing and pose details", request)
        self.assertIn("original sidecar caption", request)
        self.assertIn('"high_level_description": "A woman."', request)

    def test_custom_local_profile_uses_selected_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = folder / "model.gguf"
            mmproj = folder / "mmproj.gguf"
            model.write_text("x", encoding="utf-8")
            mmproj.write_text("x", encoding="utf-8")
            settings = CaptioningSettings(
                caption_profile_id="custom-local",
                caption_model="local-caption",
                caption_local_model_path=str(model),
                caption_local_mmproj_path=str(mmproj),
            )
            config = runtime_config_for_task(settings, "caption")
            assets = ensure_model_assets(settings, "caption")

        self.assertEqual(config.kind, "local")
        self.assertEqual(assets.model_path, model)
        self.assertEqual(assets.mmproj_path, mmproj)

    def test_parses_server_host_port(self):
        self.assertEqual(server_host_port("http://127.0.0.1:8000/v1"), ("127.0.0.1", 8000))
        self.assertEqual(server_host_port("https://example.test/v1"), ("example.test", 443))

    def test_parses_server_model_ids(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"data":[{"id":"qwen3vl"},{"id":"local-model"}]}'

        with patch("urllib.request.urlopen", return_value=Response()):
            self.assertEqual(server_model_ids("http://127.0.0.1:8000/v1"), {"qwen3vl", "local-model"})

    def test_builds_llama_server_command(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            mmproj = folder / "mmproj.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")
            mmproj.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                base_url="http://127.0.0.1:8111/v1",
                caption_model="caption-model",
                llama_extra_args="--no-webui",
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, mmproj))

        self.assertIn("llama-server.exe", command)
        self.assertIn("-m", command)
        self.assertIn("--mmproj", command)
        self.assertIn("--port 8111", command)
        self.assertIn("--alias caption-model", command)
        self.assertIn("-b 2048", command)
        self.assertIn("-ub 512", command)
        self.assertIn("-np 1", command)
        # -1 (auto) GPU layers: -ngl is omitted so llama.cpp's fitter decides
        self.assertNotIn("-ngl", command)
        self.assertIn("--reasoning off", command)

    def test_explicit_gpu_layers_passes_ngl(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")
            settings = CaptioningSettings(
                llama_server_path=str(server),
                base_url="http://127.0.0.1:8111/v1",
                caption_model="caption-model",
                llama_gpu_layers=24,
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, None))
        self.assertIn("-ngl 24", command)

    def test_can_limit_llama_reasoning_budget_when_thinking_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                disable_thinking=False,
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, None))

        self.assertNotIn("--reasoning off", command)
        self.assertIn("--reasoning-budget 2048", command)

    def test_can_leave_llama_reasoning_unrestricted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                disable_thinking=False,
                llama_reasoning_budget=-1,
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, None))

        self.assertNotIn("--reasoning off", command)
        self.assertNotIn("--reasoning-budget", command)

    def test_disable_thinking_prefixes_qwen_no_think_directive(self):
        self.assertEqual(request_user_prompt(CaptioningSettings(), "Describe this."), "/no_think\n\nDescribe this.")
        self.assertEqual(request_user_prompt(CaptioningSettings(), "/no_think\nDescribe this."), "/no_think\nDescribe this.")
        self.assertEqual(
            request_user_prompt(CaptioningSettings(disable_thinking=False), "Describe this."),
            "Describe this.",
        )

    def test_strips_thinking_tags_from_model_output(self):
        self.assertEqual(strip_thinking_output("<think>hidden</think>\nVisible caption."), "Visible caption.")

    def test_text_chat_warns_when_thinking_returns_no_visible_output(self):
        client = FakeClient([FakeResponse("", finish_reason="length", model="caption-model")])

        with patch("captioning_kit.llm_captioning._make_openai_client", return_value=client):
            with self.assertRaises(AutoCaptionError) as raised:
                chat_text(
                    CaptioningSettings(disable_thinking=False),
                    model="caption-model",
                    system="system",
                    user="Describe this.",
                    max_tokens=32,
                )

        self.assertIn("Thinking/reasoning is enabled", str(raised.exception))
        self.assertIn("finish_reason=length", str(raised.exception))
        self.assertIn("Thinking token budget", str(raised.exception))
        self.assertEqual(len(client.completions.requests), 1)
        self.assertEqual(client.completions.requests[0]["messages"][1]["content"], "Describe this.")

    def test_vision_chat_uses_image_first_then_text_first_fallback(self):
        client = FakeClient(
            [
                FakeResponse("", finish_reason="length", model="vision-model"),
                FakeResponse("Visible caption.", model="vision-model"),
            ]
        )

        with patch("captioning_kit.llm_captioning._make_openai_client", return_value=client), patch(
            "captioning_kit.llm_captioning.image_to_data_url",
            return_value="data:image/png;base64,abc",
        ):
            result = chat_vision(
                CaptioningSettings(disable_thinking=False),
                model="vision-model",
                image_path=Path("sample.png"),
                system="system",
                user="Describe this image.",
                max_tokens=32,
            )

        self.assertEqual(result, "Visible caption.")
        self.assertEqual(len(client.completions.requests), 2)
        first_user_content = client.completions.requests[0]["messages"][1]["content"]
        retry_user_content = client.completions.requests[1]["messages"][1]["content"]
        self.assertEqual(first_user_content[0]["type"], "image_url")
        self.assertEqual(first_user_content[1]["text"], "Describe this image.")
        self.assertEqual(retry_user_content[0]["text"], "Describe this image.")
        self.assertEqual(retry_user_content[1]["type"], "image_url")

    def test_vision_chat_warns_when_thinking_returns_no_visible_output(self):
        client = FakeClient(
            [
                FakeResponse("", finish_reason="length", model="vision-model"),
                FakeResponse("", finish_reason="length", model="vision-model"),
            ]
        )

        with patch("captioning_kit.llm_captioning._make_openai_client", return_value=client), patch(
            "captioning_kit.llm_captioning.image_to_data_url",
            return_value="data:image/png;base64,abc",
        ):
            with self.assertRaises(AutoCaptionError) as raised:
                chat_vision(
                    CaptioningSettings(disable_thinking=False),
                    model="vision-model",
                    image_path=Path("sample.png"),
                    system="system",
                    user="Describe this image.",
                    max_tokens=32,
                )

        self.assertIn("Thinking/reasoning is enabled", str(raised.exception))
        self.assertIn("finish_reason=length", str(raised.exception))
        self.assertIn("Context size", str(raised.exception))

    def test_safe_repo_dir(self):
        self.assertEqual(safe_repo_dir("org/model name"), "org__model__name")


class EnsureServerRunningTests(unittest.TestCase):
    def _settings(self, **over):
        s = CaptioningSettings()
        base = dict(
            server_start_mode="local",
            auto_start_server=True,
            base_url="http://127.0.0.1:8000/v1",
            api_key="x",
            server_startup_timeout=5.0,
        )
        for k, v in {**base, **over}.items():
            setattr(s, k, v)
        return s

    def test_skips_when_not_local_or_autostart_off(self):
        import captioning_kit.llm_captioning as L
        with patch.object(L, "start_server_process") as start:
            self.assertIsNone(ensure_server_running(self._settings(server_start_mode="existing"), "caption"))
            self.assertIsNone(ensure_server_running(self._settings(auto_start_server=False), "caption"))
            start.assert_not_called()

    def test_skips_launch_when_already_ready(self):
        import captioning_kit.llm_captioning as L
        with patch.object(L, "is_server_ready", return_value=True), \
             patch.object(L, "start_server_process") as start:
            self.assertIsNone(ensure_server_running(self._settings(), "caption"))
            start.assert_not_called()

    def test_launches_when_local_and_not_ready(self):
        import captioning_kit.llm_captioning as L
        with patch.object(L, "is_server_ready", return_value=False), \
             patch.object(L, "ensure_model_assets", return_value=ModelAssets(Path("m.gguf"), Path("mm.gguf"))), \
             patch.object(L, "build_llama_server_command", return_value="llama-server -m m.gguf") as cmd, \
             patch.object(L, "start_server_process", return_value="PROC") as start:
            proc = ensure_server_running(self._settings(), "caption")
            self.assertEqual(proc, "PROC")
            cmd.assert_called_once()
            start.assert_called_once()


class FindLlamaServerTests(unittest.TestCase):
    def test_default_llama_dir_is_independent_of_models_dir(self):
        import captioning_kit.llm_captioning as L
        self.assertEqual(L.default_llama_dir(), L.app_base_dir() / "llama")
        self.assertNotEqual(L.default_llama_dir(), L.default_models_dir())

    def test_finds_binary_in_managed_llama_dir(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            llama = Path(tmp) / "llama"
            (llama / "bin").mkdir(parents=True)
            exe = "llama-server.exe" if os.name == "nt" else "llama-server"
            target = llama / "bin" / exe
            target.write_text("#!/bin/sh\n")
            with patch.object(L, "default_llama_dir", return_value=llama), \
                 patch.object(L, "app_base_dir", return_value=Path(tmp) / "app"), \
                 patch.object(L.shutil, "which", return_value=None):
                self.assertEqual(L.find_llama_server(), target)

    def test_falls_back_to_path(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(L, "default_llama_dir", return_value=Path(tmp) / "llama"), \
                 patch.object(L, "app_base_dir", return_value=Path(tmp) / "app"), \
                 patch.object(L.shutil, "which", return_value="/usr/bin/llama-server"):
                self.assertEqual(L.find_llama_server(), Path("/usr/bin/llama-server"))

    def test_returns_none_when_nothing_found(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(L, "default_llama_dir", return_value=Path(tmp) / "llama"), \
                 patch.object(L, "app_base_dir", return_value=Path(tmp) / "app"), \
                 patch.object(L.shutil, "which", return_value=None):
                self.assertIsNone(L.find_llama_server())


class DetectGpuTests(unittest.TestCase):
    def test_sm_from_compute_cap(self):
        from captioning_kit.llm_captioning import _sm_from_compute_cap
        self.assertEqual(_sm_from_compute_cap("12.0"), "120")
        self.assertEqual(_sm_from_compute_cap("8.6"), "86")
        self.assertEqual(_sm_from_compute_cap("8.9"), "89")
        self.assertEqual(_sm_from_compute_cap("garbage"), "")

    def test_detects_nvidia_from_smi(self):
        import captioning_kit.llm_captioning as L
        import subprocess as sp
        cp = sp.CompletedProcess(args=[], returncode=0,
                                 stdout="0, NVIDIA GeForce RTX 5090, 12.0, 32607\n")
        with patch.object(L.subprocess, "run", return_value=cp):
            info = L.detect_gpu()
        self.assertEqual(info.vendor, "nvidia")
        self.assertEqual(info.name, "NVIDIA GeForce RTX 5090")
        self.assertEqual(info.compute_cap, "12.0")
        self.assertEqual(info.sm, "120")
        self.assertEqual(info.backend, "cuda")
        self.assertEqual(info.index, 0)
        self.assertAlmostEqual(info.vram_total_gb, 31.8, places=1)
        self.assertIn("sm120", info.summary)

    def test_detect_gpus_enumerates_all(self):
        import captioning_kit.llm_captioning as L
        import subprocess as sp
        cp = sp.CompletedProcess(args=[], returncode=0, stdout=(
            "0, NVIDIA GeForce RTX 5090, 12.0, 32607\n"
            "1, NVIDIA GeForce RTX 3090, 8.6, 24576\n"
        ))
        with patch.object(L.subprocess, "run", return_value=cp):
            gpus = L.detect_gpus()
        self.assertEqual([g.index for g in gpus], [0, 1])
        self.assertEqual([g.name for g in gpus],
                         ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3090"])
        self.assertEqual(gpus[1].sm, "86")

    def test_falls_back_to_vulkan_without_nvidia(self):
        import captioning_kit.llm_captioning as L
        with patch.object(L.subprocess, "run", side_effect=FileNotFoundError()):
            info = L.detect_gpu()
        self.assertEqual(info.vendor, "none")
        self.assertEqual(info.backend, "vulkan")
        self.assertIn("VULKAN", info.summary)


class LlamaAssetResolverTests(unittest.TestCase):
    OFFICIAL = [
        {"name": "llama-b9828-bin-win-cuda-13.3-x64.zip"},
        {"name": "llama-b9828-bin-win-cuda-12.4-x64.zip"},
        {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip"},
        {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip"},
        {"name": "llama-b9828-bin-win-vulkan-x64.zip"},
        {"name": "llama-b9828-bin-win-cpu-x64.zip"},
        {"name": "llama-b9828-bin-macos-arm64.tar.gz"},
        {"name": "llama-b9828-bin-macos-x64.tar.gz"},
        {"name": "llama-b9828-bin-ubuntu-x64.tar.gz"},
        {"name": "llama-b9828-bin-ubuntu-arm64.tar.gz"},
    ]
    LLAMAUP = [
        {"name": "llama-b9800-linux-cuda12.8-sm120-x64.tar.gz"},
        {"name": "llama-b9800-linux-cuda12.6-sm89-x64.tar.gz"},
    ]

    def names(self, assets):
        from captioning_kit.llm_captioning import _asset_name
        return [_asset_name(a) for a in assets]

    def test_windows_cuda_picks_newest_plus_cudart(self):
        from captioning_kit.llm_captioning import select_llama_assets
        got = select_llama_assets(self.OFFICIAL, system="Windows", arch="x64", backend="cuda")
        names = self.names(got)
        self.assertEqual(len(got), 2)
        self.assertTrue(any("win-cuda-13.3" in n for n in names))   # newest cuda
        self.assertTrue(any("cudart" in n and "13.3" in n for n in names))  # runtime companion
        self.assertFalse(any("12.4" in n for n in names))

    def test_windows_vulkan_and_cpu(self):
        from captioning_kit.llm_captioning import select_llama_assets
        v = select_llama_assets(self.OFFICIAL, system="Windows", arch="x64", backend="vulkan")
        c = select_llama_assets(self.OFFICIAL, system="Windows", arch="x64", backend="cpu")
        self.assertEqual(self.names(v), ["llama-b9828-bin-win-vulkan-x64.zip"])
        self.assertEqual(self.names(c), ["llama-b9828-bin-win-cpu-x64.zip"])

    def test_macos_arch_specific(self):
        from captioning_kit.llm_captioning import select_llama_assets
        got = select_llama_assets(self.OFFICIAL, system="Darwin", arch="arm64", backend="metal")
        self.assertEqual(self.names(got), ["llama-b9828-bin-macos-arm64.tar.gz"])

    def test_linux_cuda_matches_sm(self):
        from captioning_kit.llm_captioning import select_llama_assets
        got = select_llama_assets(self.LLAMAUP, system="Linux", arch="x64", backend="cuda", sm="120")
        self.assertEqual(self.names(got), ["llama-b9800-linux-cuda12.8-sm120-x64.tar.gz"])
        # unknown sm -> no match (caller falls back to build/vulkan)
        self.assertEqual(select_llama_assets(self.LLAMAUP, system="Linux", arch="x64", backend="cuda", sm="61"), ())

    def test_linux_vulkan_uses_official_ubuntu(self):
        from captioning_kit.llm_captioning import select_llama_assets
        got = select_llama_assets(self.OFFICIAL, system="Linux", arch="x64", backend="vulkan")
        self.assertEqual(self.names(got), ["llama-b9828-bin-ubuntu-x64.tar.gz"])

    def test_no_match_returns_empty(self):
        from captioning_kit.llm_captioning import select_llama_assets
        self.assertEqual(select_llama_assets([{"name": "readme.txt"}], system="Windows", arch="x64", backend="cuda"), ())

    def test_build_number_parsing_and_staleness(self):
        from captioning_kit.llm_captioning import parse_build_number, is_update_available
        self.assertEqual(parse_build_number("llama-b9828-bin-win-cpu-x64.zip"), 9828)
        self.assertIsNone(parse_build_number("v3.18.0"))
        self.assertTrue(is_update_available(9828, 9900))
        self.assertFalse(is_update_available(9900, 9828))
        self.assertFalse(is_update_available(None, 9900))

    def test_installed_record_roundtrip(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(L, "default_llama_dir", return_value=Path(tmp) / "llama"):
                self.assertIsNone(L.read_installed_llama())  # nothing yet
                rec = L.InstalledLlama(source="ggml-org/llama.cpp", build=9828, backend="cuda",
                                       sm="120", asset="llama-b9828-bin-win-cuda-13.3-x64.zip",
                                       sha256="abc", binary="/x/llama-server.exe", installed_at="2026-06-28")
                L.write_installed_llama(rec)
                back = L.read_installed_llama()
                self.assertEqual(back.build, 9828)
                self.assertEqual(back.backend, "cuda")
                self.assertEqual(back.asset, "llama-b9828-bin-win-cuda-13.3-x64.zip")


class LlamaUpdateStateTests(unittest.TestCase):
    def _rec(self, published, build=9000):
        from captioning_kit.llm_captioning import InstalledLlama
        return InstalledLlama(build=build, published_at=published)

    def test_arch_error_detection(self):
        from captioning_kit.llm_captioning import is_model_arch_error
        self.assertTrue(is_model_arch_error("error: unknown model architecture: 'glm5'"))
        self.assertTrue(is_model_arch_error("unsupported model architecture"))
        self.assertFalse(is_model_arch_error("CUDA out of memory"))
        self.assertFalse(is_model_arch_error(""))

    def test_build_age_days(self):
        import captioning_kit.llm_captioning as L
        from datetime import datetime, timezone
        now = datetime(2026, 6, 28, tzinfo=timezone.utc)
        self.assertEqual(L.build_age_days(self._rec("2026-06-28T00:00:00Z"), now=now), 0)
        self.assertEqual(L.build_age_days(self._rec("2026-05-01T00:00:00Z"), now=now), 58)
        self.assertIsNone(L.build_age_days(self._rec(""), now=now))

    def test_update_state_classification(self):
        import captioning_kit.llm_captioning as L
        from datetime import datetime, timezone
        now = datetime(2026, 6, 28, tzinfo=timezone.utc)
        self.assertEqual(L.update_state(None, 9100)["state"], "none")
        # newer exists but binary is recent -> informational only
        self.assertEqual(L.update_state(self._rec("2026-06-20T00:00:00Z", 9000), 9100, now=now)["state"], "available")
        # no newer / not older -> up to date
        self.assertEqual(L.update_state(self._rec("2026-06-25T00:00:00Z", 9100), 9100, now=now)["state"], "up_to_date")
        # old enough -> recommended (even though only ~100 builds behind)
        old = L.update_state(self._rec("2026-05-01T00:00:00Z", 9000), 9100, now=now)
        self.assertEqual(old["state"], "recommended")
        self.assertGreaterEqual(old["age_days"], 30)
        # recommended on age even if 'latest' couldn't be fetched (network down)
        self.assertEqual(L.update_state(self._rec("2026-05-01T00:00:00Z", 9000), None, now=now)["state"], "recommended")


class LlamaFetchAndInstallTests(unittest.TestCase):
    def test_fetch_release_parses_assets(self):
        import captioning_kit.llm_captioning as L
        payload = {
            "tag_name": "b9828",
            "published_at": "2026-06-27T23:18:43Z",
            "assets": [
                {"name": "llama-b9828-bin-ubuntu-x64.tar.gz", "browser_download_url": "http://x/a",
                 "size": 123, "digest": "sha256:deadbeef"},
                {"name": "other.txt", "browser_download_url": "http://x/b", "size": 1},
            ],
        }
        with patch.object(L, "_github_get", return_value=payload):
            info = L.fetch_release("ggml-org/llama.cpp")
        self.assertEqual(info.build, 9828)
        self.assertEqual(info.published_at[:4], "2026")
        self.assertEqual(len(info.assets), 2)
        self.assertEqual(info.assets[0].sha256, "deadbeef")
        with patch.object(L, "_github_get", side_effect=OSError("boom")):
            self.assertIsNone(L.fetch_release("x/y"))

    def test_verify_sha256(self):
        import captioning_kit.llm_captioning as L
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "blob"
            f.write_bytes(b"hello world")
            good = hashlib.sha256(b"hello world").hexdigest()
            self.assertTrue(L.verify_sha256(f, good))
            self.assertFalse(L.verify_sha256(f, "00" * 32))
            self.assertFalse(L.verify_sha256(f, ""))

    def _make_archive(self, path: Path):
        # a tar.gz with a nested llama-server + a sibling lib, like a real release
        import tarfile, io
        with tarfile.open(path, "w:gz") as tf:
            for name, body in (("dist/llama-server", b"#!/bin/sh\necho hi\n"),
                               ("dist/libggml.so", b"\x00lib")):
                data = body
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def test_install_unpack_swap_and_rollback(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama"
            fixture = Path(tmp) / "fixture.tar.gz"
            self._make_archive(fixture)

            def fake_downloader(url, dest, progress=None):
                shutil.copy(fixture, dest)
                return Path(dest)

            assets = (L.ReleaseAsset(name="llama-b9828-bin-ubuntu-x64.tar.gz", url="http://x", sha256=""),)
            release = L.ReleaseInfo(repo="ggml-org/llama.cpp", tag="b9828", build=9828,
                                    published_at="2026-06-27T23:18:43Z", assets=assets)
            with patch.object(L, "default_llama_dir", return_value=llama_dir):
                rec = L.install_llama_release(release, assets, backend="vulkan", downloader=fake_downloader)
                self.assertEqual(rec.build, 9828)
                self.assertEqual(rec.backend, "vulkan")
                self.assertEqual(rec.published_at, "2026-06-27T23:18:43Z")
                self.assertTrue(Path(rec.binary).exists())
                # installed.json round-trips and find_llama_server prefers it
                self.assertEqual(L.read_installed_llama().build, 9828)
                self.assertEqual(L.find_llama_server(), Path(rec.binary))
                self.assertFalse(L.has_llama_backup())
                # second install creates a backup we can roll back to
                rec2 = L.install_llama_release(release, assets, backend="vulkan", downloader=fake_downloader)
                self.assertTrue(L.has_llama_backup())
                self.assertTrue(L.rollback_llama())
                self.assertTrue(Path(L.read_installed_llama().binary).exists())


class LlamaPlanTests(unittest.TestCase):
    def test_resolve_backend_hint_overrides_detection(self):
        import captioning_kit.llm_captioning as L
        gpu = L.GpuInfo(vendor="nvidia", backend="cuda", sm="120", name="X")
        s = L.CaptioningSettings()
        self.assertEqual(L.resolve_backend(s, gpu), "cuda")          # auto -> detected
        s.llama_backend_hint = "vulkan"
        self.assertEqual(L.resolve_backend(s, gpu), "vulkan")        # hint wins

    def test_plan_resolves_and_handles_offline(self):
        import captioning_kit.llm_captioning as L
        rel = L.ReleaseInfo(repo="ggml-org/llama.cpp", tag="b9828", build=9828,
                            published_at="2026-06-27T00:00:00Z",
                            assets=(L.ReleaseAsset(name="llama-b9828-bin-ubuntu-x64.tar.gz",
                                                   url="http://x", sha256="", size=12_000_000),))
        s = L.CaptioningSettings()
        s.llama_backend_hint = "vulkan"   # deterministic regardless of test host GPU
        plan = L.plan_llama_acquisition(s, fetch=lambda repo, tag=None: rel)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.backend, "vulkan")
        self.assertEqual(len(plan.assets), 1)
        self.assertIn("VULKAN", plan.description)
        # offline / no release -> None
        self.assertIsNone(L.plan_llama_acquisition(s, fetch=lambda *a, **k: None))


class ResourceMonitorTests(unittest.TestCase):
    def test_parse_gpu_usage(self):
        import captioning_kit.llm_captioning as L
        vu, vt, gp = L._parse_gpu_usage("18342, 32768, 73")
        self.assertAlmostEqual(vu, 17.91, places=1)
        self.assertAlmostEqual(vt, 32.0, places=1)
        self.assertEqual(gp, 73.0)
        self.assertEqual(L._parse_gpu_usage("garbage"), (None, None, None))
        self.assertEqual(L._parse_gpu_usage(""), (None, None, None))

    def test_format_resources_omits_missing(self):
        import captioning_kit.llm_captioning as L
        full = L.ResourceSample(ram_percent=41.0, vram_used_gb=18.3, vram_total_gb=32.0, gpu_percent=73.0)
        self.assertEqual(L.format_resources(full), "RAM 41%  \u00b7  VRAM 18.3/32 GB  \u00b7  GPU 73%")
        self.assertEqual(L.format_resources(L.ResourceSample(ram_percent=55.0)), "RAM 55%")
        self.assertEqual(L.format_resources(L.ResourceSample()), "")

    def test_read_ram_live(self):
        import captioning_kit.llm_captioning as L
        used, total = L._read_ram()
        # this host is Linux, so we expect a real reading
        self.assertIsNotNone(total)
        self.assertTrue(0 < used <= total)


class HasModelConfigTests(unittest.TestCase):
    def test_has_model_config(self):
        import captioning_kit.llm_captioning as L
        from types import SimpleNamespace
        cfg = lambda local="", hf="": SimpleNamespace(local_model_path=local, hf_repo=hf)
        s = L.CaptioningSettings()
        with patch.object(L, "runtime_config_for_task", return_value=cfg(hf="org/repo")):
            self.assertTrue(L.has_model_config(s))
        with patch.object(L, "runtime_config_for_task", return_value=cfg(local="/x/m.gguf")):
            self.assertTrue(L.has_model_config(s))
        with patch.object(L, "runtime_config_for_task", return_value=cfg()):
            self.assertFalse(L.has_model_config(s))
        with patch.object(L, "runtime_config_for_task", side_effect=RuntimeError):
            self.assertFalse(L.has_model_config(s))


class ServerLaunchEnvTests(unittest.TestCase):
    def _var(self):
        import os, sys
        if os.name == "nt":
            return "PATH"
        return "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"

    def test_prepends_binary_and_lib_dirs(self):
        import os
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp) / "bin"
            (bindir / "lib").mkdir(parents=True)
            binp = bindir / ("llama-server.exe" if os.name == "nt" else "llama-server")
            binp.write_text("x")
            (bindir / "lib" / "libggml.so").write_text("x")
            env = L.server_launch_env(binp)
            value = env[self._var()]
            self.assertIn(str(bindir), value)
            self.assertIn(str(bindir / "lib"), value)

    def test_none_returns_untouched_copy(self):
        import captioning_kit.llm_captioning as L
        env = L.server_launch_env(None)
        self.assertIsInstance(env, dict)

    def test_resolver_prefers_explicit_path(self):
        import captioning_kit.llm_captioning as L
        s = L.CaptioningSettings()
        s.llama_server_path = "/custom/llama-server"
        self.assertEqual(str(L.resolve_llama_server_path(s)), "/custom/llama-server")
        s.llama_server_path = ""
        with patch.object(L, "find_llama_server", return_value=Path("/auto/llama-server")):
            self.assertEqual(str(L.resolve_llama_server_path(s)), "/auto/llama-server")


class MissingModelFilesTests(unittest.TestCase):
    def test_missing_then_cached(self):
        import captioning_kit.llm_captioning as L
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            s = L.CaptioningSettings()
            s.models_dir = tmp
            cfg = SimpleNamespace(local_model_path="", hf_repo="org/repo", mmproj_repo="",
                                  model_filename="m.gguf", mmproj_filename="mm.gguf")
            with patch.object(L, "runtime_config_for_task", return_value=cfg):
                missing = L.missing_model_files(s)
                self.assertIn("m.gguf", missing)
                self.assertIn("mm.gguf", missing)
                d = Path(tmp) / L.safe_repo_dir("org/repo")
                d.mkdir(parents=True, exist_ok=True)
                (d / "m.gguf").write_text("x")
                missing2 = L.missing_model_files(s)
                self.assertNotIn("m.gguf", missing2)
                self.assertIn("mm.gguf", missing2)

    def test_local_path_needs_no_download(self):
        import captioning_kit.llm_captioning as L
        from types import SimpleNamespace
        cfg = SimpleNamespace(local_model_path="/x/m.gguf", hf_repo="", mmproj_repo="",
                              model_filename="", mmproj_filename="")
        with patch.object(L, "runtime_config_for_task", return_value=cfg):
            self.assertEqual(L.missing_model_files(L.CaptioningSettings()), [])


class ModelLessLaunchTests(unittest.TestCase):
    def test_router_detection_cached(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            binp = Path(tmp) / "llama-server"
            binp.write_text("x")
            L._ROUTER_SUPPORT.clear()
            R = lambda out: type("R", (), {"stdout": out, "stderr": "", "returncode": 0})()
            with patch.object(L.subprocess, "run", return_value=R("  --models-dir DIR\n")):
                self.assertTrue(L.llama_server_supports_router(binp))
            L._ROUTER_SUPPORT.clear()
            with patch.object(L.subprocess, "run", return_value=R("  -m FILE  model path\n")):
                self.assertFalse(L.llama_server_supports_router(binp))

    def test_model_less_command_and_error(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            binp = Path(tmp) / "llama-server"
            binp.write_text("x")
            s = L.CaptioningSettings()
            s.llama_server_path = str(binp)
            s.models_dir = str(Path(tmp) / "models")
            L._ROUTER_SUPPORT[str(binp)] = True
            cmd = L.build_llama_server_command(s, "caption", L.ModelAssets(), model_less=True)
            self.assertIn("--models-dir", cmd)
            self.assertNotIn(" -m ", cmd)
            self.assertNotIn("--mmproj", cmd)
            L._ROUTER_SUPPORT[str(binp)] = False
            with self.assertRaises(L.AutoCaptionError):
                L.build_llama_server_command(s, "caption", L.ModelAssets(), model_less=True)


class ModelDiscoveryTests(unittest.TestCase):
    def test_locate_app_and_extra_dirs(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            app = tmp / "app"
            (app / L.safe_repo_dir("org/repo")).mkdir(parents=True)
            (app / L.safe_repo_dir("org/repo") / "m.gguf").write_text("x")
            lm = tmp / "lmstudio" / "models" / "author" / "model"
            lm.mkdir(parents=True)
            (lm / "vision.gguf").write_text("y")
            s = L.CaptioningSettings()
            s.models_dir = str(app)
            s.extra_model_dirs = f"{tmp/'lmstudio'/'models'}\n{tmp/'missing'}"
            self.assertEqual(L.locate_existing_model_file(s, "org/repo", "m.gguf").name, "m.gguf")
            self.assertEqual(L.locate_existing_model_file(s, "v/repo", "vision.gguf").name, "vision.gguf")
            self.assertIsNone(L.locate_existing_model_file(s, "x/y", "absent.gguf"))

    def test_missing_model_files_uses_discovery(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lm = tmp / "lm"
            lm.mkdir()
            (lm / "present.gguf").write_text("z")
            s = L.CaptioningSettings()
            s.models_dir = str(tmp / "app")
            s.extra_model_dirs = str(lm)
            s.caption_profile_id = "custom"
            s.caption_hf_repo = "org/repo"
            s.caption_model_filename = "present.gguf;absent.gguf"
            s.caption_mmproj_filename = ""
            missing = L.missing_model_files(s, "caption")
            self.assertNotIn("present.gguf", missing)
            self.assertIn("absent.gguf", missing)


    def test_discover_and_pair_mmproj(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "models" / "org__repo"
            d.mkdir(parents=True)
            (d / "model-Q4.gguf").write_text("x")
            (d / "model-mmproj.gguf").write_text("m")
            other = tmp / "lm" / "a"
            other.mkdir(parents=True)
            (other / "plain-Q5.gguf").write_text("y")
            s = L.CaptioningSettings()
            s.models_dir = str(tmp / "models")
            s.extra_model_dirs = str(tmp / "lm")
            models, mmprojs = L.discover_local_gguf_models(s)
            names = {p.name for p in models}
            self.assertIn("model-Q4.gguf", names)
            self.assertIn("plain-Q5.gguf", names)
            self.assertNotIn("model-mmproj.gguf", names)
            paired = L.guess_mmproj_for(d / "model-Q4.gguf", mmprojs)
            self.assertIsNotNone(paired)
            self.assertIn("mmproj", paired.name)
            # a model with no sibling projector pairs to nothing
            self.assertIsNone(L.guess_mmproj_for(other / "plain-Q5.gguf", mmprojs))


    def test_vram_fit_and_recommendation(self):
        import captioning_kit.llm_captioning as L
        self.assertEqual(L.vram_fit(20, 32), "fits")
        self.assertEqual(L.vram_fit(26, 32), "tight")
        self.assertEqual(L.vram_fit(30, 32), "too_big")
        self.assertEqual(L.vram_fit(20, None), "unknown")
        self.assertEqual(L.vram_fit(0, 32), "unknown")
        # tiers
        self.assertEqual(L.model_size_tier(7), "Small")
        self.assertEqual(L.model_size_tier(11), "Medium")
        self.assertEqual(L.model_size_tier(20), "Large")
        self.assertEqual(L.model_size_tier(30), "XL")
        # big card -> curated flagship; tiny/unknown -> smallest
        self.assertEqual(L.recommend_profile_for_vram("caption", 32).id, "unsloth-qwen3vl-30b-q4")
        smallest = min(p.vram_gb for p in L.profiles_for_task("caption") if p.kind == "hf" and p.vram_gb > 0)
        self.assertEqual(L.recommend_profile_for_vram("caption", 6).vram_gb, smallest)
        self.assertEqual(L.recommend_profile_for_vram("caption", None).vram_gb, smallest)

    def test_qwen25_dropped_and_migration(self):
        import captioning_kit.llm_captioning as L
        ids = [p.id for p in L.profiles_for_task("caption")]
        self.assertFalse(any("qwen25" in i for i in ids))
        self.assertEqual(ids[0], "unsloth-qwen3vl-30b-q4")
        # a saved selection pointing at the dropped profile falls back to the default
        self.assertEqual(L._profile_by_id("caption", "unsloth-qwen25vl-7b-q4").id, "unsloth-qwen3vl-30b-q4")


class ConvertModePromptTests(unittest.TestCase):
    """Image->JSON assembly switches to the convert framing + source-caption block
    only when a source caption is supplied; otherwise it is the image-only path."""

    CONVERT_PHRASE = "synthesize it into the schema fields"

    def _run(self, **kwargs):
        captured = {}

        def fake_chat_vision(settings, model, image_path, system, user, max_tokens, temperature=0.0):
            captured["system"], captured["user"] = system, user
            return ('{"high_level_description":"x","style_description":{"aesthetics":"a",'
                    '"lighting":"l","photo":"p","medium":"photograph"},'
                    '"compositional_deconstruction":{"background":"b","elements":[]}}')

        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as temp:
            img = Path(temp) / "foo.jpg"
            img.write_bytes(b"x")
            with patch.object(L, "chat_vision", fake_chat_vision):
                L.generate_json_from_image(CaptioningSettings(caption_model="m"), img, **kwargs)
        return captured

    def test_convert_mode_uses_framing_and_source_caption(self):
        cap = self._run(guidance="folder words", source_caption="four puppies, indoor, warm")
        self.assertIn(self.CONVERT_PHRASE, cap["system"])           # framing carries it
        self.assertIn("Source caption:\nfour puppies", cap["user"])  # source in user msg
        self.assertIn("folder words", cap["system"])                # guidance still present
        self.assertEqual(cap["user"].count(self.CONVERT_PHRASE), 0)  # not duplicated into user

    def test_image_only_mode_unchanged_without_source(self):
        cap = self._run(guidance="g")
        self.assertNotIn(self.CONVERT_PHRASE, cap["system"])
        self.assertIn("Do not reference any existing sidecar caption", cap["user"])


class ServerLogDiagnosisTests(unittest.TestCase):
    def _diag(self, text):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "llama-server.log"
            log.write_text(text, encoding="utf-8")
            return L.diagnose_server_log(log)

    def test_classifies_oom(self):
        cat, hint = self._diag("cudaMalloc failed: out of memory\n")
        self.assertEqual(cat, "oom")
        self.assertIn("VRAM", hint)

    def test_classifies_missing_lib_nccl(self):
        cat, hint = self._diag("error while loading shared libraries: libnccl.so.2\n")
        self.assertEqual(cat, "missing_lib")
        self.assertIn("nvidia-nccl-cu12", hint)

    def test_classifies_crash(self):
        cat, _ = self._diag("/x/ggml.c:1: GGML_ASSERT(a==b) failed\nAborted (core dumped)\n")
        self.assertEqual(cat, "crash")

    def test_clean_log_is_unclassified(self):
        self.assertEqual(self._diag("srv log_info: ready\n"), ("", ""))

    def test_oom_hint_is_server_agnostic(self):
        # The classifier must not bake in built-in-only remediation (context/GPU
        # layers aren't configurable for an external server).
        _cat, hint = self._diag("cudaMalloc failed: out of memory\n")
        self.assertNotIn("Preferences", hint)
        self.assertNotIn("GPU layers", hint)

    def test_startup_hint_adds_builtin_oom_remediation(self):
        import captioning_kit.llm_captioning as L
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "llama-server.log"
            log.write_text("cudaMalloc failed: out of memory\n", encoding="utf-8")
            hint = L._server_startup_hint(log)
        self.assertIn("GPU layers in Preferences", hint)


if __name__ == "__main__":
    unittest.main()


class LlamaAcquisitionPrefersLatestTests(unittest.TestCase):
    """A fresh install must fetch the same build the Update button would.

    Pinning the first install meant a new user installed a weeks-old binary and was
    immediately told to update — and since Update fetches latest anyway, the pin
    protected nobody. It survives only as a rescue when the newest release ships no
    asset for the user's platform/backend.
    """

    def setUp(self):
        import captioning_kit.llm_captioning as L
        self.L = L

    def _release(self, tag, build, assets=("asset.zip",)):
        L = self.L
        return L.ReleaseInfo(
            repo="r", tag=tag, build=build, published_at="",
            assets=tuple(L.ReleaseAsset(name=n, url="u") for n in assets))

    def test_latest_is_requested_first_and_the_pin_is_untouched(self):
        L = self.L
        seen = []

        def fetch(repo, tag=None, timeout=10.0):
            seen.append(tag)
            return self._release(tag or "b9999", 9999 if tag is None else 9828)

        with patch.object(L, "select_llama_assets", lambda a, **k: list(a)):
            plan = L.plan_llama_acquisition(L.CaptioningSettings(), fetch=fetch)
        self.assertEqual(seen, [None], "latest must be tried first")
        self.assertEqual(plan.release.build, 9999)

    def test_falls_back_to_pin_when_latest_has_no_matching_asset(self):
        L = self.L
        seen = []

        def fetch(repo, tag=None, timeout=10.0):
            seen.append(tag)
            return (self._release("b9999", 9999, assets=())
                    if tag is None else self._release(tag, 9828))

        with patch.object(L, "select_llama_assets",
                               lambda a, **k: [] if not a else list(a)):
            plan = L.plan_llama_acquisition(L.CaptioningSettings(), fetch=fetch)
        self.assertEqual(seen, [None, f"b{L.FALLBACK_LLAMA_BUILD}"])
        self.assertEqual(plan.release.build, 9828)

    def test_returns_none_when_nothing_is_reachable(self):
        L = self.L
        self.assertIsNone(
            L.plan_llama_acquisition(L.CaptioningSettings(), fetch=lambda *a, **k: None))


class CudaRuntimeDiscoveryTests(unittest.TestCase):
    """The CUDA prebuilts link the whole CUDA runtime without bundling it. NCCL was
    just the first soname to bite — cuBLAS fails identically — so discovery has to
    cover the family rather than one library."""

    def setUp(self):
        import captioning_kit.llm_captioning as L
        self.L = L
        self.root = Path(tempfile.mkdtemp())
        for component, soname in (("cublas", "libcublas.so.12"),
                                  ("nccl", "libnccl.so.2"),
                                  ("cuda_runtime", "libcudart.so.12")):
            lib = self.root / "nvidia" / component / "lib"
            lib.mkdir(parents=True)
            (lib / soname).write_bytes(b"")
        (self.root / "nvidia" / "empty" / "lib").mkdir(parents=True)
        torch_lib = self.root / "torch" / "lib"
        torch_lib.mkdir(parents=True)
        (torch_lib / "libtorch_cuda.so").write_bytes(b"")

    def _dirs(self):
        with patch.object(self.L, "_site_package_roots", lambda: {self.root}):
            return sorted(str(d.relative_to(self.root))
                          for d in self.L.find_cuda_lib_dirs())

    def test_finds_every_component_wheel_not_just_the_first(self):
        found = self._dirs()
        for expected in ("nvidia/cublas/lib", "nvidia/cuda_runtime/lib",
                         "nvidia/nccl/lib", "torch/lib"):
            self.assertIn(expected, found)

    def test_skips_directories_with_no_libraries(self):
        self.assertNotIn("nvidia/empty/lib", self._dirs())

    def test_all_discovered_dirs_reach_the_launched_server(self):
        with patch.object(self.L, "_site_package_roots", lambda: {self.root}):
            env = self.L.server_launch_env(Path("/opt/llama/bin/llama-server"))
        path = env["LD_LIBRARY_PATH"]
        for rel in self._dirs():
            self.assertIn(str(self.root / rel), path)


class CudaMissingLibraryHintTests(unittest.TestCase):
    def setUp(self):
        import captioning_kit.llm_captioning as L
        self.L = L
        self.tmp = Path(tempfile.mkdtemp())

    def _hint(self, soname):
        log = self.tmp / "server.log"
        log.write_text("llama-server: error while loading shared libraries: "
                       f"{soname}: cannot open shared object file")
        return self.L.diagnose_server_log(log)

    def test_names_the_exact_pip_package(self):
        category, hint = self._hint("libcublas.so.12")
        self.assertEqual(category, "missing_lib")
        self.assertIn("nvidia-cublas-cu12", hint)
        self.assertIn("cuda_fix", hint)

    def test_nccl_still_resolves_to_its_own_package(self):
        self.assertIn("nvidia-nccl-cu12", self._hint("libnccl.so.2")[1])

    def test_driver_library_does_not_suggest_pip(self):
        """libcuda.so.1 ships with the NVIDIA driver; suggesting pip would send the
        user down a dead end."""
        hint = self._hint("libcuda.so.1")[1]
        self.assertIn("driver", hint.lower())
        self.assertNotIn("pip install", hint)

    def test_unknown_library_still_gives_a_usable_message(self):
        category, hint = self._hint("libmystery.so.9")
        self.assertEqual(category, "missing_lib")
        self.assertIn("libmystery.so.9", hint)

    def test_package_map_covers_the_common_runtime(self):
        for soname, package in (("libcublasLt.so.12", "nvidia-cublas-cu12"),
                                ("libcusparse.so.12", "nvidia-cusparse-cu12"),
                                ("libnvrtc.so.12", "nvidia-cuda-nvrtc-cu12")):
            self.assertEqual(self.L.cuda_package_for_lib(soname), package)


def _write_gguf(path, pairs, with_array=False):
    """Minimal but real GGUF header, so the parser is tested against the actual
    byte layout rather than a stand-in."""
    import struct

    def w_str(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, value in pairs:
        if isinstance(value, str):
            body += w_str(key) + struct.pack("<I", 8) + w_str(value)
        else:
            body += w_str(key) + struct.pack("<I", 4) + struct.pack("<I", value)
    count = len(pairs)
    if with_array:
        body += (w_str("tokenizer.ggml.tokens") + struct.pack("<I", 9)
                 + struct.pack("<I", 8) + struct.pack("<Q", 3)
                 + w_str("a") + w_str("b") + w_str("c"))
        count += 1
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", count)
    Path(path).write_bytes(head + body + b"\x00" * 64)


class MmprojPairingTests(unittest.TestCase):
    """A projector from the wrong model doesn't fail cleanly — llama-server hangs
    on startup — so mismatches have to be caught before launch."""

    def setUp(self):
        import captioning_kit.llm_captioning as L
        self.L = L
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "qwen.gguf"
        _write_gguf(self.model, [("general.architecture", "qwen3vl"),
                                 ("qwen3vl.embedding_length", 2048)], with_array=True)
        self.good = self.tmp / "good-mmproj.gguf"
        _write_gguf(self.good, [("general.architecture", "clip"),
                                ("clip.vision.projection_dim", 2048)])
        self.wrong = self.tmp / "wrong-mmproj.gguf"
        _write_gguf(self.wrong, [("general.architecture", "clip"),
                                 ("clip.vision.projection_dim", 3584)])

    def test_reads_metadata_past_arrays(self):
        meta = self.L.read_gguf_metadata(self.model)
        self.assertEqual(meta["general.architecture"], "qwen3vl")
        self.assertEqual(meta["qwen3vl.embedding_length"], 2048)

    def test_matching_pair_is_accepted(self):
        self.assertTrue(self.L.check_mmproj_pairing(self.model, self.good)[0])

    def test_dimension_mismatch_is_rejected_with_the_numbers(self):
        ok, reason = self.L.check_mmproj_pairing(self.model, self.wrong)
        self.assertFalse(ok)
        self.assertIn("3584", reason)
        self.assertIn("2048", reason)

    def test_full_model_used_as_projector_is_rejected(self):
        ok, reason = self.L.check_mmproj_pairing(self.model, self.model)
        self.assertFalse(ok)
        self.assertIn("not a vision projector", reason)

    def test_unreadable_file_does_not_block(self):
        junk = self.tmp / "junk.gguf"
        junk.write_bytes(b"definitely not a gguf")
        self.assertTrue(self.L.check_mmproj_pairing(self.model, junk)[0])


class GenericMmprojReuseTests(unittest.TestCase):
    """Unsloth ships almost every repo's projector as mmproj-F16/BF16.gguf, so a
    bare-filename search across other apps' model folders can return a different
    model's projector."""

    def setUp(self):
        import captioning_kit.llm_captioning as L
        self.L = L
        self.root = Path(tempfile.mkdtemp())
        self.repo = "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF"
        self.stranger = self.root / "lmstudio" / "SomeOtherModel" / "mmproj-BF16.gguf"
        self.stranger.parent.mkdir(parents=True)
        self.stranger.write_bytes(b"")
        self.correct = (self.root / "lmstudio" / "unsloth"
                        / "Qwen3-VL-30B-A3B-Instruct-GGUF" / "mmproj-BF16.gguf")
        self.correct.parent.mkdir(parents=True)
        self.correct.write_bytes(b"")
        self.settings = L.CaptioningSettings()
        self.settings.models_dir = str(self.root / "app")
        self.settings.extra_model_dirs = str(self.root / "lmstudio")

    def _locate(self, filename):
        return self.L.locate_existing_model_file(self.settings, self.repo, filename)

    def test_prefers_the_copy_under_a_matching_repo_folder(self):
        self.assertEqual(self._locate("mmproj-BF16.gguf"), self.correct)

    def test_refuses_an_unattributable_generic_projector(self):
        self.correct.unlink()
        self.assertIsNone(self._locate("mmproj-BF16.gguf"))

    def test_specific_projector_names_are_still_found_anywhere(self):
        named = self.root / "lmstudio" / "misc" / "mmproj-Qwen3.5-9B-Aggressive-BF16.gguf"
        named.parent.mkdir(parents=True, exist_ok=True)
        named.write_bytes(b"")
        self.assertEqual(
            self.L.locate_existing_model_file(self.settings, self.repo, named.name), named)

    def test_model_files_keep_the_loose_search(self):
        model = self.root / "lmstudio" / "anywhere" / "Model-UD-Q4_K_XL.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"")
        self.assertEqual(
            self.L.locate_existing_model_file(self.settings, self.repo, model.name), model)

    def test_generic_name_detection(self):
        for name in ("mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj.gguf"):
            self.assertTrue(self.L.is_generic_mmproj_name(name), name)
        for name in ("mmproj-Qwen3.5-9B-BF16.gguf", "model-Q4.gguf"):
            self.assertFalse(self.L.is_generic_mmproj_name(name), name)

    def test_common_words_are_not_treated_as_identifying(self):
        """'gguf' appears in every filename; matching on it would defeat the check."""
        self.assertNotIn("gguf", self.L._distinctive_repo_tokens(self.repo))
        self.assertIn("unsloth", self.L._distinctive_repo_tokens(self.repo))
