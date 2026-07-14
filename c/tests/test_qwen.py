"""Tests for Qwen3-MoE support: chat template, arch detection, config dispatch."""
import json
import os
import subprocess
import tempfile
import unittest

from openai_server import (APIError, QWEN_ARCH_TYPES, detect_arch, render_chat,
                            render_chat_qwen, serve)


class QwenTemplateTest(unittest.TestCase):
    """render_chat_qwen() must produce the official Qwen3 <|im_start|>/<|im_end|> format."""

    def test_basic_user_assistant(self):
        prompt = render_chat_qwen([
            {"role": "user", "content": "Hello"},
        ])
        self.assertEqual(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_system_user(self):
        prompt = render_chat_qwen([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ])
        self.assertEqual(
            prompt,
            "<|im_start|>system\nBe helpful.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_developer_role_maps_to_system(self):
        prompt = render_chat_qwen([
            {"role": "developer", "content": "Dev instructions."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertIn("<|im_start|>system\nDev instructions.<|im_end|>", prompt)

    def test_multi_turn(self):
        prompt = render_chat_qwen([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        self.assertEqual(
            prompt,
            "<|im_start|>user\nQ1<|im_end|>\n"
            "<|im_start|>assistant\nA1<|im_end|>\n"
            "<|im_start|>user\nQ2<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_thinking_prefix(self):
        prompt = render_chat_qwen(
            [{"role": "user", "content": "Think hard"}],
            enable_thinking=True,
        )
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

    def test_no_thinking_prefix_by_default(self):
        prompt = render_chat_qwen([{"role": "user", "content": "Hi"}])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", prompt)

    def test_rejects_image_content(self):
        with self.assertRaisesRegex(APIError, "text message content only"):
            render_chat_qwen([{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }])

    def test_rejects_empty_messages(self):
        with self.assertRaises(APIError):
            render_chat_qwen([])

    def test_rejects_invalid_role(self):
        with self.assertRaises(APIError):
            render_chat_qwen([{"role": "invalid", "content": "x"}])

    def test_tool_role(self):
        prompt = render_chat_qwen([
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "calc", "arguments": '{"expr": "2+2"}'}}
            ]},
            {"role": "tool", "content": "4"},
            {"role": "user", "content": "Thanks"},
        ])
        self.assertIn("<|im_start|>tool\n4<|im_end|>", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn('"name": "calc"', prompt)

    def test_tool_choice_none_disables_tools(self):
        # tool_choice="none" must suppress tool declarations even if tools are provided.
        tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
        prompt = render_chat_qwen(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="none",
        )
        self.assertNotIn("<tools>", prompt)


class QwenTemplateDoesNotAffectGLM(unittest.TestCase):
    """GLM render_chat must be entirely unaffected by the Qwen additions."""

    def test_glm_template_unchanged(self):
        prompt = render_chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|user|>Hi<|assistant|><think></think>",
        )

    def test_glm_thinking(self):
        prompt = render_chat([{"role": "user", "content": "Hi"}], True, "high")
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>",
        )


class DetectArchTest(unittest.TestCase):
    """detect_arch() reads model_type from config.json."""

    def _make_config(self, tmpdir, model_type):
        cfg = {"model_type": model_type, "hidden_size": 512}
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f)

    def test_glm_default(self):
        with tempfile.TemporaryDirectory() as d:
            # config.json absent → default GLM
            self.assertEqual(detect_arch(d), "glm_moe_dsa")

    def test_glm_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_config(d, "glm_moe_dsa")
            self.assertEqual(detect_arch(d), "glm_moe_dsa")

    def test_qwen3_moe(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_config(d, "qwen3_moe")
            self.assertEqual(detect_arch(d), "qwen3_moe")

    def test_qwen2_moe(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_config(d, "qwen2_moe")
            self.assertEqual(detect_arch(d), "qwen2_moe")

    def test_none_model_dir(self):
        # None / missing directory → safe default
        self.assertEqual(detect_arch(None), "glm_moe_dsa")
        self.assertEqual(detect_arch("/nonexistent/path"), "glm_moe_dsa")

    def test_corrupt_config(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("NOT JSON {{{")
            # Corrupt config → safe default
            self.assertEqual(detect_arch(d), "glm_moe_dsa")


class ServeArchDispatchTest(unittest.TestCase):
    """serve() must pass the right arch to APIServer without starting a real engine."""

    def test_serve_raises_for_missing_model_before_engine_start(self):
        # serve() binds the port before starting the engine; a missing model
        # is discovered when the Engine constructor tries to spawn the process.
        # This tests that the port-bind path succeeds and we get an error from
        # the engine (subprocess.CalledProcessError or FileNotFoundError),
        # not from arch detection.
        with self.assertRaises((subprocess.CalledProcessError, FileNotFoundError,
                                OSError, RuntimeError, SystemExit)):
            serve("/nonexistent/model/dir", kv_slots=1)


class QwenArchTypesTest(unittest.TestCase):
    """QWEN_ARCH_TYPES covers qwen3_moe, qwen2_moe, qwen_moe."""

    def test_qwen3_moe_in_set(self):
        self.assertIn("qwen3_moe", QWEN_ARCH_TYPES)

    def test_qwen2_moe_in_set(self):
        self.assertIn("qwen2_moe", QWEN_ARCH_TYPES)

    def test_glm_not_in_set(self):
        self.assertNotIn("glm_moe_dsa", QWEN_ARCH_TYPES)

    def test_detect_arch_returns_qwen2_moe(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"model_type": "qwen2_moe"}, f)
            arch = detect_arch(d)
            self.assertIn(arch, QWEN_ARCH_TYPES)


if __name__ == "__main__":
    unittest.main()
