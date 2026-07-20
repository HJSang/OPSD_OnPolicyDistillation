import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from softtoken.prompting import (  # noqa: E402
    SOFT_CONTEXT_MARKER,
    render_user_prompt,
    resolve_prompt_format,
    split_chat_prompt,
)


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(
            self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return (
            "<|start|>user\n"
            + messages[0]["content"]
            + "<|end|>\n<|start|>assistant\n"
        )


class PromptingTest(unittest.TestCase):
    def test_soft_split_matches_raw_rendering(self):
        tokenizer = FakeTokenizer()
        context = "the document text"
        user_suffix = "\nQuestion: What is it?"

        left, right = split_chat_prompt(
            tokenizer, SOFT_CONTEXT_MARKER + user_suffix)
        raw = render_user_prompt(tokenizer, context + user_suffix)

        self.assertEqual(left + context + right, raw)
        self.assertIn("<|start|>user", left)
        self.assertIn("<|start|>assistant", right)

    def test_marker_must_appear_once(self):
        tokenizer = FakeTokenizer()
        with self.assertRaises(ValueError):
            split_chat_prompt(tokenizer, "no marker")
        with self.assertRaises(ValueError):
            split_chat_prompt(
                tokenizer, SOFT_CONTEXT_MARKER + SOFT_CONTEXT_MARKER)

    def test_missing_template_fails_clearly(self):
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "no chat_template"):
            render_user_prompt(tokenizer, "hello")

    def test_auto_preserves_old_checkpoint_format(self):
        self.assertEqual(resolve_prompt_format("auto", None), "chat")
        self.assertEqual(resolve_prompt_format("auto", {}), "plain")
        self.assertEqual(
            resolve_prompt_format("auto", {"prompt_format": "chat"}), "chat")
        self.assertEqual(resolve_prompt_format("plain", None), "plain")


if __name__ == "__main__":
    unittest.main()
