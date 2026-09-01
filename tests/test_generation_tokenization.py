from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from transformers.tokenization_utils_base import BatchEncoding


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.generation import tokenize_prompt


class _Tokenizer:
    def __init__(self, *, chat_output=None, encode_output=None):
        self.chat_output = chat_output
        self.encode_output = encode_output

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tokenize is True
        assert add_generation_prompt is True
        return self.chat_output

    def encode(self, prompt, *, add_special_tokens):
        assert prompt == "hello"
        assert add_special_tokens is True
        return self.encode_output


def test_tokenize_prompt_extracts_input_ids_from_batch_encoding() -> None:
    output = BatchEncoding(
        {
            "input_ids": [248045, 846, 198],
            "attention_mask": [1, 1, 1],
        }
    )
    assert not isinstance(output, dict)

    assert tokenize_prompt(
        _Tokenizer(chat_output=output), "hello", chat=True
    ) == [248045, 846, 198]


def test_tokenize_prompt_accepts_attribute_and_single_batch_outputs() -> None:
    output = SimpleNamespace(input_ids=((101, 102, 103),))

    assert tokenize_prompt(
        _Tokenizer(chat_output=output), "hello", chat=True
    ) == [101, 102, 103]


def test_tokenize_prompt_rejects_mapping_without_input_ids() -> None:
    with pytest.raises(ValueError, match="does not contain input_ids"):
        tokenize_prompt(
            _Tokenizer(chat_output={"attention_mask": [1]}),
            "hello",
            chat=True,
        )


def test_tokenize_prompt_rejects_multiple_sequences() -> None:
    with pytest.raises(ValueError, match="only batch size 1"):
        tokenize_prompt(
            _Tokenizer(encode_output=[[1, 2], [3, 4]]),
            "hello",
            chat=False,
        )
