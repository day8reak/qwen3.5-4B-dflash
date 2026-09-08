"""Prompt IDs must retain chat formatting across tokenizer return containers."""

from collections import UserDict
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BatchEncoding, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p.generation import tokenize_prompt
from models.dflash_v1.dflash_qwen_adapter_v1 import _tokenize_prompt_text


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "assistant": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=(
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("return_tensors", [None, "pt", "np"])
def test_chat_preserves_prompt_and_generation_marker(tokenizer, return_dict, return_tensors):
    def apply_chat_template(messages, **kwargs):
        encoded = tokenizer.apply_chat_template(
            messages, **kwargs, return_dict=return_dict, return_tensors=return_tensors
        )
        if return_dict:
            # BatchEncoding is a Mapping, but is not a dict. Fast-tokenizer
            # integer indexing accesses Encoding objects rather than token IDs.
            assert isinstance(encoded, BatchEncoding)
            assert not isinstance(encoded, dict)
        return encoded

    adapter = SimpleNamespace(apply_chat_template=apply_chat_template)
    assert tokenize_prompt(adapter, "hello world", chat=True) == [1, 2, 3]


@pytest.mark.parametrize("mapping", [dict, UserDict])
def test_chat_ignores_other_mapping_fields(mapping):
    encoded = mapping(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])
    adapter = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: encoded)
    assert tokenize_prompt(adapter, "hello world", chat=True) == [1, 2, 3]


def test_raw_prompt_has_no_chat_generation_marker(tokenizer):
    assert tokenize_prompt(tokenizer, "hello world", chat=False) == [1, 2]


@pytest.mark.parametrize("chat", [False, True])
def test_framework_and_native_npu_prompt_ids_match(tokenizer, tmp_path, chat):
    tokenizer.save_pretrained(tmp_path)
    native_ids, _ = _tokenize_prompt_text(
        "hello world", target_root=tmp_path, prompt_mode="chat" if chat else "raw"
    )
    assert tokenize_prompt(tokenizer, "hello world", chat=chat) == native_ids


def test_mapping_without_input_ids_is_rejected():
    adapter = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {"attention_mask": [1]}
    )
    with pytest.raises(ValueError, match="missing input_ids"):
        tokenize_prompt(adapter, "hello", chat=True)


@pytest.mark.parametrize(
    "input_ids, error",
    [([], "empty sequence"), ([[]], "empty sequence"), ([[1], [2]], "batch size 1")],
)
def test_mapping_keeps_empty_and_batch_validation(input_ids, error):
    adapter = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: BatchEncoding({"input_ids": input_ids})
    )
    with pytest.raises(ValueError, match=error):
        tokenize_prompt(adapter, "hello", chat=True)
