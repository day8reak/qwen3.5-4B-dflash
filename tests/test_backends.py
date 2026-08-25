from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from qwen35_mtp.backends import TransformersMainBackend


class DummyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.calls = 0

    def forward(self, input_ids, use_cache, return_dict):
        self.calls += 1
        self.assertions = (use_cache, return_dict)
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


class BackendTest(unittest.TestCase):
    def test_exact_sequence_hidden_cache_does_not_change_projection(self):
        torch.manual_seed(7)
        text = DummyTextModel()
        lm_head = nn.Linear(4, 16, bias=False)
        backend = TransformersMainBackend(
            nn.Module(), text, lm_head, device=torch.device("cpu")
        )
        ids = torch.tensor([[1, 2, 3]])
        first = backend.evaluate(ids, [2])
        second = backend.evaluate(ids, [1, 2])
        self.assertEqual(text.calls, 1)
        self.assertEqual(first.top1_token_ids.shape, (1, 1))
        self.assertEqual(second.top1_token_ids.shape, (1, 2))
        backend.evaluate(torch.tensor([[1, 2, 4]]), [2])
        self.assertEqual(text.calls, 2)
        backend.clear_cache()
        backend.evaluate(ids, [2])
        self.assertEqual(text.calls, 3)


if __name__ == "__main__":
    unittest.main()
