from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from qwen35_mtp.export import MTPCoreExportWrapper, export_mtp_core_onnx
from test_mtp_math import initialized_drafter


class OnnxExportTest(unittest.TestCase):
    def test_tiny_core_export_matches_pytorch(self):
        model = initialized_drafter()
        wrapper = MTPCoreExportWrapper(model.mtp)
        c = model.config
        with tempfile.TemporaryDirectory(prefix="qwen35-mtp-onnx-") as directory:
            path = export_mtp_core_onnx(
                wrapper,
                Path(directory) / "mtp.onnx",
                sequence_length=2,
                past_length=1,
                hidden_size=c.hidden_size,
                kv_heads=c.num_key_value_heads,
                head_dim=c.head_dim,
                dtype=torch.float32,
            )
            onnx.checker.check_model(str(path), full_check=True)
            inputs = {
                "inputs_embeds": np.zeros((1, 2, c.hidden_size), np.float32),
                "hidden_sources": np.zeros((1, 2, c.hidden_size), np.float32),
                "position_ids": np.array([[1, 2]], np.int64),
                "past_key": np.zeros(
                    (1, c.num_key_value_heads, 1, c.head_dim), np.float32
                ),
                "past_value": np.zeros(
                    (1, c.num_key_value_heads, 1, c.head_dim), np.float32
                ),
            }
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, inputs)
            with torch.inference_mode():
                expected = wrapper(*(torch.from_numpy(inputs[name]) for name in inputs))
            for target, observed in zip(expected, actual):
                np.testing.assert_allclose(
                    target.detach().numpy(), observed, rtol=1e-4, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
