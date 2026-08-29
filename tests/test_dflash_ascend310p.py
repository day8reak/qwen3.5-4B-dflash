from __future__ import annotations

import hashlib
import os
from pathlib import Path
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from qwen35_dflash.ascend310p.acl_runtime import AclOmRuntime
from qwen35_dflash.ascend310p.cli import build_parser, command_build
from qwen35_dflash.ascend310p.compiler import (
    compile_air_bundle,
    validate_soc_version,
)
from qwen35_dflash.ascend310p.contracts import AirGraphSpec, GenerationStep
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.factories import create_integrated_recompute_graph
from qwen35_dflash.ascend310p.generation import (
    benchmark_prompt,
    generate_prompt,
    load_backend,
    verify_ordinary_reference,
)
from qwen35_dflash.ascend310p.integrated import (
    IntegratedDFlashRecomputeGraph,
    integrated_recompute_graph_spec,
)
from qwen35_dflash.ascend310p.recompute_backend import RecomputeDFlashOmBackend
from qwen35_dflash.ascend310p.resources import resolve_locked_data
from qwen35_dflash.ascend310p.target_adapter import TransformersDFlashTargetAdapter
from qwen35_dflash.ascend310p.workflow import (
    DEFAULT_BACKEND_FACTORY,
    DEFAULT_GRAPH_FACTORY,
    preflight_target_pipeline,
    run_target_pipeline,
    validate_backend_pair,
)


class _FakeTorchAir:
    __version__ = "test-torchair"

    @staticmethod
    def dynamo_export(*args, **kwargs):
        del args
        export_path = Path(kwargs["export_path"])
        export_path.joinpath(kwargs["export_name"] + ".air").write_bytes(b"AIR")
        export_path.joinpath("external-weight").write_bytes(b"WEIGHT")


def _graph_factory(config):
    return (
        AirGraphSpec(
            name=config.get("name", "prefill"),
            role="prefill",
            model=torch.nn.Identity(),
            example_args=(torch.ones(1, 2),),
            input_names=("input_ids",),
            output_names=("next_token",),
            metadata={"fixture": True},
        ),
    )


class _ProbeGraph(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        return input_ids + 1, torch.tensor([[7, 8]], dtype=torch.long)


def _probe_graph_factory(config):
    del config
    values = torch.zeros((1, 4), dtype=torch.long)
    return (
        AirGraphSpec(
            name="dflash_recompute",
            role="generation-recompute",
            model=_ProbeGraph(),
            example_args=(values, torch.zeros_like(values)),
            input_names=("input_ids", "attention_mask"),
            output_names=("target_top1", "draft_top1"),
            metadata={"fixture": True},
        ),
    )


class _Tokenizer:
    eos_token_id = 2

    def encode(self, prompt, add_special_tokens=True):
        self.last_prompt = prompt
        self.last_add_special_tokens = add_special_tokens
        return [7, 8]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_chat_options = (tokenize, add_generation_prompt)
        return [9, 10]

    def decode(self, tokens, skip_special_tokens=True):
        self.last_decode_options = skip_special_tokens
        return " ".join(str(token) for token in tokens if token != 2)


class _Backend:
    backend_id = "fake-om-backend"

    def __init__(
        self,
        *,
        fallback=False,
        target=True,
        artifacts=None,
        generation_mode="dflash-strict-greedy",
    ):
        self.fallback = fallback
        self.target = target
        self.artifacts = artifacts or {"prefill": "a" * 64, "decode": "b" * 64}
        self.decode_index = 0
        self.reset_calls = 0
        self.sync_calls = 0
        self.closed = False
        self.generation_mode = generation_mode

    def metadata(self):
        return {
            "cpu_fallback": self.fallback,
            "artifacts": self.artifacts,
            "device": {
                "target_id": "ascend310p" if self.target else "cpu-simulation",
                "model": "Ascend310P3" if self.target else "CPU",
                "device_id": 0,
            },
            "cann": "test-cann",
            "driver": "test-driver",
            "firmware": "test-firmware",
            "runtime": "test-runtime",
            "generation_mode": self.generation_mode,
        }

    def synchronize(self):
        self.sync_calls += 1

    def reset(self):
        self.reset_calls += 1
        self.decode_index = 0

    def prefill(self, prompt_token_ids, *, max_new_tokens, eos_token_ids):
        self.prefill_args = (list(prompt_token_ids), max_new_tokens, list(eos_token_ids))
        return GenerationStep(token_ids=(11,), metadata={"graph": "prefill"})

    def decode(
        self,
        committed_token_ids,
        *,
        max_new_tokens,
        max_draft_tokens,
        eos_token_ids,
    ):
        self.decode_args = (
            list(committed_token_ids),
            max_new_tokens,
            max_draft_tokens,
            list(eos_token_ids),
        )
        self.decode_index += 1
        if self.decode_index == 1:
            return GenerationStep(
                token_ids=(12,),
                drafted_tokens=2,
                accepted_draft_tokens=1,
                rejected_draft_tokens=1,
                metadata={"graph": "draft+verify"},
            )
        return GenerationStep(token_ids=(2,), finished=True)

    def close(self):
        self.closed = True


_LAST_FACTORY_BACKEND = None


def _manifest_backend_factory(*, bundle_dir, manifest, device_id, options):
    del bundle_dir, device_id
    expected = {
        graph["name"]: graph["om"]["sha256"] for graph in manifest["graphs"]
    }
    if options.get("mismatch"):
        expected = {"wrong": "c" * 64}
    backend = _Backend(
        artifacts=expected,
        generation_mode=(
            "ordinary-greedy"
            if options.get("ordinary_only") is True
            else "dflash-strict-greedy"
        ),
    )
    global _LAST_FACTORY_BACKEND
    _LAST_FACTORY_BACKEND = backend
    return backend


class _FakeAclRt:
    def __init__(self, owner):
        self.owner = owner
        self.next_pointer = 100

    def set_device(self, device_id):
        self.owner.device_id = device_id
        return 0

    def reset_device(self, device_id):
        self.owner.reset_device_id = device_id
        return 0

    def synchronize_device(self):
        self.owner.syncs += 1
        return 0

    def malloc(self, size, policy):
        del policy
        pointer = self.next_pointer
        self.next_pointer += 1
        self.owner.memory[pointer] = bytearray(size)
        return pointer, 0

    def free(self, pointer):
        self.owner.memory.pop(pointer, None)
        return 0

    def memcpy(self, destination, destination_size, source, count, kind):
        self.owner.copy_kinds.append(kind)
        if kind == self.owner.ACL_MEMCPY_HOST_TO_DEVICE:
            payload = np.asarray(source).tobytes()
            self.owner.memory[destination][:count] = payload[:count]
        else:
            payload = self.owner.memory[source][:count]
            values = np.frombuffer(payload, dtype=destination.dtype).reshape(destination.shape)
            np.copyto(destination, values)
        self.owner.last_copy_size = destination_size
        return 0


class _FakeAclMdl:
    def __init__(self, owner):
        self.owner = owner

    def load_from_file(self, path):
        self.owner.loaded_path = path
        return 1, 0

    def unload(self, model_id):
        self.owner.unloaded = model_id
        return 0

    def create_desc(self):
        return object()

    def destroy_desc(self, desc):
        del desc
        return 0

    def get_desc(self, desc, model_id):
        del desc, model_id
        return 0

    def get_num_inputs(self, desc):
        del desc
        return 1

    def get_num_outputs(self, desc):
        del desc
        return 1

    def get_input_size_by_index(self, desc, index):
        del desc, index
        return 8

    def get_output_size_by_index(self, desc, index):
        del desc, index
        return 8

    def get_input_dims(self, desc, index):
        del desc, index
        return {"dims": [1, 2]}, 0

    def get_output_dims(self, desc, index):
        del desc, index
        return {"dims": [1, 2]}, 0

    def get_input_data_type(self, desc, index):
        del desc, index
        return self.owner.ACL_FLOAT

    def get_output_data_type(self, desc, index):
        del desc, index
        return self.owner.ACL_FLOAT

    def get_input_name_by_index(self, desc, index):
        del desc, index
        return "x"

    def get_output_name_by_index(self, desc, index):
        del desc, index
        return "y"

    def create_dataset(self):
        return []

    def destroy_dataset(self, dataset):
        del dataset
        return 0

    def add_dataset_buffer(self, dataset, buffer):
        dataset.append(buffer)
        return dataset, 0

    def execute(self, model_id, inputs, outputs):
        del model_id
        input_pointer = inputs[0][0]
        output_pointer = outputs[0][0]
        values = np.frombuffer(self.owner.memory[input_pointer], dtype=np.float32)
        self.owner.memory[output_pointer][:] = (values * 2).astype(np.float32).tobytes()
        return 0


class _FakeAclUtil:
    @staticmethod
    def numpy_to_ptr(value):
        return value


class _FakeAcl:
    ACL_FLOAT = 0
    ACL_MEM_MALLOC_HUGE_FIRST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self):
        self.memory = {}
        self.copy_kinds = []
        self.syncs = 0
        self.rt = _FakeAclRt(self)
        self.mdl = _FakeAclMdl(self)
        self.util = _FakeAclUtil()
        self.finalized = False

    def init(self):
        return 0

    def finalize(self):
        self.finalized = True
        return 0

    @staticmethod
    def create_data_buffer(pointer, size):
        return (pointer, size)

    @staticmethod
    def destroy_data_buffer(buffer):
        del buffer
        return 0


class _IntegratedTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.output = torch.nn.Linear(4, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output

    def forward(self, *, input_ids, attention_mask, **kwargs):
        self.last_attention_mask = attention_mask
        self.last_kwargs = kwargs
        logits = torch.zeros((*input_ids.shape, 32), dtype=torch.float32)
        logits.scatter_(-1, ((input_ids + 1) % 32).unsqueeze(-1), 1.0)
        features = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
        return SimpleNamespace(logits=logits, dflash_features=features)


class _IntegratedDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            block_size=3,
            mask_token_id=31,
            vocab_size=32,
            hidden_size=4,
        )

    def embed_block(self, block_ids, embedding_weight):
        self.block_ids = block_ids.detach().clone()
        return torch.nn.functional.embedding(block_ids, embedding_weight)

    def draft_top1(
        self,
        target_hidden,
        noise_embedding,
        position_ids,
        lm_head_weight,
        *,
        context_attention_mask,
    ):
        del target_hidden, noise_embedding, lm_head_weight
        self.position_ids = position_ids.detach().clone()
        self.context_attention_mask = context_attention_mask.detach().clone()
        return torch.tensor([[7, 8]], dtype=torch.long)


class _RecomputeRuntime:
    def __init__(self, *, reject_second=False):
        self.reject_second = reject_second
        self.calls = []
        self.syncs = 0
        self.closed = False

    def graph_inputs(self, name):
        self.graph_name = name
        return (
            {"name": "input_ids", "shape": [1, 12], "dtype": "int64"},
            {"name": "attention_mask", "shape": [1, 12], "dtype": "int64"},
        )

    def graph_outputs(self, name):
        self.graph_name = name
        return (
            {"name": "target_top1", "shape": [1, 12], "dtype": "int64"},
            {"name": "draft_top1", "shape": [1, 2], "dtype": "int64"},
        )

    def artifact_hashes(self):
        return {"dflash_recompute": "d" * 64}

    def run_graph(self, name, inputs):
        self.calls.append((name, {key: value.copy() for key, value in inputs.items()}))
        length = int(inputs["attention_mask"].sum())
        values = inputs["input_ids"][0, :length].astype(np.int64)
        target = np.zeros((1, 12), dtype=np.int64)
        target[0, :length] = values + 1
        proposals = np.array([[values[-1] + 1, values[-1] + 2]], dtype=np.int64)
        if self.reject_second:
            proposals[0, 1] += 50
        return {"target_top1": target, "draft_top1": proposals}

    def synchronize(self):
        self.syncs += 1

    def close(self):
        self.closed = True


class _AdapterLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(20, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        self.last_attention_mask = attention_mask
        self.last_kwargs = kwargs
        embedding = self.embed_tokens(input_ids)
        hidden_states = tuple(embedding + float(index) for index in range(4))
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
        )


class DFlashAscend310PDeploymentTest(unittest.TestCase):
    def _run_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def _export(self, run_root: Path):
        bundle = run_root / "out" / "bundle"
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            manifest = export_air_bundle(
                _graph_factory,
                {"name": "target_prefill"},
                bundle,
                torchair_module=_FakeTorchAir,
            )
        return bundle, manifest

    def _deployment_manifest(self, run_root: Path):
        bundle = run_root / "out" / "load-backend"
        om_dir = bundle / "om"
        om_dir.mkdir(parents=True)
        om_path = om_dir / "graph.om"
        om_path.write_bytes(b"graph-om")
        import hashlib

        digest = hashlib.sha256(b"graph-om").hexdigest()
        manifest = {
            "schema_version": 1,
            "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
            "status": "PASS",
            "graphs": [
                {
                    "name": "graph",
                    "role": "generation",
                    "om": {"path": "om/graph.om", "bytes": 8, "sha256": digest},
                }
            ],
        }
        manifest_path = bundle / "deployment-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_air_export_is_hash_complete_and_run_scoped(self):
        run_root = self._run_root()
        bundle, manifest = self._export(run_root)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["graphs"][0]["role"], "prefill")
        self.assertEqual(len(manifest["graphs"][0]["payload_files"]), 2)
        self.assertTrue((bundle / "air-manifest.json").is_file())
        self.assertTrue((bundle / manifest["graphs"][0]["air"]["path"]).is_file())

    def test_air_export_rejects_output_outside_active_run(self):
        run_root = self._run_root()
        outside = self._run_root() / "bundle"
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with self.assertRaisesRegex(RuntimeError, "below AI_RUN_DIR"):
                export_air_bundle(
                    _graph_factory,
                    {},
                    outside,
                    torchair_module=_FakeTorchAir,
                )

    def test_air_export_checks_torchair_before_loading_factory(self):
        run_root = self._run_root()
        bundle = run_root / "out" / "bundle"
        factory_called = False

        def expensive_factory(config):
            del config
            nonlocal factory_called
            factory_called = True
            return _graph_factory({})

        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with mock.patch(
                "qwen35_dflash.ascend310p.exporter.importlib.import_module",
                side_effect=ImportError("no torchair"),
            ):
                with self.assertRaisesRegex(RuntimeError, "TorchAir is required"):
                    export_air_bundle(expensive_factory, {}, bundle)
        self.assertFalse(factory_called)
        self.assertFalse(bundle.exists())

    def test_atc_compile_uses_framework_one_and_hashes_om(self):
        run_root = self._run_root()
        bundle, _manifest = self._export(run_root)
        commands = []

        def runner(command, cwd):
            commands.append((list(command), cwd))
            output = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
            Path(output + ".om").write_bytes(b"OM")
            return subprocess.CompletedProcess(command, 0, "ATC run success")

        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            deployment = compile_air_bundle(
                bundle / "air-manifest.json",
                soc_version="Ascend310P3",
                atc_bin="/bin/true",
                extra_args=("--precision_mode=force_fp16",),
                runner=runner,
                atc_identity="fake-atc",
            )
        command = commands[0][0]
        self.assertIn("--framework=1", command)
        self.assertIn("--soc_version=Ascend310P3", command)
        self.assertIn("--precision_mode=force_fp16", command)
        self.assertEqual(deployment["status"], "PASS")
        self.assertEqual(
            deployment["graphs"][0]["om"]["sha256"],
            "7469366c72362c3a23450fda3394230498dd7f0477025589044cfefd7ec9fa51",
        )

    def test_atc_compile_detects_tampered_air_before_invocation(self):
        run_root = self._run_root()
        bundle, manifest = self._export(run_root)
        air = bundle / manifest["graphs"][0]["air"]["path"]
        air.write_bytes(b"tampered")
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with self.assertRaisesRegex(ValueError, "(size|hash) mismatch"):
                compile_air_bundle(
                    bundle / "air-manifest.json",
                    soc_version="Ascend310P3",
                    atc_bin="/bin/true",
                    runner=lambda command, cwd: self.fail("ATC must not run"),
                    atc_identity="fake-atc",
                )

    def test_atc_compile_detects_tampered_external_weight(self):
        run_root = self._run_root()
        bundle, manifest = self._export(run_root)
        weight = next(
            item
            for item in manifest["graphs"][0]["payload_files"]
            if item["path"].endswith("external-weight")
        )
        (bundle / weight["path"]).write_bytes(b"changed")
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with self.assertRaisesRegex(ValueError, "payload (size|hash) mismatch"):
                compile_air_bundle(
                    bundle / "air-manifest.json",
                    soc_version="Ascend310P3",
                    atc_bin="/bin/true",
                    runner=lambda command, cwd: self.fail("ATC must not run"),
                    atc_identity="fake-atc",
                )

    def test_atc_core_options_cannot_be_overridden(self):
        run_root = self._run_root()
        bundle, _manifest = self._export(run_root)
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with self.assertRaisesRegex(ValueError, "cannot be overridden"):
                compile_air_bundle(
                    bundle / "air-manifest.json",
                    soc_version="Ascend310P3",
                    atc_bin="/bin/true",
                    extra_args=("--framework=5",),
                    atc_identity="fake-atc",
                )

    def test_generic_310p_soc_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "concrete 310P ATC variant"):
            validate_soc_version("Ascend310P")

    def test_build_om_resolves_atc_before_exporting_factory(self):
        args = SimpleNamespace(
            atc=Path("/definitely/missing/atc"),
            soc_version="Ascend310P3",
            factory=f"{__name__}:_graph_factory",
            factory_config=None,
            bundle_dir=Path("/unused"),
            atc_arg=[],
        )
        with mock.patch(
            "qwen35_dflash.ascend310p.cli.export_air_bundle"
        ) as exporter:
            with self.assertRaisesRegex(RuntimeError, "not an executable"):
                command_build(args)
        exporter.assert_not_called()

    def test_target_pipeline_runs_declared_device_preflight_before_imports(self):
        run_root = self._run_root()
        completed = subprocess.CompletedProcess(
            ["preflight"], 0, "strict target preflight PASS\n"
        )
        environment = {
            "AI_RUN_DIR": str(run_root),
            "AI_MODEL_ROOT": str(run_root),
            "AI_TARGET_PREFLIGHT": "/bin/true",
        }
        with mock.patch.dict(os.environ, environment):
            with mock.patch(
                "qwen35_dflash.ascend310p.workflow.subprocess.run",
                return_value=completed,
            ) as runner, mock.patch(
                "qwen35_dflash.ascend310p.workflow._require_importable"
            ) as require_importable:
                atc, soc, log = preflight_target_pipeline(
                    factory="custom.graph:create",
                    factory_config={},
                    backend="custom.backend:create",
                    atc_bin=Path("/bin/true"),
                    soc_version="Ascend310P3",
                )
        self.assertEqual(atc, Path("/bin/true").resolve())
        self.assertEqual(soc, "Ascend310P3")
        self.assertEqual(log.read_text(encoding="utf-8"), completed.stdout)
        command = runner.call_args.args[0]
        self.assertIn("--require-atc", command)
        self.assertIn("--require-device", command)
        require_importable.assert_called_once_with("torchair", "AIR export")

    def test_prompt_generation_reports_synchronized_stage_latency(self):
        backend = _Backend()
        report = generate_prompt(
            backend,
            _Tokenizer(),
            "hello",
            max_new_tokens=3,
            max_draft_tokens=2,
        )
        self.assertEqual(report["generated_token_ids"], [11, 12, 2])
        self.assertEqual(report["stop_reason"], "eos")
        self.assertEqual(report["counters"]["decode_iterations"], 2)
        self.assertEqual(report["counters"]["drafted_tokens"], 2)
        self.assertEqual(len(report["decode_iterations"]), 2)
        self.assertGreaterEqual(report["latency_ms"]["model_total"], 0.0)
        self.assertEqual(backend.sync_calls, 6)

    def test_target_generation_rejects_any_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "cpu_fallback=false"):
            generate_prompt(_Backend(fallback=True), _Tokenizer(), "hello")

    def test_non_target_metadata_needs_explicit_simulation_mode(self):
        with self.assertRaisesRegex(RuntimeError, "device.target_id"):
            generate_prompt(_Backend(target=False), _Tokenizer(), "hello")
        report = generate_prompt(
            _Backend(target=False),
            _Tokenizer(),
            "hello",
            max_new_tokens=1,
            require_target=False,
        )
        self.assertEqual(report["status"], "PASS")

    def test_benchmark_keeps_raw_repetitions_and_checks_tokens(self):
        backend = _Backend()
        report = benchmark_prompt(
            backend,
            _Tokenizer(),
            "hello",
            max_new_tokens=3,
            max_draft_tokens=2,
            warmup=1,
            repetitions=2,
            require_target=False,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["measurements"]), 2)
        self.assertEqual(report["latency_ms"]["prefill"]["count"], 2)
        self.assertEqual(backend.reset_calls, 3)

    def test_target_benchmark_requires_frozen_three_plus_ten_counts(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 warmups and 10"):
            benchmark_prompt(
                _Backend(),
                _Tokenizer(),
                "hello",
                warmup=0,
                repetitions=1,
            )

    def test_static_pyacl_runtime_loads_hash_locked_om_and_runs_named_io(self):
        run_root = self._run_root()
        bundle = run_root / "out" / "acl-bundle"
        om_dir = bundle / "om"
        om_dir.mkdir(parents=True)
        om_path = om_dir / "double.om"
        om_path.write_bytes(b"fake-om")
        import hashlib

        digest = hashlib.sha256(b"fake-om").hexdigest()
        manifest = {
            "schema_version": 1,
            "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
            "status": "PASS",
            "graphs": [
                {
                    "name": "double",
                    "role": "test",
                    "input_names": ["logical_x"],
                    "output_names": ["logical_y"],
                    "om": {"path": "om/double.om", "bytes": 7, "sha256": digest},
                }
            ],
        }
        manifest_path = bundle / "deployment-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        acl = _FakeAcl()
        runtime = AclOmRuntime(manifest_path, device_id=3, acl_module=acl)
        self.assertEqual(runtime.graph_names, ("double",))
        outputs = runtime.run_graph(
            "double", {"logical_x": np.array([[1.5, -2.0]], dtype=np.float32)}
        )
        np.testing.assert_array_equal(
            outputs["logical_y"], np.array([[3.0, -4.0]], dtype=np.float32)
        )
        self.assertEqual(runtime.graph_inputs("double")[0]["runtime_name"], "x")
        self.assertEqual(runtime.graph_outputs("double")[0]["runtime_name"], "y")
        runtime.synchronize()
        runtime.close()
        self.assertEqual(acl.syncs, 1)
        self.assertEqual(acl.reset_device_id, 3)
        self.assertTrue(acl.finalized)
        self.assertEqual(acl.memory, {})

    def test_infer_cli_defaults_to_three_warmups_and_ten_measurements(self):
        args = build_parser().parse_args(
            [
                "infer-om",
                "--deployment-manifest",
                "/run/deployment-manifest.json",
                "--backend",
                "existing.backend:create_backend",
                "--model-dir",
                "/checkpoint",
                "--prompt",
                "hello",
                "--output",
                "/run/report.json",
            ]
        )
        self.assertEqual(args.warmup, 3)
        self.assertEqual(args.repetitions, 10)
        self.assertFalse(args.allow_simulation)

    def test_run_e2e_cli_uses_locked_factories_and_draft_width(self):
        args = build_parser().parse_args(
            [
                "run-e2e",
                "--factory-config",
                "/run/factory.json",
                "--bundle-dir",
                "/run/bundle",
                "--soc-version",
                "Ascend310P3",
                "--ordinary-backend-config",
                "/run/ordinary.json",
                "--dflash-backend-config",
                "/run/dflash.json",
                "--prompt",
                "hello",
                "--report-dir",
                "/run/reports",
            ]
        )
        self.assertEqual(args.factory, DEFAULT_GRAPH_FACTORY)
        self.assertEqual(args.backend, DEFAULT_BACKEND_FACTORY)
        self.assertEqual(args.max_draft_tokens, 15)
        self.assertFalse(hasattr(args, "warmup"))

    def test_target_pipeline_requires_identical_backend_identity(self):
        ordinary = {"ordinary_only": True, "runtime": "acl-a"}
        dflash = {"ordinary_only": False, "runtime": "acl-b"}
        with self.assertRaisesRegex(ValueError, r"different fields=\['runtime'\]"):
            validate_backend_pair(ordinary, dflash)

    def test_target_pipeline_builds_then_runs_ordinary_and_dflash_three_plus_ten(self):
        run_root = self._run_root()
        bundle = run_root / "out" / "bundle"
        reports = run_root / "out" / "performance"

        def fake_export(factory, config, bundle_dir):
            return export_air_bundle(
                factory,
                config,
                bundle_dir,
                torchair_module=_FakeTorchAir,
            )

        def fake_compile(
            air_manifest,
            *,
            soc_version,
            atc_bin,
            extra_args,
        ):
            def runner(command, cwd):
                del cwd
                output = next(
                    item.split("=", 1)[1]
                    for item in command
                    if item.startswith("--output=")
                )
                Path(output + ".om").write_bytes(b"target-om")
                return subprocess.CompletedProcess(command, 0, "ATC PASS")

            return compile_air_bundle(
                air_manifest,
                soc_version=soc_version,
                atc_bin=atc_bin,
                extra_args=extra_args,
                runner=runner,
                atc_identity="fake-atc",
            )

        identity = {
            "graph_name": "graph",
            "pad_token_id": 0,
            "device_model": "Ascend310P3-TestProduct",
            "cann": "test-cann",
            "driver": "test-driver",
            "firmware": "test-firmware",
            "runtime": "test-pyacl",
        }
        preflight_log = run_root / "log" / "dflash-run-e2e-preflight.log"
        preflight_log.parent.mkdir(parents=True)
        preflight_log.write_text("target preflight PASS\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            with mock.patch(
                "qwen35_dflash.ascend310p.workflow.preflight_target_pipeline",
                return_value=(Path("/bin/true"), "Ascend310P3", preflight_log),
            ), mock.patch(
                "qwen35_dflash.ascend310p.workflow.export_air_bundle",
                side_effect=fake_export,
            ), mock.patch(
                "qwen35_dflash.ascend310p.workflow.compile_air_bundle",
                side_effect=fake_compile,
            ), mock.patch(
                "qwen35_dflash.ascend310p.workflow.load_tokenizer",
                return_value=(
                    _Tokenizer(),
                    {
                        "path": "/locked/checkpoint",
                        "asset_id": "qwen3.5-4b",
                        "manifest_sha256": "d" * 64,
                    },
                ),
            ):
                summary = run_target_pipeline(
                    factory=f"{__name__}:_graph_factory",
                    factory_config={"name": "graph"},
                    bundle_dir=bundle,
                    soc_version="Ascend310P3",
                    atc_bin=Path("/bin/true"),
                    atc_args=("--precision_mode=force_fp16",),
                    backend_factory=f"{__name__}:_manifest_backend_factory",
                    ordinary_backend_options={**identity, "ordinary_only": True},
                    dflash_backend_options={**identity, "ordinary_only": False},
                    report_dir=reports,
                    prompt="hello",
                    max_new_tokens=3,
                    max_draft_tokens=2,
                )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["ordinary_parity"]["token_id_mismatches"], 0)
        self.assertEqual(summary["output"]["token_ids"], [11, 12, 2])
        self.assertEqual(summary["output"]["text"], "11 12")
        self.assertEqual(summary["latency_ms"]["ordinary"]["prefill"]["count"], 10)
        self.assertEqual(summary["latency_ms"]["dflash"]["decode"]["count"], 10)
        self.assertTrue((reports / "ordinary.json").is_file())
        self.assertTrue((reports / "dflash.json").is_file())
        self.assertTrue((reports / "summary.json").is_file())

    def test_pytorch_probe_writes_run_scoped_real_graph_report(self):
        run_root = self._run_root()
        config = run_root / "config.json"
        config.write_text("{}", encoding="utf-8")
        output = run_root / "out" / "probe.json"
        args = build_parser().parse_args(
            [
                "probe-pytorch",
                "--factory",
                f"{__name__}:_probe_graph_factory",
                "--factory-config",
                str(config),
                "--input-token-ids",
                "3,4",
                "--output",
                str(output),
            ]
        )
        with mock.patch.dict(os.environ, {"AI_RUN_DIR": str(run_root)}):
            self.assertEqual(args.handler(args), 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["ordinary_next_token_id"], 5)
        self.assertEqual(report["draft_token_ids"], [7, 8])

    def test_backend_factory_must_report_exact_manifest_om_hashes(self):
        run_root = self._run_root()
        manifest_path = self._deployment_manifest(run_root)
        backend = load_backend(
            f"{__name__}:_manifest_backend_factory",
            manifest_path,
            device_id=0,
            options={},
        )
        self.assertEqual(set(backend.metadata()["artifacts"]), {"graph"})
        backend.close()

        with self.assertRaisesRegex(ValueError, "artifact identities differ"):
            load_backend(
                f"{__name__}:_manifest_backend_factory",
                manifest_path,
                device_id=0,
                options={"mismatch": True},
            )
        self.assertTrue(_LAST_FACTORY_BACKEND.closed)

    def test_integrated_graph_uses_right_padding_anchor_and_logical_positions(self):
        target = _IntegratedTarget()
        draft = _IntegratedDraft()
        graph = IntegratedDFlashRecomputeGraph(target, draft)
        target_top1, draft_top1 = graph(
            torch.tensor([[3, 4, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        )
        self.assertEqual(target_top1.tolist(), [[4, 5, 1, 1]])
        self.assertEqual(draft_top1.tolist(), [[7, 8]])
        self.assertEqual(draft.block_ids.tolist(), [[4, 31, 31]])
        self.assertEqual(draft.position_ids.tolist(), [[0, 0, 0, 0, 1, 2, 3]])
        self.assertEqual(
            draft.context_attention_mask.tolist(), [[True, False, False, False]]
        )
        self.assertFalse(target.last_kwargs["use_cache"])
        self.assertTrue(target.last_kwargs["output_dflash_features"])

    def test_transformers_adapter_selects_official_post_layer_offsets(self):
        language = _AdapterLanguageModel()
        head = torch.nn.Linear(4, 20, bias=False)
        adapter = TransformersDFlashTargetAdapter(
            language,
            head,
            layer_ids=(0, 2),
            target_hidden_size=4,
            target_num_hidden_layers=3,
            vocab_size=20,
        )
        tokens = torch.tensor([[1, 2]], dtype=torch.long)
        mask = torch.ones_like(tokens)
        logits, features = adapter(
            tokens,
            mask,
            use_cache=False,
            return_dict=True,
            output_dflash_features=True,
        )
        embedding = language.embed_tokens(tokens)
        torch.testing.assert_close(features[..., :4], embedding + 1.0)
        torch.testing.assert_close(features[..., 4:], embedding + 3.0)
        self.assertEqual(tuple(logits.shape), (1, 2, 20))
        self.assertTrue(language.last_kwargs["output_hidden_states"])
        self.assertFalse(language.last_kwargs["use_cache"])

    def test_integrated_spec_metadata_cannot_replace_core_semantics(self):
        with self.assertRaisesRegex(ValueError, "reserved fields"):
            integrated_recompute_graph_spec(
                _IntegratedTarget(),
                _IntegratedDraft(),
                max_sequence_length=8,
                device="cpu",
                metadata={"block_size": 999},
            )

    def test_integrated_factory_requires_explicit_static_gear_before_loading(self):
        with self.assertRaisesRegex(ValueError, "explicit max_sequence_length"):
            create_integrated_recompute_graph({"device": "cpu"})

    def test_locked_resource_resolves_manifests_and_detects_file_tamper(self):
        workspace = self._run_root()
        model_root = workspace / "models" / "fixture"
        resource_root = workspace / "shared" / "data" / "fixture-data"
        (model_root / "specs").mkdir(parents=True)
        resource_root.mkdir(parents=True)
        config_bytes = b'{"fixture": true}\n'
        config_path = resource_root / "config.json"
        config_path.write_bytes(config_bytes)
        manifest = {
            "asset_id": "fixture-data",
            "source": {"revision": "revision-1"},
            "checkpoint": {
                "tensor_count": 1,
                "total_tensor_bytes": 4,
                "shards": 1,
            },
            "files": [
                {
                    "path": "config.json",
                    "bytes": len(config_bytes),
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                }
            ],
        }
        manifest_path = resource_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (workspace / "workspace.yaml").write_text(
            json.dumps({"shared": {"data": "shared/data"}}), encoding="utf-8"
        )
        (model_root / "project.yaml").write_text(
            json.dumps({"dependencies": {"data": ["fixture-data"]}}),
            encoding="utf-8",
        )
        (model_root / "specs" / "data.lock.json").write_text(
            json.dumps(
                {
                    "resources": [
                        {
                            "asset_id": "fixture-data",
                            "manifest_sha256": hashlib.sha256(
                                manifest_path.read_bytes()
                            ).hexdigest(),
                            "source_revision": "revision-1",
                            "tensor_count": 1,
                            "tensor_bytes": 4,
                            "shards": 1,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        resource = resolve_locked_data(
            "fixture-data", workspace_root=workspace, model_root=model_root
        )
        self.assertEqual(resource.file("config.json"), config_path)
        config_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            resource.file("config.json")

    def _recompute_backend(self, runtime, *, ordinary_only=False):
        return RecomputeDFlashOmBackend(
            runtime,
            graph_name="dflash_recompute",
            pad_token_id=0,
            device={
                "target_id": "ascend310p",
                "model": "Ascend310P3",
                "device_id": 0,
            },
            cann="test-cann",
            driver="test-driver",
            firmware="test-firmware",
            runtime_identity="test-pyacl",
            ordinary_only=ordinary_only,
        )

    def test_recompute_backend_accepts_draft_prefix_and_bonus(self):
        runtime = _RecomputeRuntime()
        backend = self._recompute_backend(runtime)
        prefill = backend.prefill([1, 2], max_new_tokens=5, eos_token_ids=())
        self.assertEqual(prefill.token_ids, (3,))
        self.assertFalse(prefill.finished)
        step = backend.decode(
            [1, 2, 3],
            max_new_tokens=4,
            max_draft_tokens=2,
            eos_token_ids=(),
        )
        self.assertEqual(step.token_ids, (4, 5, 6))
        self.assertEqual(step.drafted_tokens, 2)
        self.assertEqual(step.accepted_draft_tokens, 2)
        self.assertEqual(step.rejected_draft_tokens, 0)
        self.assertEqual(step.metadata["mode"], "draft-verify-bonus")
        self.assertEqual(len(runtime.calls), 3)

    def test_recompute_backend_marks_prefill_eos_finished(self):
        backend = self._recompute_backend(_RecomputeRuntime())
        step = backend.prefill([1, 2], max_new_tokens=5, eos_token_ids=(3,))
        self.assertEqual(step.token_ids, (3,))
        self.assertTrue(step.finished)

    def test_recompute_backend_commits_target_correction_on_rejection(self):
        runtime = _RecomputeRuntime(reject_second=True)
        backend = self._recompute_backend(runtime)
        step = backend.decode(
            [1, 2, 3],
            max_new_tokens=4,
            max_draft_tokens=2,
            eos_token_ids=(),
        )
        self.assertEqual(step.token_ids, (4, 5))
        self.assertEqual(step.accepted_draft_tokens, 1)
        self.assertEqual(step.rejected_draft_tokens, 1)
        self.assertEqual(step.metadata["mode"], "draft-verify-correction")

    def test_recompute_backend_rejects_generation_beyond_static_gear(self):
        backend = self._recompute_backend(_RecomputeRuntime())
        with self.assertRaisesRegex(ValueError, "exceeds the fixed"):
            backend.prefill(list(range(11)), max_new_tokens=3, eos_token_ids=())

    def test_recompute_backend_uses_last_input_row_to_fill_static_gear(self):
        backend = self._recompute_backend(_RecomputeRuntime())
        step = backend.prefill(list(range(12)), max_new_tokens=1, eos_token_ids=())
        self.assertEqual(step.token_ids, (12,))

    def test_recompute_backend_rejects_non_int64_abi(self):
        runtime = _RecomputeRuntime()
        original = runtime.graph_outputs

        def wrong_outputs(name):
            outputs = list(original(name))
            outputs[0] = {**outputs[0], "dtype": "int32"}
            return tuple(outputs)

        runtime.graph_outputs = wrong_outputs
        with self.assertRaisesRegex(ValueError, "target_top1 must use int64"):
            self._recompute_backend(runtime)

    def test_dflash_report_has_zero_mismatch_against_ordinary_mode(self):
        ordinary = benchmark_prompt(
            self._recompute_backend(_RecomputeRuntime(), ordinary_only=True),
            _Tokenizer(),
            "hello",
            max_new_tokens=4,
            max_draft_tokens=2,
            warmup=3,
            repetitions=10,
            require_target=True,
        )
        candidate = benchmark_prompt(
            self._recompute_backend(_RecomputeRuntime()),
            _Tokenizer(),
            "hello",
            max_new_tokens=4,
            max_draft_tokens=2,
            warmup=3,
            repetitions=10,
            require_target=True,
        )
        parity = verify_ordinary_reference(candidate, ordinary)
        self.assertEqual(parity["token_id_mismatches"], 0)
        self.assertEqual(parity["eos_mismatches"], 0)
        candidate["stable_generated_token_ids"][-1] += 1
        with self.assertRaisesRegex(RuntimeError, "differs from ordinary"):
            verify_ordinary_reference(candidate, ordinary)


if __name__ == "__main__":
    unittest.main()
