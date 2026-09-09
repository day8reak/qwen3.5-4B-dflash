"""Host contract tests only: neither these fixtures nor fake ACL implement GDR."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from tools.debug_gdr import run as debug


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / 'debug space 测试'
    args = debug.parser().parse_args(["prepare", "--work-dir", str(root)])
    debug.prepare(args)
    return root


def test_frozen_inputs_same_for_both_runtime_lengths(prepared):
    a, b = debug.arrays(prepared, 16), debug.arrays(prepared, 8)
    assert [x.dtype.name for x in a] == list(debug.DTYPES)
    assert [x.shape for x in a] == list(debug.SHAPES)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a[:6], b[:6]))
    assert a[-1].tobytes() == b'\x10\x00' and b[-1].tobytes() == b'\x08\x00'
    assert np.any(a[5] != 0)


def test_inputs_cannot_be_overwritten_or_mutated(prepared):
    args = debug.parser().parse_args(["prepare", "--work-dir", str(prepared)])
    with pytest.raises(FileExistsError): debug.prepare(args)
    path = prepared / 'inputs/length-8.bin'
    path.write_bytes(b'\x08\x00\x00\x00')
    with pytest.raises(ValueError, match='changed effective length'): debug.arrays(prepared, 8)


def test_hash_mismatch_rejected(prepared):
    (prepared / 'inputs/query.bin').write_bytes(b'X' * 131072)
    with pytest.raises(ValueError, match='changed input bytes'): debug.arrays(prepared, 16)


def test_requires_managed_output(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path / 'run'))
    with pytest.raises(RuntimeError, match='must be below'): debug.output_dir(tmp_path / 'outside', new=True)


@pytest.mark.parametrize('text', ['16,16', '0,8', '17', '-1', ''])
def test_bad_lengths(text):
    with pytest.raises((ValueError, argparse.ArgumentTypeError)): debug.length_list(text)


def test_core_tail_is_not_a_correctness_failure():
    a = np.zeros(16 * 32 * 128, dtype=np.float16)
    b = a.copy(); b[8 * 32 * 128:] = np.nan
    result = debug.compare_output(a.tobytes(), b.tobytes(), 'float16', 8 * 32 * 128 * 2)
    assert result['finite'] and result['exact_equal']
    b[0] = 2
    result = debug.compare_output(a.tobytes(), b.tobytes(), 'float16', 8 * 32 * 128 * 2)
    assert not result['exact_equal'] and result['max_abs_diff'] == 2


def test_nonfinite_valid_output_is_reported():
    a = np.zeros(4, dtype=np.float32); b = a.copy(); b[1] = np.nan
    result = debug.compare_output(a.tobytes(), b.tobytes(), 'float32', 16)
    assert not result['finite'] and result['max_abs_diff'] is None
    json.dumps(result, allow_nan=False)


def test_npz_validation(prepared, tmp_path):
    data = dict(zip(debug.NAMES, debug.arrays(prepared, 16)))
    data['effective_length'] = data['effective_length'].astype(np.int32)
    with pytest.raises(ValueError, match='effective_length'): debug.validate_arrays(data)
    data['effective_length'] = np.array([16], dtype=np.int16)
    source = tmp_path / 'captured.npz'; np.savez(source, **data)
    root = tmp_path / 'replayed'
    debug.prepare(debug.parser().parse_args(['prepare', '--work-dir', str(root), '--inputs', str(source)]))
    assert debug.arrays(root, 8)[0].tobytes() == data['query'].tobytes()
    assert debug.load_manifest(root)['source']['kind'] == 'supplied_npz'


@pytest.mark.parametrize('variant', debug.VARIANTS)
def test_wrapper_keeps_operator_attributes_and_public_length(variant):
    import torch
    calls = []
    def operation(q, k, v, **kw):
        calls.append((q, k, v, kw))
        return v.reshape(-1, 128), kw['initial_state']
    m = debug.gdr_module(torch, operation, variant)
    inputs = tuple(torch.zeros(s, dtype=getattr(torch, d)) for s, d in zip(debug.SHAPES, debug.DTYPES))
    outputs = m(*inputs)
    call = calls[0][3]
    assert {k: call[k] for k in debug.ATTRS} == debug.ATTRS
    assert call['effective_length'] is inputs[-1] and call['initial_state'] is inputs[-2]
    assert [x.dtype for x in outputs] == [getattr(torch, x[1]) for x in debug.output_specs(variant, 16)]
    assert len(outputs) == (2 if variant == 'both' else 1)


def test_exported_graphs_keep_one_gdr_and_all_seven_inputs():
    import torch
    lib = torch.library.Library('gdr_debug_contract_test', 'DEF')
    lib.define('gdr(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, Tensor effective_length, int chunk_size, Tensor initial_state, bool output_final_state, bool use_qk_l2norm_in_kernel) -> (Tensor, Tensor)')
    def fake(query, key, value, g, beta, effective_length, chunk_size, initial_state, output_final_state, use_qk_l2norm_in_kernel):
        return torch.empty_like(value), torch.empty_like(initial_state)
    lib.impl('gdr', fake, 'Meta')
    operation = torch.ops.gdr_debug_contract_test.gdr.default
    inputs = tuple(torch.zeros(s, dtype=getattr(torch, d)) for s, d in zip(debug.SHAPES, debug.DTYPES))
    for variant in debug.VARIANTS:
        exported = torch.export.export(debug.gdr_module(torch, operation, variant), inputs, strict=True)
        nodes = list(exported.graph.nodes)
        assert len([n for n in nodes if n.op == 'placeholder']) == 7
        ops = [n for n in nodes if n.op == 'call_function' and n.target is operation]
        assert len(ops) == 1
        node = ops[0]
        # torch.export canonicalizes dispatcher arguments to positional form.
        assert node.args[6] == 64 and node.args[8:] == (True, True)
        assert node.args[5].name == 'effective_length'
        assert node.args[7].name == 'initial_state'
        assert len(next(n for n in nodes if n.op == 'output').args[0]) == (2 if variant == 'both' else 1)


@pytest.fixture(scope='session')
def fake_runner(tmp_path_factory):
    build = tmp_path_factory.mktemp('gdr-fake-build')
    binary = build / 'gdr_debug_runner_fake'
    subprocess.run(['g++', '-std=c++17', '-O2', '-DGDR_DEBUG_FAKE_ACL=1', '-Wall', '-Wextra', '-Wpedantic',
        '-I', str(REPO / 'framework/runtime/cpp/tests/fake_acl'),
        '-I', str(REPO / 'framework/runtime/cpp/include'), str(HERE / 'runner.cpp'),
        str(HERE / 'fake_acl_test.cpp'), str(REPO / 'framework/runtime/cpp/src/sha256.cpp'),
        '-o', str(binary)], check=True, capture_output=True, text=True)
    return binary


def fixture_models(prepared):
    models = {}
    for variant in debug.VARIANTS:
        file = prepared / f'gdr_{variant}.om'; file.write_bytes(b'HOST_FIXTURE_NOT_OM')
        models[variant] = {'path': file.name, 'sha256': debug.digest(file)}
    debug.write_json(prepared / 'om.json', {'graphs': models})


def fake_command(prepared, runner, variant='both', length=8, profile=None):
    if not (prepared / 'om.json').exists(): fixture_models(prepared)
    args = argparse.Namespace(device_id=0, warmup=3, repetitions=3, profile_output=profile, metrics='PipeUtilization')
    out = prepared / ('result-' + variant + ('-profile' if profile else ''))
    plan = debug.make_plan(prepared, out, variant, length, args)
    env = dict(os.environ); env.pop('ASCEND310P_SIMULATION_ONLY', None)
    return [str(runner), str(plan)], env, out


@pytest.mark.parametrize('variant', debug.VARIANTS)
def test_cpp_io_roundtrip_padding_and_stability(prepared, fake_runner, variant):
    command, env, out = fake_command(prepared, fake_runner, variant)
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['cpu_fallback'] is True
    assert report['stable'] and len(report['samples']) == 3
    scalar = next(t for t in report['tensors'] if t['name'] == 'effective_length')
    assert scalar['logical_bytes'] == 2 and scalar['allocated_bytes'] == 32
    for name, dtype, _, valid in debug.output_specs(variant, 8):
        source = prepared / 'inputs' / ('query.bin' if name == 'core_attn' else 'initial_state.bin')
        assert (out / (name + '.bin')).read_bytes()[:valid] == source.read_bytes()[:valid]
        assert len({r['valid_output_sha256'][name] for r in report['samples']}) == 1


@pytest.mark.parametrize('flag, message', [('GDR_TEST_BAD_DTYPE', 'OM dtype/size mismatch'),
                                           ('GDR_TEST_MUTATE', 'modified declared read-only input')])
def test_cpp_detects_contract_violations(prepared, fake_runner, flag, message):
    command, env, _ = fake_command(prepared, fake_runner)
    env[flag] = '1'
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 1 and message in proc.stderr


def test_cpp_detects_unstable_valid_output(prepared, fake_runner):
    command, env, out = fake_command(prepared, fake_runner)
    env['GDR_TEST_UNSTABLE'] = '1'
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['stable'] is False
    assert [x['stable'] for x in report['samples']] == [True, False, False]


def test_cpp_profiles_exactly_one_call_after_warmup(prepared, fake_runner):
    parent, child = socket.socketpair()
    parent.settimeout(10)
    command, env, out = fake_command(prepared, fake_runner, profile=str(prepared / 'capture'))
    env.update(PROFILING_MODE='dynamic', DFLASH_MSPROF_CONTROL_FD=str(child.fileno()), DFLASH_MSPROF_CONTROL_TIMEOUT='10')
    proc = subprocess.Popen(command, env=env, pass_fds=(child.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child.close()
    try:
        reader = parent.makefile('rb')
        ready = json.loads(reader.readline())
        assert ready['event'] == 'ready' and ready['pid'] == proc.pid
        assert ready['stage'] == 'verify'
        parent.sendall(b'{"event": "started"}\n')
        done = json.loads(reader.readline()); assert done == {'event': 'done', 'success': True}
        parent.sendall(b'{"event": "stopped"}\n')
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, stderr
        report = json.loads((out / 'report.json').read_text())
        assert report['warmup'] == 3 and len(report['samples']) == 1 and report['profiled']
        reader.close()
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
        parent.close()


def test_profiler_rows_preserve_metrics_and_do_not_double_count_exports(tmp_path):
    for name in ('one', 'two'):
        root = tmp_path / name; root.mkdir()
        (root / 'op_summary_0.csv').write_text('OP Type,Op Name,Task Duration(us),mte3_ratio,Tiling Key\nChunkGatedDeltaRule,GDR,28470.94,0.996,7\nReshape,R,2,0,0\n')
    output = tmp_path / 'report.json'
    debug.extract_gdr_rows(tmp_path, output)
    report = json.loads(output.read_text())
    assert len(report['exports']) == 2
    assert report['exports'][0]['raw_row']['Tiling Key'] == '7'
    assert report['exports'][0]['gdr_duration_us'] == 28470.94
    (tmp_path / 'two/op_summary_0.csv').write_text('OP Type,Task Duration(us)\nReshape,2\n')
    with pytest.raises(ValueError, match='exactly one GDR'): debug.extract_gdr_rows(tmp_path, output)


def test_native_benchmark_restores_inputs_and_records_each_call(prepared, monkeypatch):
    import torch
    from types import SimpleNamespace
    original_to = torch.Tensor.to
    def host_to(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('npu:'):
            args = ('cpu', *args[1:])
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', host_to)
    calls = []
    def fake(q, k, v, **kw):
        calls.append(kw['initial_state'].clone())
        return q.clone(), kw['initial_state'].clone()
    npu = SimpleNamespace(npu_chunk_gated_delta_rule=fake, __version__='HOST_TEST_ONLY')
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda _: None,
                         get_device_name=lambda _: 'HOST_TEST_ONLY'), raising=False)
    monkeypatch.setattr(debug, 'require_device', lambda _: (torch, npu, SimpleNamespace(_schema='test-only')))
    monkeypatch.setattr(debug, 'package_identity', lambda: {'scope': 'host test'})
    out = prepared / 'native-host-fixture'
    args = debug.parser().parse_args(['_native', '--work-dir', str(prepared), '--result-dir', str(out),
                                     '--length', '8', '--warmup', '3', '--repetitions', '10'])
    debug.native(args)
    report = json.loads((out / 'report.json').read_text())
    assert len(calls) == 13 and len(report['samples']) == 10 and report['stable']
    assert all(torch.equal(calls[0], other) for other in calls)
    assert all(len(x['valid_output_sha256']['core_attn']) == 64 for x in report['samples'])


def test_summary_compares_valid_outputs_and_keeps_native_and_om_separate(prepared, fake_runner):
    fixture_models(prepared)
    directory = prepared / 'measurements'; directory.mkdir()
    env = dict(os.environ); env.pop('ASCEND310P_SIMULATION_ONLY', None)
    for length in (16, 8):
        ref = directory / f'native-both-L{length}'; ref.mkdir()
        (ref / 'core_attn.bin').write_bytes((prepared / 'inputs/query.bin').read_bytes())
        (ref / 'last_recurrent_state.bin').write_bytes((prepared / 'inputs/initial_state.bin').read_bytes())
        debug.write_json(ref / 'report.json', {'samples': [{'elapsed_ms': 100}], 'stable': True})
        for variant in debug.VARIANTS:
            out = directory / f'om-{variant}-L{length}'
            args = argparse.Namespace(device_id=0, warmup=1, repetitions=2, profile_output=None, metrics='Memory')
            plan = debug.make_plan(prepared, out, variant, length, args)
            proc = subprocess.run([str(fake_runner), str(plan)], env=env, capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr
    debug.summarize(argparse.Namespace(work_dir=str(prepared)))
    result = json.loads((prepared / 'summary.json').read_text())
    assert len(result['rows']) == 8
    assert all(all(c['exact_equal'] for c in row['comparison_to_native'].values()) for row in result['rows'])
    assert result['rows'][0]['median_ms'] == 100
    assert {row['measurements'] for row in result['rows']} == {1, 2}


def test_profile_runs_export_before_csv_parsing(prepared, fake_runner, monkeypatch):
    fixture_models(prepared)
    calls = []
    def execute(cmd, log, **_):
        calls.append(cmd)
        log = Path(log)
        if '--control-report' in cmd:
            report = Path(cmd[cmd.index('--control-report')+1])
            debug.write_json(report, {'status': 'PASS_CONTROL', **{k: True for k in (
                'start_acknowledged', 'stop_acknowledged', 'quit_acknowledged', 'capture_completed')}})
            out = report.parent / 'result'; out.mkdir()
            debug.write_json(out / 'report.json', {'samples': [{'elapsed_ms': 1}], 'profiled': True})
        else:
            assert '--export=on' in cmd and '--summary-format=csv' in cmd
            output = Path(next(v.split('=', 1)[1] for v in cmd if v.startswith('--output=')))
            output.mkdir()
            (output / 'op_summary.csv').write_text('OP Type,Op Name,Task Duration(us),mte3_ratio\nChunkGatedDeltaRule,GDR,28000,0.996\n')
        log.write_text('host test\n')
    monkeypatch.setattr(debug, 'execute', execute)
    args = debug.parser().parse_args(['profile', '--work-dir', str(prepared), '--backend', 'om', '--length', '16'])
    debug.profile(args)
    assert len(calls) == 2
    report = next((prepared / 'profiles').glob('*/gdr-rows.json'))
    assert json.loads(report.read_text())['exports'][0]['gdr_duration_us'] == 28000


def test_capture_preserves_api_and_stops_after_selected_call(prepared, monkeypatch):
    import torch
    from types import SimpleNamespace
    names = ('query', 'key', 'value', 'g', 'beta', 'effective_length', 'chunk_size',
             'initial_state', 'output_final_state', 'use_qk_l2norm_in_kernel')
    schema = SimpleNamespace(arguments=[SimpleNamespace(name=n, default_value=None) for n in names])
    invoked = []
    def original(*_, **__): invoked.append(True)
    fake_npu = SimpleNamespace(npu_chunk_gated_delta_rule=original)
    monkeypatch.setattr(debug, 'require_device', lambda _: (torch, fake_npu, SimpleNamespace(_schema=schema)))
    def module(*_, **__):
        vals = [torch.from_numpy(x.copy()) for x in debug.arrays(prepared, 16)]
        for _ in range(3):
            fake_npu.npu_chunk_gated_delta_rule(*vals[:5], effective_length=vals[6], initial_state=vals[5], **debug.ATTRS)
    monkeypatch.setattr(debug.runpy, 'run_module', module)
    root = prepared.parent / 'capture'
    args = debug.parser().parse_args(['capture', '--work-dir', str(root), '--capture-index', '1', '--stop-after-capture'])
    debug.capture(args)
    assert len(invoked) == 1
    assert fake_npu.npu_chunk_gated_delta_rule is original
    assert json.loads((root / 'capture.json').read_text())['matching_call_index'] == 1
    with np.load(root / 'inputs.npz') as archive:
        assert debug.validate_arrays(dict(archive))['effective_length'][0] == 16
