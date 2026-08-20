"""Fail-closed import probes for Qwen3.5 DFlash receiver overlays.

The checker loads no model weights.  ``target`` validates the two-file target
feature overlay against a receiver-owned configuration module.  ``v1-cli``
validates the complete V1 runtime copy closure and executes the CLI help entry
point from the receiver's real package namespace, including internal layouts
such as ``transformer.model.qwen3_5``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPECTED_TRANSFORMERS_VERSION = "5.14.1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CONTRACT = PACKAGE_ROOT / "TARGET_OVERLAY.json"
FULL_CONTRACT = PACKAGE_ROOT / "TARGET_OVERLAY_FULL.json"
sys.dont_write_bytecode = True


class OverlayValidationError(RuntimeError):
    """The receiver overlay is incomplete or resolves from the wrong path."""


def _relative_modules(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise OverlayValidationError(f"cannot parse {path}: {error}") from error
    unsupported = sorted(
        {
            f"level={node.level}, module={node.module!r}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level not in (0, 1)
        }
    )
    if unsupported:
        raise OverlayValidationError(
            f"unsupported receiver-relative imports in {path.name}: "
            + ", ".join(unsupported)
        )
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is not None
    }
    return tuple(sorted(modules))


def _python_root(package_dir: Path, package_name: str) -> Path:
    parts = package_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise OverlayValidationError(f"invalid Python package name: {package_name!r}")
    cursor = package_dir
    for expected in reversed(parts):
        if cursor.name != expected:
            raise OverlayValidationError(
                f"package directory {package_dir} does not end with "
                f"{package_name.replace('.', '/')}"
            )
        cursor = cursor.parent
    return cursor


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OverlayValidationError(f"cannot read overlay contract {path}: {error}") from error
    if not isinstance(payload, dict):
        raise OverlayValidationError(f"overlay contract must be a JSON object: {path}")
    return payload


def _copy_entries(contract: dict[str, Any]) -> tuple[str, ...]:
    raw_entries = contract.get("required_copy_files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise OverlayValidationError("overlay contract has no required_copy_files")
    entries: list[str] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, str):
            raise OverlayValidationError("required_copy_files must contain strings")
        path = Path(raw_entry)
        if path.parts != ("models", "dflash_v1", path.name) or path.suffix != ".py":
            raise OverlayValidationError(
                "overlay copy entry must be "
                f"models/dflash_v1/<module>.py: {raw_entry!r}"
            )
        entries.append(raw_entry)
    if len(entries) != len(set(entries)):
        raise OverlayValidationError("overlay contract contains duplicate copy entries")
    return tuple(entries)


def _receiver_owned_entries(contract: dict[str, Any]) -> tuple[str, ...]:
    raw_entry = contract.get("receiver_owned_dependency")
    if not isinstance(raw_entry, str):
        raise OverlayValidationError(
            "overlay contract has no receiver_owned_dependency"
        )
    path = Path(raw_entry)
    if path.parts != (path.name,) or path.suffix != ".py":
        raise OverlayValidationError(
            "receiver_owned_dependency must be a package-local Python filename"
        )
    return (raw_entry,)


def _receiver_loader_policy(contract: dict[str, Any]) -> dict[str, str]:
    raw = contract.get("receiver_owned_runtime_loader")
    expected_keys = {
        "file",
        "module",
        "required_callable",
        "required_facade_class",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise OverlayValidationError("full overlay has invalid receiver loader policy")
    if not all(isinstance(value, str) and value for value in raw.values()):
        raise OverlayValidationError("receiver loader policy values must be strings")
    if Path(raw["file"]).parts != (raw["file"],) or not raw["file"].endswith(".py"):
        raise OverlayValidationError("receiver loader must be a package-local .py file")
    if Path(raw["file"]).stem != raw["module"]:
        raise OverlayValidationError("receiver loader file/module mismatch")
    return dict(raw)


def _module_suffixes(contract: dict[str, Any], field: str) -> tuple[str, ...]:
    raw_modules = contract.get(field)
    if not isinstance(raw_modules, list) or not raw_modules:
        raise OverlayValidationError(f"overlay contract has invalid {field}")
    if not all(isinstance(value, str) and value.isidentifier() for value in raw_modules):
        raise OverlayValidationError(f"overlay contract has invalid {field}")
    modules = tuple(raw_modules)
    if len(modules) != len(set(modules)):
        raise OverlayValidationError(f"overlay contract has duplicate {field}")
    return modules


def _assert_full_contract_ownership(
    copy_entries: tuple[str, ...],
    receiver_owned: tuple[str, ...],
    import_suffixes: tuple[str, ...],
    receiver_import_suffixes: tuple[str, ...],
) -> None:
    copied_names = {Path(entry).name for entry in copy_entries}
    receiver_names = set(receiver_owned)
    overlap = copied_names & receiver_names
    if overlap:
        raise OverlayValidationError(
            "receiver-owned files must not be copied or hashed: "
            + ", ".join(sorted(overlap))
        )

    copied_modules = {Path(entry).stem for entry in copy_entries}
    imported_modules = set(import_suffixes)
    if copied_modules != imported_modules:
        missing_imports = sorted(copied_modules - imported_modules)
        unowned_imports = sorted(imported_modules - copied_modules)
        details: list[str] = []
        if missing_imports:
            details.append("not imported=" + ",".join(missing_imports))
        if unowned_imports:
            details.append("not copied=" + ",".join(unowned_imports))
        raise OverlayValidationError(
            "full overlay must import every delivered runtime module exactly once: "
            + "; ".join(details)
        )

    expected_receiver_modules = {Path(name).stem for name in receiver_owned}
    if set(receiver_import_suffixes) != expected_receiver_modules:
        raise OverlayValidationError(
            "receiver_import_modules must exactly match receiver-owned files"
        )


def _required_symbols(contract: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw_mapping = contract.get("receiver_required_symbols")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise OverlayValidationError(
            "overlay contract has invalid receiver_required_symbols"
        )
    result: dict[str, tuple[str, ...]] = {}
    for module_name, raw_symbols in raw_mapping.items():
        if not isinstance(module_name, str) or not module_name.isidentifier():
            raise OverlayValidationError(
                "receiver_required_symbols has an invalid module name"
            )
        if not isinstance(raw_symbols, list) or not raw_symbols or not all(
            isinstance(symbol, str) and symbol.isidentifier()
            for symbol in raw_symbols
        ):
            raise OverlayValidationError(
                f"receiver_required_symbols[{module_name!r}] is invalid"
            )
        symbols = tuple(raw_symbols)
        if len(symbols) != len(set(symbols)):
            raise OverlayValidationError(
                f"receiver_required_symbols[{module_name!r}] has duplicates"
            )
        result[module_name] = symbols
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loader_contract_tree(path: Path) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise OverlayValidationError(f"cannot parse receiver loader: {error}") from error
    factories = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "create_internal_target"
    ]
    if len(factories) != 1:
        raise OverlayValidationError(
            "receiver loader must contain exactly one create_internal_target function"
        )
    protected = {"InternalTargetFacade", "load_target"}
    referenced = sorted(
        {
            node.id
            for node in ast.walk(factories[0])
            if isinstance(node, ast.Name) and node.id in protected
        }
    )
    if referenced:
        raise OverlayValidationError(
            "create_internal_target must not reference protected facade/loader "
            "symbols: " + ", ".join(referenced)
        )
    return tree, factories[0]


def _require_implemented_receiver_factory(path: Path) -> None:
    """Reject the delivered placeholder while leaving execution to the receiver."""

    _tree, factory = _loader_contract_tree(path)
    raises_not_implemented = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "NotImplementedError"
        for node in ast.walk(factory)
    )
    has_value_return = any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(factory)
    )
    if raises_not_implemented or not has_value_return:
        raise OverlayValidationError(
            "receiver create_internal_target() is still the delivered "
            "placeholder; implement the receiver factory before formal-ready "
            "v1-cli validation"
        )


def _loader_contract_ast(path: Path) -> str:
    tree, factory = _loader_contract_tree(path)
    factory.body = [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _validate_formal_hiai_source(
    source: Path,
    package_dir: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    """Run the delivered semantic checker against the receiver-owned source."""

    raw_policy = contract.get("formal_hiai_source_patch")
    if not isinstance(raw_policy, dict) or raw_policy.get("required") is not True:
        raise OverlayValidationError("full overlay lacks a required HIAI source policy")
    expected_basename = raw_policy.get("source_basename")
    expected_contract = raw_policy.get("patch_contract_id")
    expected_sidecars = raw_policy.get("required_relative_sidecars")
    if (
        not isinstance(expected_basename, str)
        or not isinstance(expected_contract, str)
        or not isinstance(expected_sidecars, list)
        or not all(isinstance(item, str) for item in expected_sidecars)
    ):
        raise OverlayValidationError("formal HIAI source policy is malformed")
    source = source.expanduser()
    if source.is_symlink():
        raise OverlayValidationError("formal HIAI source must not be a symlink")
    source = source.resolve()
    expected_source = (package_dir / expected_basename).resolve()
    if source != expected_source:
        raise OverlayValidationError(
            "formal HIAI source must be the receiver package-local "
            f"{expected_basename}"
        )
    if source.name != expected_basename or not source.is_file():
        raise OverlayValidationError(
            f"formal HIAI source must be a regular {expected_basename} file"
        )
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as error:
        raise OverlayValidationError(f"cannot parse formal HIAI source: {error}") from error
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and isinstance(node.module, str)
    }
    missing_sidecars = sorted(set(expected_sidecars) - relative_modules)
    if missing_sidecars:
        raise OverlayValidationError(
            "formal HIAI source lacks required relative sidecars: "
            + ", ".join(missing_sidecars)
        )
    patcher = package_dir / "dflash_hiai_feature_patch.py"
    completed = subprocess.run(
        [sys.executable, str(patcher), "--source", str(source), "--check"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OverlayValidationError(f"formal HIAI source check failed: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OverlayValidationError("formal HIAI source checker returned invalid JSON") from error
    if (
        report.get("status") != "verified"
        or report.get("patch_contract_id") != expected_contract
        or report.get("source_sha256") != _sha256(source)
    ):
        raise OverlayValidationError("formal HIAI source identity does not match its contract")
    return {
        "status": "PASS",
        "path": str(source),
        "source_sha256": report["source_sha256"],
        "patch_contract_id": report["patch_contract_id"],
        "feature_source": report["feature_source"],
        "capture_point": report["capture_point"],
        "required_relative_sidecars": sorted(expected_sidecars),
        "custom_operators_modified": report["custom_operators_modified"],
        "weights_loaded": False,
    }


def _missing_files(
    package_dir: Path,
    copy_entries: tuple[str, ...],
    receiver_owned: tuple[str, ...],
    receiver_runtime: tuple[str, ...] = (),
) -> tuple[str, ...]:
    names = [Path(entry).name for entry in copy_entries]
    names.extend(receiver_owned)
    names.extend(receiver_runtime)
    return tuple(sorted(name for name in names if not (package_dir / name).is_file()))


def _reject_package_bytecode(package_dir: Path) -> None:
    if os.environ.get("PYTHONPYCACHEPREFIX") or sys.pycache_prefix is not None:
        raise OverlayValidationError(
            "PYTHONPYCACHEPREFIX/sys.pycache_prefix is forbidden for formal "
            "overlay validation"
        )
    bytecode = sorted(package_dir.glob("*.pyc"))
    cache_dir = package_dir / "__pycache__"
    if cache_dir.exists():
        bytecode.extend(sorted(cache_dir.glob("*.pyc")))
    if bytecode:
        raise OverlayValidationError(
            "receiver package must not contain precompiled bytecode during "
            "overlay validation: "
            + ", ".join(path.name for path in bytecode)
        )


def _assert_exact_copies(
    package_dir: Path,
    source_models_dir: Path,
    copy_entries: tuple[str, ...],
) -> dict[str, str]:
    receiver_symlinks = sorted(
        Path(entry).name
        for entry in copy_entries
        if (package_dir / Path(entry).name).is_symlink()
    )
    if receiver_symlinks:
        raise OverlayValidationError(
            "receiver runtime files must be real copies, not symlinks: "
            + ", ".join(receiver_symlinks)
        )
    source_symlinks = sorted(
        Path(entry).name
        for entry in copy_entries
        if (source_models_dir / Path(entry).name).is_symlink()
    )
    if source_symlinks:
        raise OverlayValidationError(
            "packaged source files must not be symlinks: "
            + ", ".join(source_symlinks)
        )
    missing_sources = sorted(
        Path(entry).name
        for entry in copy_entries
        if not (source_models_dir / Path(entry).name).is_file()
    )
    if missing_sources:
        raise OverlayValidationError(
            "package source files are missing: " + ", ".join(missing_sources)
        )

    mismatches: list[str] = []
    hashes: dict[str, str] = {}
    for entry in copy_entries:
        name = Path(entry).name
        source = source_models_dir / name
        receiver = package_dir / name
        source_hash = _sha256(source)
        receiver_hash = _sha256(receiver)
        hashes[name] = receiver_hash
        if receiver_hash != source_hash:
            mismatches.append(name)
    if mismatches:
        raise OverlayValidationError(
            "receiver files differ from the packaged sources: "
            + ", ".join(sorted(mismatches))
        )
    return dict(sorted(hashes.items()))


def _dependency_graph(
    package_dir: Path,
    module_files: tuple[str, ...],
) -> dict[str, list[str]]:
    return {
        Path(name).stem: list(_relative_modules(package_dir / name))
        for name in sorted(module_files)
    }


def _assert_dependency_closure(
    graph: dict[str, list[str]],
    available_files: set[str],
) -> None:
    missing: set[str] = set()
    for importer, dependencies in graph.items():
        for dependency in dependencies:
            dependency_file = f"{dependency}.py"
            if dependency_file not in available_files:
                missing.add(f"{dependency_file} (imported by {importer}.py)")
    if missing:
        raise OverlayValidationError(
            "receiver-local dependency closure is incomplete: "
            + ", ".join(sorted(missing))
        )


def _assert_frozen_graph(
    graph: dict[str, list[str]], contract: dict[str, Any]
) -> None:
    expected = contract.get("receiver_local_dependency_closure")
    if expected is None:
        return
    if graph != expected:
        raise OverlayValidationError(
            "receiver-local dependency graph differs from TARGET_OVERLAY_FULL.json: "
            f"expected {expected}, got {graph}"
        )


def _generic_models_modules() -> list[str]:
    return sorted(
        name for name in sys.modules if name == "models" or name.startswith("models.")
    )


def _import_receiver_modules(
    package_dir: Path,
    package_name: str,
    root: Path,
    module_suffixes: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    preexisting = _generic_models_modules()
    if preexisting:
        raise OverlayValidationError(
            "generic models namespace was imported before the receiver probe: "
            + ", ".join(preexisting)
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    identities: dict[str, dict[str, str]] = {}
    for suffix in module_suffixes:
        module_name = f"{package_name}.{suffix}"
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise OverlayValidationError(
                f"failed to import {module_name}: {type(error).__name__}: {error}"
            ) from error
        package = getattr(module, "__package__", None)
        if package != package_name:
            raise OverlayValidationError(
                f"{module_name} has wrong __package__: {package!r}"
            )
        try:
            loaded_path = Path(inspect.getfile(module)).resolve()
        except (TypeError, OSError) as error:
            raise OverlayValidationError(
                f"cannot resolve __file__ for {module_name}: {error}"
            ) from error
        expected_path = (package_dir / f"{suffix}.py").resolve()
        if loaded_path != expected_path:
            raise OverlayValidationError(
                f"{module_name} resolved the wrong file: {loaded_path}; "
                f"expected {expected_path}"
            )
        identities[module_name] = {
            "package": package,
            "file": str(loaded_path),
        }

    leaked = _generic_models_modules()
    if leaked:
        raise OverlayValidationError(
            "receiver import leaked through the generic models namespace: "
            + ", ".join(leaked)
        )
    return dict(sorted(identities.items()))


def _assert_receiver_symbols(
    package_name: str,
    required_symbols: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    verified: dict[str, list[str]] = {}
    for suffix, symbols in sorted(required_symbols.items()):
        module_name = f"{package_name}.{suffix}"
        module = sys.modules.get(module_name)
        if module is None:
            raise OverlayValidationError(
                f"receiver module was not imported before symbol validation: {module_name}"
            )
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            raise OverlayValidationError(
                f"receiver module {module_name} lacks required symbols: "
                + ", ".join(missing)
            )
        verified[suffix] = list(symbols)
    return verified


def _pythonpath_entries() -> list[str]:
    raw = os.environ.get("PYTHONPATH", "")
    return [entry for entry in raw.split(os.pathsep) if entry]


def _run_v1_cli_help(root: Path, package_name: str) -> dict[str, Any]:
    module_name = f"{package_name}.dflash_qwen_adapter_v1"
    environment = os.environ.copy()
    entries = [str(root), *_pythonpath_entries()]
    deduplicated = list(dict.fromkeys(entries))
    environment["PYTHONPATH"] = os.pathsep.join(deduplicated)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OverlayValidationError(
            f"{sys.executable} -m {module_name} --help failed with "
            f"exit {completed.returncode}: {detail}"
        )
    if "models.modeling_qwen3_5_dflash" in completed.stdout:
        raise OverlayValidationError(
            "V1 CLI help still advertises the generic models namespace"
        )
    return {
        "status": "PASS",
        "module": module_name,
        "returncode": completed.returncode,
        "weights_loaded": False,
    }


def validate_overlay(
    package_dir: Path,
    package_name: str,
    *,
    scope: str = "target",
    source_models_dir: Path | None = None,
    hiai_source: Path | None = None,
) -> dict[str, object]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise OverlayValidationError(f"receiver package directory is missing: {package_dir}")
    trusted_source_models_dir = (PACKAGE_ROOT / "models" / "dflash_v1").resolve()
    if (
        source_models_dir is not None
        and source_models_dir.resolve() != trusted_source_models_dir
    ):
        raise OverlayValidationError(
            "--source-models-dir must be the models/dflash_v1 directory "
            "beside this delivered validator; arbitrary comparison roots "
            "are forbidden"
        )
    source_models_dir = trusted_source_models_dir
    contract_path = MINIMAL_CONTRACT if scope == "target" else FULL_CONTRACT
    contract = _read_contract(contract_path)
    copy_entries = _copy_entries(contract)
    receiver_owned = _receiver_owned_entries(contract)
    if scope == "target":
        receiver_import_suffixes = ("configuration_qwen3_5",)
        import_suffixes = (
            "dflash_target_features",
            "modeling_qwen3_5_dflash",
        )
        required_symbols = {
            "configuration_qwen3_5": (
                "Qwen3_5Config",
                "Qwen3_5TextConfig",
                "Qwen3_5VisionConfig",
            )
        }
    else:
        receiver_loader_policy = _receiver_loader_policy(contract)
        receiver_import_suffixes = _module_suffixes(
            contract, "receiver_import_modules"
        )
        import_suffixes = _module_suffixes(contract, "required_import_modules")
        required_symbols = _required_symbols(contract)
        _assert_full_contract_ownership(
            copy_entries,
            receiver_owned,
            import_suffixes,
            receiver_import_suffixes,
        )
        if set(required_symbols) != set(receiver_import_suffixes):
            raise OverlayValidationError(
                "receiver_required_symbols must exactly cover receiver_import_modules"
            )
    receiver_runtime = (
        ()
        if scope == "target"
        else (receiver_loader_policy["file"],)
    )
    missing = _missing_files(
        package_dir,
        copy_entries,
        receiver_owned,
        receiver_runtime,
    )
    if missing:
        raise OverlayValidationError(
            f"missing required files for {scope}: " + ", ".join(missing)
        )
    _reject_package_bytecode(package_dir)

    hashes = _assert_exact_copies(package_dir, source_models_dir, copy_entries)
    # A minimal target overlay must not impose source-style rules on the
    # receiver-owned configuration module.  Internal projects may legitimately
    # use multi-level relative imports there.  We parse and freeze only files
    # delivered by this package, then validate the receiver configuration by a
    # real import and symbol check below.
    module_files = tuple(Path(entry).name for entry in copy_entries)
    graph = _dependency_graph(package_dir, module_files)
    _assert_dependency_closure(graph, set(module_files) | set(receiver_owned))
    if scope == "v1-cli":
        _assert_frozen_graph(graph, contract)
        if hiai_source is None:
            raise OverlayValidationError(
                "v1-cli formal overlay validation requires --hiai-source"
            )
        hiai_source_report = _validate_formal_hiai_source(
            hiai_source,
            package_dir,
            contract,
        )

    import transformers

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise OverlayValidationError(
            "Transformers version mismatch: expected "
            f"{EXPECTED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )

    root = _python_root(package_dir, package_name)
    # Import the receiver-owned configuration first and validate its public
    # contract before importing any delivered module.  It is intentionally not
    # hashed or AST-parsed: internal packages may use multi-level relative
    # imports that are valid only in their singular receiver namespace.
    receiver_identities = _import_receiver_modules(
        package_dir,
        package_name,
        root,
        receiver_import_suffixes,
    )
    verified_receiver_symbols = _assert_receiver_symbols(
        package_name, required_symbols
    )
    delivered_identities = _import_receiver_modules(
        package_dir,
        package_name,
        root,
        import_suffixes,
    )
    identities = dict(sorted({**receiver_identities, **delivered_identities}.items()))

    if scope == "v1-cli":
        loader_path = package_dir / receiver_loader_policy["file"]
        if loader_path.is_symlink() or not loader_path.is_file():
            raise OverlayValidationError(
                "receiver-owned internal_target_loader.py must be a real package-local file"
            )
        loader_template = package_dir / "internal_target_loader_template.py"
        if _loader_contract_ast(loader_path) != _loader_contract_ast(loader_template):
            raise OverlayValidationError(
                "receiver loader differs from the delivered template outside "
                "create_internal_target() body"
            )
        _require_implemented_receiver_factory(loader_path)
        loader_identities = _import_receiver_modules(
            package_dir,
            package_name,
            root,
            (receiver_loader_policy["module"],),
        )
        loader_module = sys.modules[
            f"{package_name}.{receiver_loader_policy['module']}"
        ]
        loader_callable = getattr(
            loader_module,
            receiver_loader_policy["required_callable"],
            None,
        )
        facade_class = getattr(
            loader_module,
            receiver_loader_policy["required_facade_class"],
            None,
        )
        from torch import nn

        if not callable(loader_callable):
            raise OverlayValidationError("receiver loader lacks callable load_target")
        if not isinstance(facade_class, type) or not issubclass(facade_class, nn.Module):
            raise OverlayValidationError(
                "receiver loader lacks an nn.Module InternalTargetFacade class"
            )
        identities.update(loader_identities)
        receiver_loader_report = {
            "file": receiver_loader_policy["file"],
            "module": f"{package_name}.{receiver_loader_policy['module']}",
            "path": str(loader_path.resolve()),
            "sha256": _sha256(loader_path),
            "load_target_callable": True,
            "facade_class": (
                f"{facade_class.__module__}.{facade_class.__name__}"
            ),
            "evidence_authority": "receiver_owned_import_and_static_identity",
            "factory_behavior": "PENDING_DEVICE_EXECUTION",
            "factory_static_status": "PASS_NON_PLACEHOLDER",
            "template_ast_match_except_factory_body": True,
        }

    modeling = sys.modules[f"{package_name}.modeling_qwen3_5_dflash"]
    if not hasattr(modeling, "Qwen3_5ForConditionalGeneration"):
        raise OverlayValidationError(
            "patched modeling lacks Qwen3_5ForConditionalGeneration"
        )

    result: dict[str, object] = {
        "status": "PASS",
        "scope": scope,
        "package": package_name,
        "package_dir": str(package_dir),
        "transformers_version": transformers.__version__,
        "required_copy_files": list(copy_entries),
        "copied_file_sha256": hashes,
        "trusted_source_models_dir": str(source_models_dir),
        "receiver_owned_files": list(receiver_owned),
        "receiver_required_symbols": verified_receiver_symbols,
        "receiver_local_dependency_closure": graph,
        "receiver_module_identities": identities,
        "generic_models_modules": _generic_models_modules(),
        "pythonpath_entries": _pythonpath_entries(),
        "weights_loaded": False,
        "bytecode_cache_scan": "PASS_ABSENT",
    }
    if scope == "target":
        result["draft_config_required_by_target_overlay"] = False
    else:
        result["cli_help"] = _run_v1_cli_help(root, package_name)
        result["formal_hiai_source_patch"] = hiai_source_report
        result["receiver_owned_runtime_loader"] = receiver_loader_report
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3.5 DFlash receiver overlays without weights"
    )
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--package-name", required=True)
    parser.add_argument(
        "--scope",
        choices=("target", "v1-cli"),
        default="target",
        help="target checks the minimal feature overlay; v1-cli checks all V1 runtime files",
    )
    parser.add_argument(
        "--source-models-dir",
        type=Path,
        help=(
            "packaged models/dflash_v1 directory used for byte-for-byte "
            "copy verification"
        ),
    )
    parser.add_argument(
        "--hiai-source",
        type=Path,
        help=(
            "receiver-owned modeling_qwen3_5_hiai_nd.py; required for "
            "--scope v1-cli"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = validate_overlay(
            args.package_dir,
            args.package_name,
            scope=args.scope,
            source_models_dir=args.source_models_dir,
            hiai_source=args.hiai_source,
        )
    except OverlayValidationError as error:
        print(f"overlay validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
