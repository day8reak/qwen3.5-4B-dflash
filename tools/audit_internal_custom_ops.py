#!/usr/bin/env python3
"""Read-only, fail-closed audit for deployed ACLNN packages.

This program deliberately has no framework Python binding or dynamic-loader
path, does not source a vendor setup script, and does not execute an operator.
It only reads package metadata and C headers.  When requested, it invokes
``nm`` as an offline symbol-table reader.

Passing this audit proves the static package/header/shared-object pairing in
``internal_custom_ops_v1.json``.  It does *not* prove tensor shape, dtype,
format, optional-output, alignment, runtime registration, or numerical
correctness.  Those items are intentionally left closed until receiver-side
evidence is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PACKAGE_ROOT / "config" / "internal_custom_ops_v1.json"


class AuditError(RuntimeError):
    """The static audit could not make a safe determination."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read contract {path}: {error}") from error
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise AuditError("unsupported or malformed custom-operator contract")

    packages = contract.get("packages")
    if not isinstance(packages, list) or not packages:
        raise AuditError("contract packages must be a non-empty list")
    seen_ops: set[str] = set()
    seen_providers: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise AuditError("each package contract must be an object")
        op_type = package.get("op_type")
        provider = package.get("provider")
        symbols = package.get("required_aclnn_symbols")
        if not isinstance(op_type, str) or not op_type:
            raise AuditError("each package needs a non-empty op_type")
        if (
            not isinstance(provider, str)
            or not provider
            or Path(provider).parts != (provider,)
            or provider in (".", "..")
        ):
            raise AuditError(f"provider for {op_type} must be one relative name")
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(
                not isinstance(symbol, str)
                or re.fullmatch(r"aclnn[A-Za-z0-9_]+", symbol) is None
                for symbol in symbols
            )
        ):
            raise AuditError(f"invalid required ACLNN symbols for {op_type}")
        if op_type in seen_ops or provider in seen_providers:
            raise AuditError("contract contains duplicate op_type or provider")
        seen_ops.add(op_type)
        seen_providers.add(provider)

    route = contract.get("route")
    if not isinstance(route, dict):
        raise AuditError("contract route must be an object")
    direct = route.get("controller_direct_custom_operator_dependencies")
    if direct != []:
        raise AuditError(
            "the DFlash controller/draft must not directly call receiver target operators"
        )
    target_packages = route.get("required_receiver_target_static_packages")
    if target_packages != ["ChunkGatedDeltaRule", "CacheUpdate"]:
        raise AuditError(
            "the receiver target static gate must include ChunkGatedDeltaRule "
            "and CacheUpdate"
        )
    route_dependencies = {"ChunkGatedDeltaRule", "CacheUpdate"}
    for op_type in sorted(route_dependencies):
        package = next(
            (item for item in packages if item["op_type"] == op_type),
            None,
        )
        if package is None or package.get("required_for_v1_route") is not True:
            raise AuditError(f"{op_type} must be marked as a V1 route dependency")
    if any(
        item.get("required_for_v1_route") is True
        for item in packages
        if item["op_type"] not in route_dependencies
    ):
        raise AuditError("an unused package is incorrectly marked as a V1 dependency")

    known_abi = contract.get("known_aclnn_abi", {})
    known_return_types = contract.get("known_aclnn_return_types", {})
    if not isinstance(known_return_types, dict) or set(known_return_types) != set(
        known_abi
    ):
        raise AuditError("every locked ACLNN ABI must declare one return type")
    if any(value != "aclnnStatus" for value in known_return_types.values()):
        raise AuditError("locked ChunkGatedDeltaRule APIs must return aclnnStatus")

    tensor_contracts = contract.get("tensor_contracts", {})
    for op_type in route_dependencies:
        tensor_contract = tensor_contracts.get(op_type)
        if not isinstance(tensor_contract, dict):
            raise AuditError(f"missing {op_type} tensor contract")
        if any(
            tensor_contract.get(field) is not None
            for field in ("shape", "dtype", "format", "optionality_and_contiguity")
        ):
            raise AuditError(f"unknown {op_type} tensor constraints must stay null")
        if tensor_contract.get("execution_authorized_by_this_contract") is not False:
            raise AuditError("static contract must not authorize operator execution")
        if tensor_contract.get("status") != "unknown_must_be_observed_on_receiver":
            raise AuditError(f"{op_type} tensor contract must remain receiver-observed")
    return contract


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise AuditError(f"audit path escaped vendors root: {path}") from error


def _resolve_within_provider(path: Path, provider: Path, *, kind: str) -> Path:
    """Resolve one evidence file and reject cross-provider symlink reuse."""

    try:
        resolved = path.resolve(strict=True)
        provider_root = provider.resolve(strict=True)
        resolved.relative_to(provider_root)
    except OSError as error:
        raise AuditError(f"cannot resolve {kind} {path}: {error}") from error
    except ValueError as error:
        raise AuditError(
            f"{kind} escapes selected provider: {_relative(path, provider.parent)}"
        ) from error
    return resolved


def _candidate_ops_info_files(provider: Path, target_component: str) -> list[Path]:
    result: list[Path] = []
    try:
        candidates = provider.rglob("*.json")
        for path in candidates:
            if not path.is_file() or "ops-info" not in path.name.casefold():
                continue
            relative_parts = path.relative_to(provider).parts
            if target_component.casefold() in (part.casefold() for part in relative_parts):
                result.append(path)
    except OSError as error:
        raise AuditError(f"cannot scan ops-info below {provider}: {error}") from error
    return sorted(result)


def _json_key_locations(value: Any, wanted: str, prefix: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            if key == wanted:
                locations.append(child_prefix)
            locations.extend(_json_key_locations(child, wanted, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_json_key_locations(child, wanted, f"{prefix}[{index}]"))
    return locations


def _read_ops_info(
    provider: Path,
    vendors_root: Path,
    target_component: str,
    op_type: str,
) -> tuple[dict[str, Any], list[str]]:
    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    files = _candidate_ops_info_files(provider, target_component)
    if not files:
        errors.append(f"no {target_component} ops-info JSON found")
    for path in files:
        resolved = _resolve_within_provider(path, provider, kind="ops-info file")
        item: dict[str, Any] = {
            "path": _relative(path, vendors_root),
            "sha256": _sha256(resolved),
            "op_type_locations": [],
        }
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            item["parse_error"] = str(error)
            errors.append(f"invalid ops-info JSON: {item['path']}")
        else:
            item["op_type_locations"] = _json_key_locations(payload, op_type)
        evidence.append(item)
    matches = [
        item
        for item in evidence
        if item.get("op_type_locations") and "parse_error" not in item
    ]
    if files and not matches:
        errors.append(f"{op_type} is absent from {target_component} ops-info")
    return {
        "target_component": target_component,
        "files": evidence,
        "matched_files": [item["path"] for item in matches],
    }, errors


def _remove_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def _split_c_parameters(text: str) -> list[str]:
    if not text.strip() or text.strip() == "void":
        return []
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(depth - 1, 0)
        elif character == "," and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return pieces


def _canonical_c_type(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\s*([*&])\s*", r"\1", value)
    return value


def _parameter(raw: str) -> dict[str, str]:
    declaration = re.sub(r"\s+", " ", raw.strip())
    declaration = declaration.split("=", maxsplit=1)[0].strip()
    match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^]]*\])?\s*$", declaration)
    if match is None:
        raise AuditError(f"cannot parse C parameter: {raw!r}")
    name = match.group(1)
    c_type = declaration[: match.start(1)].strip()
    if not c_type:
        raise AuditError(f"C parameter has no type: {raw!r}")
    return {
        "name": name,
        "c_type": _canonical_c_type(c_type),
    }


def _extract_header_declarations(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read ACLNN header {path}: {error}") from error
    text = _remove_c_comments(text)
    declarations: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\b(?P<return_type>[A-Za-z_]\w*)\s+"
        r"(?P<name>aclnn[A-Za-z0-9_]+)\s*"
        r"\((?P<parameters>.*?)\)\s*;",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        parameters = [
            _parameter(raw) for raw in _split_c_parameters(match.group("parameters"))
        ]
        declarations.append(
            {
                "symbol": match.group("name"),
                "return_type": _canonical_c_type(match.group("return_type")),
                "parameters": parameters,
                "canonical_signature": _canonical_signature(
                    match.group("name"),
                    parameters,
                    match.group("return_type"),
                ),
            }
        )
    return declarations


def _canonical_signature(
    symbol: str,
    parameters: Iterable[dict[str, str]],
    return_type: str | None = None,
) -> str:
    rendered = ", ".join(
        f"{parameter['c_type']} {parameter['name']}" for parameter in parameters
    )
    prefix = "" if return_type is None else f"{_canonical_c_type(return_type)} "
    return f"{prefix}{symbol}({rendered})"


def _read_headers(
    provider: Path,
    vendors_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    include = provider / "op_api" / "include"
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        headers = sorted(path for path in include.rglob("*.h") if path.is_file())
    except OSError as error:
        raise AuditError(f"cannot scan ACLNN headers below {include}: {error}") from error
    if not headers:
        errors.append("no ACLNN headers found below op_api/include")
    for path in headers:
        resolved = _resolve_within_provider(path, provider, kind="ACLNN header")
        try:
            declarations = _extract_header_declarations(resolved)
        except AuditError as error:
            errors.append(str(error))
            declarations = []
        item = {
            "path": _relative(path, vendors_root),
            "sha256": _sha256(resolved),
            "symbols": sorted({item["symbol"] for item in declarations}),
        }
        evidence.append(item)
        for declaration in declarations:
            occurrence = {
                **declaration,
                "header": item["path"],
            }
            by_symbol.setdefault(declaration["symbol"], []).append(occurrence)
    return by_symbol, evidence, errors


def _shared_objects(provider: Path, vendors_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    lib_dir = provider / "op_api" / "lib"
    try:
        provider_root = provider.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"cannot resolve selected provider {provider}: {error}") from error
    errors: list[str] = []
    try:
        candidates = sorted(
            path
            for path in lib_dir.rglob("*")
            if ".so" in path.name and (path.is_file() or path.is_symlink())
        )
    except OSError as error:
        raise AuditError(f"cannot scan shared objects below {lib_dir}: {error}") from error
    evidence: list[dict[str, Any]] = []
    for path in candidates:
        exists = path.exists()
        item = {
            "path": _relative(path, vendors_root),
            "exists": exists,
            "is_symlink": path.is_symlink(),
            "resolved_within_selected_provider": False,
        }
        if exists:
            try:
                resolved = path.resolve(strict=True)
                relative_resolved = resolved.relative_to(provider_root)
            except OSError:
                item["resolved_path"] = None
                errors.append(f"cannot resolve shared object: {item['path']}")
            except ValueError:
                item["resolved_path"] = None
                errors.append(
                    "shared object escapes selected provider: " + item["path"]
                )
            else:
                item["resolved_within_selected_provider"] = True
                # Keep reports portable and make the later nm invocation use a
                # path proven to remain inside this provider.  Never expose an
                # absolute receiver path in the JSON report.
                item["resolved_path"] = _relative(
                    provider / relative_resolved,
                    vendors_root,
                )
                item["sha256"] = _sha256(resolved)
        if not exists:
            errors.append(f"broken shared-object link: {item['path']}")
        evidence.append(item)
    if not candidates:
        errors.append("no shared object found below op_api/lib")
    return evidence, errors


def _resolve_nm(program: str) -> str:
    if Path(program).name == program:
        resolved = shutil.which(program)
        if resolved is None:
            raise AuditError(f"nm program not found on PATH: {program}")
        return resolved
    path = Path(program)
    if not path.is_file():
        raise AuditError(f"nm program is not a file: {program}")
    return str(path)


def _nm_symbols(program: str, library: Path) -> tuple[list[str], str | None]:
    try:
        completed = subprocess.run(
            [program, "-D", "--defined-only", str(library)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], str(error)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return [], f"nm exited {completed.returncode}: {detail}"
    symbols = sorted(set(re.findall(r"\baclnn[A-Za-z0-9_]+\b", completed.stdout)))
    return symbols, None


def _check_locked_abi(
    required_symbols: list[str],
    declarations: dict[str, list[dict[str, Any]]],
    known_abi: dict[str, Any],
    known_return_types: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for symbol in required_symbols:
        expected = known_abi.get(symbol)
        if expected is None:
            continue
        occurrences = declarations.get(symbol, [])
        signatures = sorted({item["canonical_signature"] for item in occurrences})
        expected_return_type = known_return_types.get(symbol)
        expected_signature = _canonical_signature(
            symbol,
            expected,
            expected_return_type,
        )
        status = "PASS" if signatures == [expected_signature] else "FAIL"
        checks.append(
            {
                "symbol": symbol,
                "status": status,
                "expected_parameter_order": [item["name"] for item in expected],
                "expected_return_type": expected_return_type,
                "expected_signature": expected_signature,
                "observed_signatures": signatures,
            }
        )
        if status == "FAIL":
            errors.append(f"locked ACLNN ABI mismatch for {symbol}")
    return checks, errors


def _audit_package(
    package_contract: dict[str, Any],
    contract: dict[str, Any],
    vendors_root: Path,
    nm_program: str | None,
) -> dict[str, Any]:
    provider_name = package_contract["provider"]
    provider = vendors_root / provider_name
    op_type = package_contract["op_type"]
    required_inventory = package_contract["required_for_environment_inventory"]
    required_route = package_contract["required_for_v1_route"]
    result: dict[str, Any] = {
        "op_type": op_type,
        "provider": provider_name,
        "required_for_environment_inventory": required_inventory,
        "required_for_v1_route": required_route,
        "provider_exists": provider.is_dir(),
        "provider_is_symlink": provider.is_symlink(),
        "status": "FAIL",
        "errors": [],
        "warnings": [],
    }
    if provider.is_symlink():
        result["errors"].append(
            "selected provider root must be a real directory, not a symlink"
        )
        return result
    if not provider.is_dir():
        if not required_inventory and not required_route:
            result["status"] = "ABSENT_OPTIONAL"
        else:
            result["errors"].append("selected provider directory is absent")
        return result

    target_component = contract["target"]["ops_info_directory_component"]
    ops_info, ops_errors = _read_ops_info(
        provider, vendors_root, target_component, op_type
    )
    declarations, headers, header_errors = _read_headers(provider, vendors_root)
    libraries, library_errors = _shared_objects(provider, vendors_root)
    required_symbols = package_contract["required_aclnn_symbols"]
    missing_header_symbols = sorted(
        symbol for symbol in required_symbols if symbol not in declarations
    )
    if missing_header_symbols:
        header_errors.append(
            "missing ACLNN header declarations: " + ", ".join(missing_header_symbols)
        )
    abi_checks, abi_errors = _check_locked_abi(
        required_symbols,
        declarations,
        contract.get("known_aclnn_abi", {}),
        contract.get("known_aclnn_return_types", {}),
    )

    nm_evidence: list[dict[str, Any]] = []
    if nm_program is not None:
        union: set[str] = set()
        for item in libraries:
            if not item["exists"] or not item["resolved_within_selected_provider"]:
                continue
            path = vendors_root / item["resolved_path"]
            symbols, error = _nm_symbols(nm_program, path)
            nm_item: dict[str, Any] = {
                "path": item["path"],
                "aclnn_exports": symbols,
            }
            if error is not None:
                nm_item["error"] = error
                library_errors.append(f"cannot inspect exports for {item['path']}: {error}")
            else:
                union.update(symbols)
            nm_evidence.append(nm_item)
        missing_exports = sorted(set(required_symbols) - union)
        if missing_exports:
            library_errors.append(
                "missing ACLNN shared-object exports: " + ", ".join(missing_exports)
            )

    result.update(
        {
            "ops_info": ops_info,
            "headers": headers,
            "required_header_symbols": required_symbols,
            "locked_abi_checks": abi_checks,
            "shared_objects": libraries,
            "nm": {
                "requested": nm_program is not None,
                "libraries": nm_evidence,
            },
        }
    )
    result["errors"].extend(
        [*ops_errors, *header_errors, *library_errors, *abi_errors]
    )
    result["status"] = "PASS" if not result["errors"] else "FAIL"
    return result


def _provider_cache_signature(
    provider: Path,
    vendors_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    declarations, _, errors = _read_headers(provider, vendors_root)
    signatures = declarations.get("aclnnCacheUpdateGetWorkspaceSize", [])
    compact = [
        {
            "provider": provider.name,
            "header": item["header"],
            "canonical_signature": item["canonical_signature"],
            "parameters": item["parameters"],
        }
        for item in signatures
    ]
    return compact, errors


def _cache_update_conflict(
    contract: dict[str, Any],
    vendors_root: Path,
    nm_program: str | None,
) -> dict[str, Any]:
    conflict_contract = contract.get("known_conflicts", {}).get("CacheUpdate", {})
    selected = conflict_contract.get("selected_provider")
    signatures: list[dict[str, Any]] = []
    scan_errors: list[str] = []
    provider_names: list[str] = []
    try:
        children = sorted(path for path in vendors_root.iterdir() if path.is_dir())
    except OSError as error:
        raise AuditError(f"cannot enumerate vendors root {vendors_root}: {error}") from error
    for provider in children:
        if provider.is_symlink():
            scan_errors.append(
                f"provider root symlink was not scanned: {provider.name}"
            )
            continue
        include = provider / "op_api" / "include"
        if not include.is_dir():
            continue
        observed, errors = _provider_cache_signature(provider, vendors_root)
        scan_errors.extend(errors)
        if observed:
            provider_names.append(provider.name)
            signatures.extend(observed)

    variants: dict[str, list[str]] = {}
    for item in signatures:
        variants.setdefault(item["canonical_signature"], []).append(item["provider"])
    variants = {
        signature: sorted(set(providers))
        for signature, providers in sorted(variants.items())
    }

    export_providers: dict[str, list[str]] = {}
    nm_evidence: list[dict[str, Any]] = []
    if nm_program is not None:
        for provider_name in sorted(set(provider_names)):
            provider = vendors_root / provider_name
            libraries, errors = _shared_objects(provider, vendors_root)
            scan_errors.extend(errors)
            for library in libraries:
                if (
                    not library["exists"]
                    or not library["resolved_within_selected_provider"]
                ):
                    continue
                symbols, error = _nm_symbols(
                    nm_program, vendors_root / library["resolved_path"]
                )
                item: dict[str, Any] = {
                    "provider": provider_name,
                    "path": library["path"],
                    "cache_update_exports": [
                        symbol for symbol in symbols if symbol.startswith("aclnnCacheUpdate")
                    ],
                }
                if error is not None:
                    item["error"] = error
                    scan_errors.append(
                        f"cannot inspect CacheUpdate exports for {library['path']}: {error}"
                    )
                for symbol in item["cache_update_exports"]:
                    export_providers.setdefault(symbol, []).append(provider_name)
                nm_evidence.append(item)
    export_providers = {
        symbol: sorted(set(providers))
        for symbol, providers in sorted(export_providers.items())
    }
    duplicate_exports = {
        symbol: providers
        for symbol, providers in export_providers.items()
        if len(providers) > 1
    }
    abi_conflict = len(variants) > 1

    discriminator_checks: list[dict[str, Any]] = []
    expected_by_provider = conflict_contract.get("discriminator_parameters", {})
    for item in signatures:
        expected = expected_by_provider.get(item["provider"])
        if not isinstance(expected, dict):
            continue
        actual = {
            parameter["name"]: parameter["c_type"] for parameter in item["parameters"]
        }
        mismatches = {
            name: {"expected": c_type, "observed": actual.get(name)}
            for name, c_type in expected.items()
            if actual.get(name) != c_type
        }
        discriminator_checks.append(
            {
                "provider": item["provider"],
                "header": item["header"],
                "status": "PASS" if not mismatches else "FAIL",
                "mismatches": mismatches,
            }
        )

    return {
        "selected_provider": selected,
        "providers_with_header_symbol": sorted(set(provider_names)),
        "abi_conflict_detected": abi_conflict,
        "signature_variants": variants,
        "discriminator_checks": discriminator_checks,
        "nm_requested": nm_program is not None,
        "nm_evidence": nm_evidence,
        "duplicate_export_providers": duplicate_exports,
        "blocks_v1_full_prefix_route": bool(
            scan_errors
            or any(item["status"] == "FAIL" for item in discriminator_checks)
        ),
        "reason": (
            "The controller/draft do not call CacheUpdate directly, but the "
            "receiver HIAI full-attention path uses it inside each isolated "
            "full-prefix call. ABI variants alone do not block the route when "
            "the selected customize_scatter header/shared object pair passes; "
            "runtime link order still requires trace evidence."
        ),
        "scan_errors": scan_errors,
    }


def audit(
    vendors_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    nm: str | None = None,
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    contract = _load_contract(contract_path)
    vendors_root = vendors_root.resolve(strict=True)
    if not vendors_root.is_dir():
        raise AuditError(f"vendors root is not a directory: {vendors_root}")
    nm_program = _resolve_nm(nm) if nm is not None else None

    package_results = [
        _audit_package(package, contract, vendors_root, nm_program)
        for package in contract["packages"]
    ]
    required_inventory_failures = [
        item["op_type"]
        for item in package_results
        if item["required_for_environment_inventory"] and item["status"] != "PASS"
    ]
    route_failures = [
        item["op_type"]
        for item in package_results
        if item["required_for_v1_route"] and item["status"] != "PASS"
    ]
    malformed_optional = [
        item["op_type"]
        for item in package_results
        if not item["required_for_environment_inventory"]
        and item["status"] == "FAIL"
    ]
    cache_conflict = _cache_update_conflict(
        contract, vendors_root, nm_program
    )
    selected_cache_checks = [
        item
        for item in cache_conflict["discriminator_checks"]
        if item["provider"] == cache_conflict["selected_provider"]
    ]
    if any(item["status"] == "FAIL" for item in selected_cache_checks):
        required_inventory_failures.append("CacheUpdate:selected-provider-ABI")
    if cache_conflict["scan_errors"]:
        required_inventory_failures.append("CacheUpdate:provider-scan")

    inventory_status = (
        "PASS"
        if not required_inventory_failures and not malformed_optional
        else "FAIL"
    )
    route_status = "PASS" if not route_failures else "FAIL"
    static_status = (
        "PASS" if inventory_status == "PASS" and route_status == "PASS" else "FAIL"
    )
    tensor_contract = contract["tensor_contracts"]["ChunkGatedDeltaRule"]
    warnings: list[str] = []
    if cache_conflict["abi_conflict_detected"]:
        warnings.append(
            "CacheUpdate same-name ABI variants exist; keep headers and shared objects pinned to customize_scatter"
        )
    if cache_conflict["duplicate_export_providers"]:
        warnings.append(
            "nm found duplicate CacheUpdate export providers; runtime link order must not select a shadow provider"
        )
    if malformed_optional:
        warnings.append(
            "optional providers are present but incomplete: " + ", ".join(malformed_optional)
        )

    return {
        "schema_version": 1,
        "status": static_status,
        "audit_scope": contract["scope"],
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
            "contract_id": contract["contract_id"],
        },
        "vendors_root": str(vendors_root),
        "target": contract["target"],
        "inventory_status": inventory_status,
        "required_inventory_failures": sorted(set(required_inventory_failures)),
        "v1_route": {
            "name": contract["route"]["name"],
            "static_package_binding_status": route_status,
            "controller_direct_custom_operator_dependencies": contract["route"][
                "controller_direct_custom_operator_dependencies"
            ],
            "required_receiver_target_static_packages": contract["route"][
                "required_receiver_target_static_packages"
            ],
            "failed_dependencies": route_failures,
            "target_cache": contract["route"]["target_cache"],
            "gdn_state_transaction": contract["route"]["gdn_state_transaction"],
            "execution_authorized": False,
            "execution_blocker": tensor_contract["status"],
        },
        "binding": contract["route"]["binding"],
        "packages": package_results,
        "cache_update_same_name_abi": cache_conflict,
        "tensor_contracts": contract["tensor_contracts"],
        "warnings": warnings,
        "safety": {
            "shared_libraries_loaded": False,
            "operators_executed": False,
            "torch_ops_assumed": False,
            "nm_requested": nm_program is not None,
            "nm_program": nm_program,
            "unknown_shape_dtype_format_inferred": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statically audit deployed Ascend custom-operator packages."
    )
    parser.add_argument(
        "--vendors-root",
        type=Path,
        required=True,
        help="Receiver-side directory whose direct children are vendor packages.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Audit contract (defaults to config/internal_custom_ops_v1.json).",
    )
    parser.add_argument(
        "--nm",
        nargs="?",
        const="nm",
        default=None,
        metavar="PROGRAM",
        help="Optionally inspect dynamic exports with nm; no library is loaded.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit(args.vendors_root, args.contract, args.nm)
    except (AuditError, OSError) as error:
        report = {
            "schema_version": 1,
            "status": "ERROR",
            "error": str(error),
            "safety": {
                "shared_libraries_loaded": False,
                "operators_executed": False,
                "torch_ops_assumed": False,
            },
        }
        exit_code = 2
    else:
        exit_code = 0 if report["status"] == "PASS" else 1
    json.dump(
        report,
        sys.stdout,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
