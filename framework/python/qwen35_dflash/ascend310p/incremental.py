"""Functional, explicit-state AIR graphs for the two-pass chunk-GDR route.

No Python bridge/cache object survives an invocation. Verify computes acceptance
and executes both GDR passes in the same OM. Only fully committed states cross
the OM boundary; the per-layer commit capsules remain graph intermediates.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import AirGraphSpec, CustomOpExportSpec
from .incremental_plan import ABI, ATTENTION_EXPORT_POLICY, DRAFT_LENGTH_POLICY

CHUNK_ABI = ABI


def conv_chunk(x: Tensor, state: Tensor, weight: Tensor, bias: Tensor | None):
    """The branch's causal-conv formula, including every valid-prefix state."""
    rows = x.shape[-1]
    history = torch.cat((state, x), dim=-1).to(weight.dtype)
    output = F.silu(F.conv1d(history, weight.unsqueeze(1), bias, groups=x.shape[1]))
    # Window i is history[..., i+1:i+1+K], the state after input row i.
    # TorchAir cannot lower aten.unfold. Stack K shifted slices instead of
    # creating an overlapping view; K is the static convolution width (4).
    bank = torch.stack(
        [history[..., offset + 1 : offset + 1 + rows]
         for offset in range(state.shape[-1])],
        dim=-1,
    )
    return output[..., -rows:].to(x.dtype), bank.permute(0, 2, 1, 3).contiguous().to(
        x.dtype
    )


def prefix_state(bank: Tensor, rows: Tensor) -> Tensor:
    return torch.index_select(bank, 1, rows.to(torch.long) - 1).squeeze(1).contiguous()


def accepted_prefix_length(input_ids: Tensor, top1: Tensor, valid_rows: Tensor) -> Tensor:
    """Count matching proposals before the first mismatch, ignoring padding.

    Use the INT32 Cumsum/ReduceSum route from the quant AIR transaction graph:
    TorchAir has no amin/min.dim/cumprod converter on the receiver toolchain.
    The scan contains at most 15 bits, so integer accumulation is exact. Keep
    the public accepted_count INT64 ABI and accepted+1 GDR commit unchanged.
    """
    indices = torch.arange(input_ids.shape[1] - 1, device=input_ids.device)
    proposal_count = valid_rows.to(torch.long) - 1
    within_requested = indices[None, :] < proposal_count[:, None]
    mismatch = (input_ids[:, 1:] != top1[:, :-1]) & within_requested
    cumulative_mismatches = torch.cumsum(
        mismatch.to(torch.int32), dim=1, dtype=torch.int32
    )
    accepted_mask = within_requested & cumulative_mismatches.eq(0)
    return accepted_mask.to(torch.int32).sum(dim=1, dtype=torch.int32).to(torch.long)


def copy_cache_rows(cache: Tensor, dim: int, positions: Tensor, values: Tensor) -> Tensor:
    """Replace complete cache rows at distinct positions without mutating input.

    Every caller constructs consecutive positions within the locked capacity.
    Repeating the row indices makes scatter equivalent to index_copy here;
    TorchAir lowers scatter.src to ScatterElements, while index_copy has no GE
    converter on the receiver route.
    """
    index_shape = [1] * cache.ndim
    index_shape[dim] = -1
    # Materialize the static head/channel repeats with Tile. This is the quant
    # AIR branch's cache-index route; dynamic BroadcastTo shape inputs can fail
    # receiver ATC auto-tiling even when the inferred index shape is correct.
    repeats = list(values.shape)
    repeats[dim] = 1
    indices = positions.reshape(index_shape).repeat(*repeats)
    return torch.scatter(cache, dim, indices, values)


def update_paged(cache: Tensor, values: Tensor, positions: Tensor) -> Tensor:
    """Functional row writes in the receiver's [blocks,H*D/16,64,16] layout.

    Padded rows can overwrite only uncommitted slots. Capacity includes a
    private 64-row scratch tail, so even the final short gear stays in bounds.
    """
    blocks, width, block_size, tile = cache.shape
    rows = cache.permute(0, 2, 1, 3).reshape(blocks * block_size, width, tile)
    values = values.reshape(values.shape[1], width, tile)
    rows = copy_cache_rows(rows, 0, positions, values)
    return (
        rows.reshape(blocks, block_size, width, tile).permute(0, 2, 1, 3).contiguous()
    )


class AirTargetAttention(nn.Module):
    """Receiver projections/RoPE/fused attention with explicit KV outputs."""

    def __init__(self, base: nn.Module, operation: Callable, rotary: Callable):
        super().__init__()
        self.base, self.operation, self.apply_rotary = base, operation, rotary

    def forward(self, x, key_cache, value_cache, positions, mask):
        base = self.base
        shape = x.shape[:-1]
        view = (*shape, -1, base.head_dim)
        query, gate = torch.chunk(
            base.q_proj(x).view(*shape, -1, base.head_dim * 2), 2, dim=-1
        )
        query = base.q_norm(query.reshape(view))
        key = base.k_norm(base.k_proj(x).view(view))
        value = base.v_proj(x).view(view)
        cosine, sine = base.rotary_emb(x, positions.unsqueeze(0))
        query, key = self.apply_rotary(
            query.transpose(1, 2), key.transpose(1, 2), cosine, sine
        )
        key_cache = update_paged(key_cache, key.transpose(1, 2), positions)
        value_cache = update_paged(value_cache, value, positions)
        query = query.contiguous()
        query_shape = query.shape
        query_nz = base.transform_nd_2_nz(query).reshape(
            1, base.num_heads * base.head_dim // 16, shape[1], 16
        )
        output = self.operation(
            query=query_nz,
            key=[key_cache],
            value=[value_cache],
            # The receiver frontend exposes lengths as SymInt[], and GE puts
            # them in INT64 all_seq_lengths_q. Follow the quant AIR static
            # route: use physical capacity and the runtime causal/prefix mask.
            # pse_shift is an optional FP16 bias, never a sequence-length slot.
            all_seq_lengths_q=[base.kv_max_len],
            actual_seq_lengths_q=[shape[1]],
            actual_seq_lengths_kv=[base.kv_max_len],
            block_table=base.block_table,
            num_heads=base.num_heads,
            num_key_value_heads=base.num_key_value_heads,
            block_size=base.block_size,
            input_layout="BNSD",
            scale_value=base.scaling,
            inner_precise=2,
            atten_mask=mask.to(torch.float16),
        )
        output = (
            base.transform_nz_2_nd(output.reshape(query_shape))
            .transpose(1, 2)
            .contiguous()
            .reshape(*shape, -1)
        )
        return (
            base.o_proj(output * torch.sigmoid(gate.reshape(*shape, -1))),
            key_cache,
            value_cache,
        )


class AirGdn(nn.Module):
    def __init__(self, base: nn.Module, operation: Callable):
        super().__init__()
        self.base, self.operation = base, operation

    def forward(self, x, conv, recurrent, valid_rows):
        base = self.base
        batch, rows, _ = x.shape
        mixed, bank = conv_chunk(
            base.in_proj_qkv(x).transpose(1, 2),
            conv,
            base.conv1d.weight.squeeze(1),
            base.conv1d.bias,
        )
        query, key, value = torch.split(
            mixed.transpose(1, 2), [base.key_dim, base.key_dim, base.value_dim], dim=-1
        )
        query = query.reshape(batch, rows, -1, base.head_k_dim)
        key = key.reshape(batch, rows, -1, base.head_k_dim)
        value = value.reshape(batch, rows, -1, base.head_v_dim).contiguous()
        repeat = base.num_v_heads // base.num_k_heads
        if repeat > 1:
            query, key = (
                query.repeat_interleave(repeat, dim=2),
                key.repeat_interleave(repeat, dim=2),
            )
        beta = base.in_proj_b(x).sigmoid()
        g = -base.A_log.float().exp() * F.softplus(
            base.in_proj_a(x).float() + base.dt_bias
        )
        initial = recurrent.float()
        output, final = self.operation(
            query,
            key,
            value,
            g=g,
            beta=beta,
            effective_length=valid_rows,
            chunk_size=1 if rows == 1 else 64,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        z = base.in_proj_z(x).reshape(-1, base.head_v_dim)
        output = base.norm(output.reshape(-1, base.head_v_dim), z).reshape(
            batch, rows, -1
        )
        capsule = (query, key, value, g, beta, initial, bank)
        return (
            base.out_proj(output),
            prefix_state(bank, valid_rows),
            final.to(recurrent.dtype),
            capsule,
        )


class TargetRowsGraph(nn.Module):
    def __init__(
        self,
        target,
        *,
        rows: int,
        verify: bool,
        feature_layers: tuple[int, ...],
        gdr: Callable,
        attention: Callable,
        rotary: Callable,
    ):
        super().__init__()
        model = target.dflash_execution_model
        self.body = model.language_model
        self.embedding = (
            getattr(target, "_target_quantized_embedding", None)
            or target.get_input_embeddings()
        )
        # The bridge's public embedding getters retain the FP16 checkpoint
        # modules for Draft. Target must use the execution model's W8A8 head.
        self.head = getattr(model, "lm_head", None)
        if not isinstance(self.head, nn.Module):
            raise TypeError("incremental Target requires execution-model lm_head")
        self.rows, self.verify, self.feature_layers = rows, verify, feature_layers
        self.cache_capacity = target.kv_cache_max_len
        self.blocks = nn.ModuleList(
            [
                AirGdn(layer.linear_attn, gdr)
                if layer.block_type == "linear_attention"
                else AirTargetAttention(layer.self_attn, attention, rotary)
                for layer in self.body.layers
            ]
        )
        self.linear_indices = tuple(
            i for i, block in enumerate(self.blocks) if isinstance(block, AirGdn)
        )
        self.commit = TargetCommitGraph(gdr, len(self.linear_indices))

    def forward(self, input_ids, start_position, valid_rows, *state):
        positions = start_position + torch.arange(
            self.rows, dtype=torch.long, device=input_ids.device
        )
        logical_end = start_position + valid_rows.to(torch.long)
        columns = torch.arange(self.cache_capacity, device=input_ids.device)
        visible = (columns[None, :] <= positions[:, None]) & (
            columns[None, :] < logical_end
        )
        mask = torch.where(visible, 0.0, float("-inf"))[None, None]
        hidden = self.embedding(input_ids).to(torch.float16)
        row_valid = (
            torch.arange(self.rows, device=input_ids.device) < valid_rows.to(torch.long)
        )[:, None]
        next_state, capsules, features = [], [], []
        for index, (layer, block) in enumerate(zip(self.body.layers, self.blocks)):
            normalized = layer.input_layernorm(hidden)
            if isinstance(block, AirGdn):
                mixed, conv, recurrent, capsule = block(
                    normalized, state[2 * index], state[2 * index + 1], valid_rows
                )
                next_state.extend((conv, recurrent))
                if self.verify:
                    capsules.extend(capsule)
            else:
                mixed, key, value = block(
                    normalized,
                    state[2 * index],
                    state[2 * index + 1],
                    positions,
                    mask,
                )
                next_state.extend((key, value))
            hidden = hidden + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            # Receiver GDR padding outputs are not part of its valid-row
            # contract. Sanitize them before a later layer writes paged KV.
            hidden = torch.where(row_valid[None], hidden, torch.zeros_like(hidden))
            if index in self.feature_layers:
                features.append(hidden.clone())
        hidden = self.body.norm(hidden)
        # Prefill/decode applies the full-vocabulary head to one real row only.
        head_rows = (
            hidden
            if self.verify
            else torch.index_select(hidden, 1, valid_rows.to(torch.long) - 1)
        )
        top1 = torch.argmax(self.head(head_rows), dim=-1)
        acceptance_output = ()
        if self.verify:
            accepted = accepted_prefix_length(input_ids, top1, valid_rows)
            committed = self.commit((accepted + 1).to(torch.int16), *capsules)
            for offset, layer_index in enumerate(self.linear_indices):
                next_state[2 * layer_index : 2 * layer_index + 2] = committed[
                    2 * offset : 2 * offset + 2
                ]
            acceptance_output = (accepted,)
        # One Draft input gear serves prompt chunks and committed verify rows.
        feature_output = ()
        if features:
            features = torch.cat(features, dim=-1)
            if self.rows < 64:
                features = F.pad(features, (0, 0, 0, 64 - self.rows))
            feature_output = (features,)
        return (top1, *acceptance_output, *feature_output, *next_state)


class TargetCommitGraph(nn.Module):
    def __init__(self, operation: Callable, layers: int):
        super().__init__()
        self.operation, self.layers = operation, layers

    def forward(self, committed_rows, *capsules):
        result = []
        for index in range(self.layers):
            query, key, value, g, beta, initial, bank = capsules[
                index * 7 : index * 7 + 7
            ]
            _, final = self.operation(
                query,
                key,
                value,
                g=g,
                beta=beta,
                effective_length=committed_rows,
                chunk_size=64,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            result.extend((prefix_state(bank, committed_rows), final.to(bank.dtype)))
        return tuple(result)


class DraftContextGraph(nn.Module):
    """Append only committed Target features; transient noise never enters KV."""

    def __init__(self, draft, rows):
        super().__init__()
        self.draft, self.rows = draft, rows

    def forward(self, features, start_position, *state):
        draft = self.draft
        projected = draft.hidden_norm(draft.fc(features))
        positions = start_position + torch.arange(
            self.rows, dtype=torch.long, device=features.device
        )
        cosine, sine = draft.rotary(positions[None], projected.dtype)
        cosine, sine = cosine[:, None], sine[:, None]
        result = []
        from .quant_factory import _rotate_half

        for index, layer in enumerate(draft.layers):
            base, config = layer.self_attn, draft.config
            key = base.k_norm(
                base.k_proj(projected).reshape(
                    1, self.rows, config.num_key_value_heads, config.head_dim
                )
            ).transpose(1, 2)
            key = key * cosine + _rotate_half(key) * sine
            value = (
                base.v_proj(projected)
                .reshape(1, self.rows, config.num_key_value_heads, config.head_dim)
                .transpose(1, 2)
            )
            result.extend(
                (
                    copy_cache_rows(state[2 * index], 2, positions, key),
                    copy_cache_rows(state[2 * index + 1], 2, positions, value),
                )
            )
        return tuple(result)


class DraftProposeGraph(nn.Module):
    def __init__(self, draft, embedding, head):
        super().__init__()
        self.draft, self.embedding, self.head = draft, embedding, head

    def forward(self, anchor, context_length, proposal_count, *state):
        draft, config = self.draft, self.draft.config
        block_ids = torch.cat(
            (
                anchor.reshape(1, 1),
                torch.full(
                    (1, config.block_size - 1),
                    config.mask_token_id,
                    dtype=torch.long,
                    device=anchor.device,
                ),
            ),
            dim=1,
        )
        hidden = self.embedding(block_ids) * config.input_embedding_scale
        offsets = torch.arange(config.block_size, device=anchor.device)
        positions = context_length + offsets
        cosine, sine = draft.rotary(positions[None], hidden.dtype)
        capacity = state[0].shape[2]
        context_positions = torch.arange(capacity, device=anchor.device)
        key_positions = torch.cat((context_positions, positions))
        valid = torch.cat(
            (
                context_positions < context_length,
                # A short native block contains anchor + K masks. Hidden
                # rows beyond K must never become attention keys, including
                # in the final non-causal layer.
                offsets <= proposal_count.to(torch.long),
            )
        )
        distance = positions[:, None] - key_positions[None, :]
        for index, layer in enumerate(draft.layers):
            base = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            query = base.q_norm(
                base.q_proj(normalized).reshape(
                    1, config.block_size, config.num_attention_heads, config.head_dim
                )
            ).transpose(1, 2)
            key = base.k_norm(
                base.k_proj(normalized).reshape(
                    1, config.block_size, config.num_key_value_heads, config.head_dim
                )
            ).transpose(1, 2)
            value = (
                base.v_proj(normalized)
                .reshape(
                    1, config.block_size, config.num_key_value_heads, config.head_dim
                )
                .transpose(1, 2)
            )
            query, key = base.ops.rotary(query, key, cosine, sine)
            key, value = (
                torch.cat((state[2 * index], key), dim=2),
                torch.cat((state[2 * index + 1], value), dim=2),
            )
            mask = valid[None, :].expand(config.block_size, -1)
            if base.is_causal:
                mask = mask & (distance >= 0)
            if base.sliding_window is not None:
                mask = mask & (distance < base.sliding_window)
                if not base.is_causal:
                    mask = mask & (-distance < base.sliding_window)
            mixed = base.ops.attention(
                query,
                key,
                value,
                mask[None, None],
                base.scale,
                config.num_key_value_groups,
            )
            mixed = (
                mixed.transpose(1, 2)
                .contiguous()
                .reshape(1, config.block_size, config.query_width)
            )
            hidden = hidden + base.o_proj(mixed)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        top1 = draft.ops.top1(draft.norm(hidden)[:, 1:], self.head.weight)
        active = offsets[1:] <= proposal_count.to(torch.long)
        return (torch.where(active[None], top1, torch.zeros_like(top1)),)


class DraftGraph(nn.Module):
    """One OM appends context and generates a block, sharing all Draft weights."""

    def __init__(self, draft, embedding, head):
        super().__init__()
        self.context = DraftContextGraph(draft, 64)
        self.propose = DraftProposeGraph(draft, embedding, head)

    def forward(self, features, start_position, valid_rows, anchor, proposal_count, *state):
        visible = torch.arange(64, device=features.device) < valid_rows.to(torch.long)
        features = torch.where(
            visible[None, :, None], features, torch.zeros_like(features)
        )
        updated = self.context(features, start_position, *state)
        proposals = self.propose(
            anchor, start_position + valid_rows.to(torch.long), proposal_count, *updated
        )
        return (*proposals, *updated)


def tensor_spec(name, tensor):
    return {
        "name": name,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }


def incremental_graph_specs(
    target,
    draft,
    *,
    capacity: int,
    metadata: dict,
    gdr: Callable,
    attention: Callable,
    rotary: Callable,
    custom_ops: tuple[CustomOpExportSpec, ...] = (),
    include_ordinary_decode: bool = True,
):
    """Build static graphs, with shapes derived from the loaded models."""
    device = target.requested_device
    state = tuple(t for pair in target._fresh_hybrid_cache(batch_size=1) for t in pair)
    names, gdn_names, kv_names, capsules, capsule_names = [], [], [], [], []
    layers = target.dflash_execution_model.language_model.layers
    for index, layer in enumerate(layers):
        linear = layer.block_type == "linear_attention"
        pair = [
            f"t{index}_{suffix}"
            for suffix in (("conv", "recurrent") if linear else ("key", "value"))
        ]
        names.extend(pair)
        (gdn_names if linear else kv_names).extend(pair)
        if linear:
            base = layer.linear_attn
            shapes = ((1, 16, base.num_v_heads, base.head_k_dim),) * 2 + (
                (1, 16, base.num_v_heads, base.head_v_dim),
                (1, 16, base.num_v_heads),
                (1, 16, base.num_v_heads),
                tuple(state[2 * index + 1].shape),
                (1, 16, base.conv_dim, base.conv_kernel_size),
            )
            for suffix, shape, dtype in zip(
                ("q", "k", "v", "g", "beta", "initial", "conv_bank"),
                shapes,
                (
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.float32,
                    torch.float16,
                    torch.float32,
                    torch.float16,
                ),
            ):
                capsule_names.append(f"c{index}_{suffix}")
                # Shape documentation only: capsules are internal graph values.
                capsules.append(torch.empty(shape, dtype=dtype, device="meta"))
    start = torch.zeros(1, dtype=torch.long, device=device)
    valid = torch.ones(1, dtype=torch.int16, device=device)
    draft_names = tuple(
        f"d{i}_{kind}" for i in range(len(draft.layers)) for kind in ("key", "value")
    )
    draft_state = tuple(
        torch.zeros(
            (
                1,
                draft.config.num_key_value_heads,
                target.kv_cache_max_len,
                draft.config.head_dim,
            ),
            dtype=torch.float16,
            device=device,
        )
        for _ in draft_names
    )
    contract = {
        "abi": CHUNK_ABI,
        "capacity": capacity,
        "cache_capacity": target.kv_cache_max_len,
        "block_size": 16,
        "prefill_rows": 64,
        "target_states": [tensor_spec(n, t) for n, t in zip(names, state)],
        "draft_states": [tensor_spec(n, t) for n, t in zip(draft_names, draft_state)],
        "gdn_states": gdn_names,
        "kv_states": kv_names,
        "capsules": [tensor_spec(n, t) for n, t in zip(capsule_names, capsules)],
        "vocab_size": draft.config.vocab_size,
        "feature_width": draft.config.feature_size,
        "state_policy": "in-graph-acceptance-two-pass-gdr-atomic-fp16-state-output",
        "attention_export": ATTENTION_EXPORT_POLICY,
        "draft_length_policy": DRAFT_LENGTH_POLICY,
        "single_row_policy": "ordinary_decode1_chunk1; speculative_fallback_verify16_valid1",
        "commit_capsules": "internal_to_target_verify_not_external_OM_IO",
    }
    common = {**metadata, "incremental_contract": contract}
    specs = []

    def add(name, model, args, inputs, outputs, output_tensors, ops=()):
        meta = {
            **common,
            "tensor_abi": {
                "inputs": [tensor_spec(n, t) for n, t in zip(inputs, args)],
                "outputs": [tensor_spec(n, t) for n, t in zip(outputs, output_tensors)],
            },
        }
        if not ops:
            meta.pop("custom_op_export_contract", None)
            meta.pop("custom_op_export_contracts", None)
            meta.pop("standard_op_export_contracts", None)
        specs.append(
            AirGraphSpec(
                name=name,
                role=name.replace("_", "-"),
                model=model,
                example_args=tuple(args),
                input_names=tuple(inputs),
                output_names=tuple(outputs),
                metadata=meta,
                custom_ops=ops,
            )
        )

    for name, rows, verify in (
        ("target_prefill", 64, False),
        ("target_decode", 1, False),
        ("target_verify", 16, True),
    ):
        if rows == 1 and not include_ordinary_decode:
            continue
        ids = torch.zeros((1, rows), dtype=torch.long, device=device)
        feature_tensor = torch.zeros(
            (1, 64, draft.config.feature_size), dtype=torch.float16, device=device
        )
        top1 = torch.zeros((1, rows if verify else 1), dtype=torch.long, device=device)
        out_names, out_tensors = ["target_top1"], [top1]
        if verify:
            out_names.append("accepted_count")
            out_tensors.append(start)
        if rows != 1:
            out_names.append("features")
            out_tensors.append(feature_tensor)
        selected_names = names
        out_names += selected_names
        state_by_name = dict(zip(names, state))
        out_tensors += [state_by_name[n] for n in selected_names]
        graph = TargetRowsGraph(
            target,
            rows=rows,
            verify=verify,
            feature_layers=tuple(draft.config.target_layer_ids) if rows != 1 else (),
            gdr=gdr,
            attention=attention,
            rotary=rotary,
        )
        add(
            name,
            graph,
            (ids, start, valid, *state),
            ("input_ids", "start_position", "valid_rows", *names),
            out_names,
            out_tensors,
            custom_ops,
        )
    embedding = (
        target.get_input_embeddings()
    )  # Draft uses the authoritative FP16 embedding.
    features = torch.zeros(
        (1, 64, draft.config.feature_size), dtype=torch.float16, device=device
    )
    add(
        "draft",
        DraftGraph(draft, embedding, target.get_output_embeddings()),
        (features, start, valid, start.clone(),
         torch.full_like(valid, 15), *draft_state),
        ("features", "start_position", "valid_rows", "anchor", "proposal_count", *draft_names),
        ("draft_top1", *draft_names),
        (torch.zeros((1, 15), dtype=torch.long, device=device), *draft_state),
    )
    return tuple(specs)
