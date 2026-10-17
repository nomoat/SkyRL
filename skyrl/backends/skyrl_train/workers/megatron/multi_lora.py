"""Batched multi-tenant LoRA on top of the single-adapter training state.

The model's LoRA layers are Megatron-Bridge ``MultiLoRALinear`` modules with
``n`` slots. Slot 0 is the *live* adapter: the only trainable slot, so it alone
sits in the DDP buffers and the optimizer, and ``AdapterStore`` swaps tenants in
and out of it exactly as for a single-adapter model. Optimizer steps,
checkpoints and the vLLM export therefore keep operating on one adapter.

Slots 1..n-1 are scratch: a batched forward/backward can lend them to other
tenants (weights copied in, gradients copied out) so several adapters run in one
pass. Outside such a pass they are zeroed.

Checkpoints and exports use the single-adapter names (``.adapter.``): slot 0 is
saved and loaded under those names, so files are interchangeable between the
single- and multi-LoRA builds.
"""

from __future__ import annotations

import math
import re
import zlib
from contextlib import contextmanager
from typing import Iterable, Iterator, List

import torch

_SLOT_RE = re.compile(r"\.adapters\.(\d+)\.")


def _chunks(model) -> List[torch.nn.Module]:
    return list(model) if isinstance(model, (list, tuple)) else [model]


def multi_lora_modules(model) -> Iterator[torch.nn.Module]:
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    for chunk in _chunks(model):
        for module in chunk.modules():
            if isinstance(module, MultiLoRALinear):
                yield module


def canonical_adapter_name(name: str) -> str:
    """Map slot 0 of a multi-LoRA layer to the single-adapter parameter name."""
    return name.replace(".adapters.0.", ".adapter.")


def slot_of(name: str) -> int | None:
    match = _SLOT_RE.search(name)
    return int(match.group(1)) if match else None


def prepare_slots(model, rank: int, alpha: float) -> int:
    """Bind rank/alpha to every slot and make only slot 0 trainable.

    Runs before DDP wrapping, so the scratch slots never enter the DDP buffers
    or the optimizer. Returns the number of multi-LoRA layers.
    """
    count = 0
    for module in multi_lora_modules(model):
        for idx in range(module.n_adapters):
            module.init_adapter_slot(idx, rank, alpha)
            for param in module.adapters[idx].parameters():
                param.requires_grad_(idx == 0)
        count += 1
    return count


def lora_parameters(model) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """(canonical name, param) for every live adapter parameter, in model order."""
    for chunk in _chunks(model):
        for name, param in chunk.named_parameters():
            if ".adapter." in name or slot_of(name) == 0:
                yield canonical_adapter_name(name), param


@torch.no_grad()
def deterministic_lora_init(model, seed: int, init_method: str) -> int:
    """Initialize live LoRA weights from (seed, canonical parameter name).

    The result does not depend on how many other parameters were created first,
    so a single-adapter and a multi-LoRA build start from identical weights.
    lora_A uses the method's full-fan-in distribution (as Megatron-Bridge does
    at TP=1); lora_B is zero. Returns the number of tensors initialized.
    """
    count = 0
    for name, param in lora_parameters(model):
        if "linear_out" in name:
            param.zero_()
        elif "linear_in" in name:
            gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            rank, fan_in = param.shape
            values = torch.empty(param.shape, dtype=torch.float32)
            if init_method == "kaiming":
                bound = 1.0 / math.sqrt(fan_in)
                values.uniform_(-bound, bound, generator=gen)
            elif init_method == "xavier":
                values.normal_(0.0, math.sqrt(2.0 / (fan_in + rank)), generator=gen)
            elif init_method == "normal":
                values.normal_(0.0, 0.2, generator=gen)
            elif init_method == "zero":
                values.zero_()
            else:
                raise ValueError(f"unsupported LoRA init method {init_method!r}")
            param.copy_(values.to(device=param.device, dtype=param.dtype))
        else:
            continue
        count += 1
    for module in multi_lora_modules(model):
        for idx in range(1, module.n_adapters):
            for p in module.adapters[idx].parameters():
                p.zero_()
    return count


def single_slot_counts(model, num_tokens: int) -> torch.Tensor:
    """tokens_per_adapter for a forward that uses only the live slot."""
    n = next(multi_lora_modules(model)).n_adapters
    counts = torch.zeros(n, dtype=torch.int64)
    counts[0] = num_tokens
    return counts


def install_default_routing(model) -> int:
    """Route every forward to slot 0 unless a caller set per-slot counts.

    A forward pre-hook on each model chunk sets tokens_per_adapter from the
    input size, so code paths that know nothing about multi-LoRA (logprob
    passes, checkpoint checks) keep working. A batched pass sets explicit counts
    with ``route_slots`` instead. Returns the number of hooked chunks.
    """
    from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

    hooked = 0
    for chunk in _chunks(model):
        if not any(True for _ in multi_lora_modules(chunk)):
            continue

        def hook(module, args, kwargs, _chunk=chunk):
            if getattr(_chunk, "_multi_lora_explicit_routing", False):
                return None
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is None:
                raise RuntimeError("multi-LoRA default routing needs input_ids to size the batch")
            set_tokens_per_adapter_slot(_chunk, single_slot_counts(_chunk, input_ids.numel()))
            return None

        chunk.register_forward_pre_hook(hook, with_kwargs=True)
        hooked += 1
    return hooked


@contextmanager
def expose_live_slot(model):
    """Present slot 0 as the single ``.adapter`` (for Megatron-Bridge exports)."""
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    if not any(True for _ in multi_lora_modules(model)):
        yield
        return
    with expose_adapter_slot(model, 0):
        yield


# ----------------------------------------------------------------------------
# Batched passes: lend scratch slots to other tenants for one forward/backward.
# ----------------------------------------------------------------------------


def slot_parameter_groups(model) -> List[List[torch.nn.Parameter]]:
    """One list per adapter tensor: [slot 0 param, slot 1 param, ...]."""
    groups = []
    for module in multi_lora_modules(model):
        for name, live in module.adapters[0].named_parameters():
            groups.append([live] + [dict(module.adapters[i].named_parameters())[name] for i in range(1, module.n_adapters)])
    return groups


def buffer_offsets(model_chunks) -> dict:
    """Live (slot 0) param -> (model chunk, buffer, start, end) in the DDP buffers.

    AdapterStore's per-tenant CPU copies (cpu_param_data / cpu_grad_data) mirror
    these buffers element for element, so the offsets locate a tenant's tensors.
    """
    from skyrl.backends.skyrl_train.workers.megatron.adapter_store import _iter_buffers

    offsets = {}
    for mc_idx, buf_idx, buf in _iter_buffers(model_chunks):
        for param, (start, end, _bucket) in buf.param_index_map.items():
            offsets[param] = (mc_idx, buf_idx, start, end)
    return offsets


@torch.no_grad()
def lend_slots(model_chunks, store, tenants: List[str]) -> None:
    """Load tenants[1:] (not live) into scratch slots 1.. and make them trainable."""
    offsets = buffer_offsets(model_chunks)
    for group in slot_parameter_groups([m.module if hasattr(m, "module") else m for m in model_chunks]):
        mc_idx, buf_idx, start, end = offsets[group[0]]
        for slot, tenant in enumerate(tenants[1:], start=1):
            src = store._slots[tenant].cpu_param_data[mc_idx][buf_idx][start:end]
            group[slot].copy_(src.view_as(group[slot]), non_blocking=True)
            group[slot].grad = None
            group[slot].requires_grad_(True)
    torch.cuda.current_stream().synchronize()


@torch.no_grad()
def return_slots(model_chunks, store, tenants: List[str], keep_grads: bool) -> None:
    """Add scratch-slot gradients to their tenants' stored gradients, then clear the slots.

    The live slot's gradient goes into its DDP grad buffer as main_grad += grad;
    a tenant's stored gradient gets the same fp32 add, on the GPU, so a batched
    pass accumulates exactly as that tenant's own pass would.
    """
    unwrapped = [m.module if hasattr(m, "module") else m for m in model_chunks]
    groups = slot_parameter_groups(unwrapped)
    if keep_grads:
        offsets = buffer_offsets(model_chunks)
        for slot, tenant in enumerate(tenants[1:], start=1):
            stored = store._slots[tenant].cpu_grad_data
            live = {}
            for group in groups:
                param = group[slot]
                if param.grad is None:
                    continue
                mc_idx, buf_idx, start, end = offsets[group[0]]
                key = (mc_idx, buf_idx)
                if key not in live:
                    live[key] = stored[mc_idx][buf_idx].to(param.device, non_blocking=True)
                live[key][start:end] += param.grad.reshape(-1).to(live[key].dtype)
            for (mc_idx, buf_idx), tensor in live.items():
                stored[mc_idx][buf_idx].copy_(tensor, non_blocking=True)
    for group in groups:
        for param in group[1:]:
            param.grad = None
            param.requires_grad_(False)
            param.zero_()
    torch.cuda.current_stream().synchronize()


def route_packed_microbatch(model, batch: dict, packed_seq_params, input_ids, sub_seq_lengths) -> None:
    """Set per-slot token counts for a packed micro-batch carrying ``lora_slot``.

    Rows are sorted by slot (the caller orders tenants), and the packed layout
    keeps row order, so each slot's tokens form one contiguous span. Trailing
    pack padding goes to the last row's slot.
    """
    slots = batch.get("lora_slot")
    if slots is None:
        return
    from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

    if packed_seq_params is None:
        raise RuntimeError("batched multi-LoRA passes need packed micro-batches (remove_microbatch_padding)")
    cu = packed_seq_params.cu_seqlens_q_padded.tolist()
    entries = [b - a for a, b in zip(cu[:-1], cu[1:])]
    if sub_seq_lengths is not None:  # several sub-sequences per row
        per_row, i = [], 0
        for row in sub_seq_lengths:
            per_row.append(sum(entries[i : i + len(row)]))
            i += len(row)
        entries = per_row
    slots = [int(s) for s in slots.tolist()]
    if len(entries) != len(slots):
        raise RuntimeError(f"{len(slots)} lora_slot entries for {len(entries)} packed rows")
    if any(b < a for a, b in zip(slots, slots[1:])):
        raise RuntimeError(f"micro-batch rows are not sorted by LoRA slot: {slots}")
    chunk = unwrap_model_chunk(model)
    counts = single_slot_counts(chunk, 0)
    for slot, n in zip(slots, entries):
        counts[slot] += n
    counts[slots[-1]] += input_ids.numel() - int(counts.sum())
    chunk._multi_lora_explicit_routing = True
    set_tokens_per_adapter_slot(chunk, counts)


def unwrap_model_chunk(model):
    from megatron.core.utils import unwrap_model

    return unwrap_model(model)


def clear_explicit_routing(model_chunks) -> None:
    for chunk in model_chunks:
        unwrap_model_chunk(chunk)._multi_lora_explicit_routing = False
