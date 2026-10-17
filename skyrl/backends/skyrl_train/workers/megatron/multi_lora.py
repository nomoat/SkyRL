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
