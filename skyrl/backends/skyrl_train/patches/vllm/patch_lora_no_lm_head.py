"""Keep vLLM from wrapping ``lm_head`` with LoRA.

vLLM wraps every module a model lists in ``embedding_modules`` -- for Qwen3 that
includes ``lm_head`` -- whether or not any adapter targets it. The wrapper,
``LogitsProcessorWithLoRA``, runs ``add_lora_logits`` on every logits call,
indexed by the punica *prompt* mapping, which has one row per sampled token.

Prompt logprobs break that assumption: the runner computes logits for every
prompt token of a request, so ``add_lora_logits`` sees more rows than the
mapping describes and the LoRA kernels read stale indices left by an earlier
batch. With LoRA adapters active, a batch that mixes a prompt-logprobs request
(``prompt_logprobs`` set, e.g. Tinker's ``compute_logprobs``) with ordinary
generation requests ends in ``CUDA error: an illegal memory access``, which kills
the engine. Model Runner V2 (the 0.30 default) hits it within seconds; V1 needs
more adapters than ``max_loras`` but hits it too.

SkyRL's trainers never put LoRA on the output layer, so the ``lm_head`` wrapper
only ever adds zeros. Dropping ``lm_head`` from the supported modules removes the
wrapper and with it the broken path. An adapter that does carry ``lm_head``
weights is then rejected at load time ("unexpected modules") instead of being
silently ignored.

Applied from ``new_inference_worker_wrap`` so it runs in every worker before the
LoRA manager is created.

TODO: remove once vLLM sizes the LoRA logits mapping for prompt logprobs.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_lora_no_lm_head_patch() -> None:
    """Exclude ``lm_head`` from vLLM's LoRA-capable modules (idempotent, process-local)."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import vllm.lora.model_manager as model_manager
    except (ModuleNotFoundError, ImportError) as e:
        logger.info(f"Skipping LoRA lm_head patch: {e}")
        return

    original = model_manager.get_supported_lora_modules

    def get_supported_lora_modules(model) -> list[str]:
        return [name for name in original(model) if name != "lm_head"]

    model_manager.get_supported_lora_modules = get_supported_lora_modules
    _PATCHED = True
    logger.info("Patched vLLM LoRA: lm_head is not wrapped (prompt-logprobs crash with LoRA logits)")
