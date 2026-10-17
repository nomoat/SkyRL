"""CPU tests for the patch that keeps vLLM from wrapping ``lm_head`` with LoRA.

A fake ``vllm.lora.model_manager`` stands in for vLLM, so no install or GPU is needed.
"""

import sys
import types

import pytest

from skyrl.backends.skyrl_train.patches.vllm import patch_lora_no_lm_head as patch


@pytest.fixture
def fake_model_manager(monkeypatch):
    module = types.ModuleType("vllm.lora.model_manager")
    module.get_supported_lora_modules = lambda model: ["qkv_proj", "lm_head", "embed_tokens"]
    vllm, lora = types.ModuleType("vllm"), types.ModuleType("vllm.lora")
    vllm.lora, lora.model_manager = lora, module
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.model_manager", module)
    monkeypatch.setattr(patch, "_PATCHED", False)
    return module


def test_drops_only_lm_head(fake_model_manager):
    patch.apply_lora_no_lm_head_patch()
    assert fake_model_manager.get_supported_lora_modules(object()) == ["qkv_proj", "embed_tokens"]


def test_apply_is_idempotent(fake_model_manager):
    patch.apply_lora_no_lm_head_patch()
    wrapped = fake_model_manager.get_supported_lora_modules
    patch.apply_lora_no_lm_head_patch()
    assert fake_model_manager.get_supported_lora_modules is wrapped
