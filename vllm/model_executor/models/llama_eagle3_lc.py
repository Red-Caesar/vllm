# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM


class Eagle3LCLlamaForCausalLM(Eagle3LlamaForCausalLM):
    """Eagle3-LC draft model for long-context speculative decoding.

    Inherits the full Eagle3 architecture from ``Eagle3LlamaForCausalLM``.
    RoPE behaviour (full, YaRN, Llama-3.1, or partial) is configured at the
    model-config level:

    - "full"    — standard RoPE, no changes to config needed
    - "yarn"    — ``rope_scaling`` with ``rope_type: "yarn"`` in config
    - "llama3"  — ``rope_scaling`` with ``rope_type: "llama3"`` in config
    - "partial" — ``partial_rotary_factor`` in config (Qwen3 style); vLLM's
                      ``get_rope()`` applies rotation to only the specified fraction
                      of head dimensions, leaving the rest unrotated

    The speculators ``algos.py`` updater for ``eagle3_lc`` translates the
    ``rope_method`` field from the speculators config into the appropriate
    ``rope_scaling`` or ``partial_rotary_factor`` entry in the vLLM config,
    which ``patch_rope_parameters`` then converts to ``config.rope_parameters``
    consumed by ``get_rope()``.
    """
