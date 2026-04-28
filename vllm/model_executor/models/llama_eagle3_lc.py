# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM


class Eagle3LCLlamaForCausalLM(Eagle3LlamaForCausalLM):
    """Eagle3-LC draft model for long-context speculative decoding.

    Inherits the full Eagle3 architecture from ``Eagle3LlamaForCausalLM``.
    RoPE behaviour is configured at the model-config level via ``rope_method``:

    - "full"         — standard RoPE; inherits the verifier's native scaling
                       (Llama3 for Llama-3.1-8B).
    - "yarn"         — ``rope_scaling`` with ``rope_type: "yarn"`` in config;
                       static YaRN applied at all context lengths.
    - "dynamic_yarn" — same vLLM behaviour as "full" (verifier's native scaling);
                       the speculators training-time ``DynamicYaRNRotaryEmbedding``
                       uses the verifier's base RoPE for short sequences and YaRN
                       for longer ones, but since ``total_seq_len`` equals
                       ``original_max_position_embeddings`` the YaRN branch is
                       never triggered during training.  vLLM therefore applies
                       the same native scaling at inference, keeping train and
                       inference aligned.
    - "llama3"       — ``rope_scaling`` with ``rope_type: "llama3"`` in config.
    - "partial"      — ``partial_rotary_factor`` in config (Qwen3 style); vLLM's
                       ``get_rope()`` applies rotation to only the specified
                       fraction of head dimensions, leaving the rest unrotated.

    The speculators ``algos.py`` updater for ``eagle3_lc`` translates the
    ``rope_method`` field from the speculators config into the appropriate
    ``rope_scaling`` or ``partial_rotary_factor`` entry in the vLLM config,
    which ``patch_rope_parameters`` then converts to ``config.rope_parameters``
    consumed by ``get_rope()``.
    """
