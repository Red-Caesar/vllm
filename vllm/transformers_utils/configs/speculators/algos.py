# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, vllm_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - eagle_aux_hidden_state_layer_ids: List of layer indices from the base
        model to use as auxiliary inputs for the Eagle3 drafter. These layers
        provide intermediate hidden states that help the drafter make better
        predictions. This is the standard field used in Eagle3 checkpoints.
    """

    vllm_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        vllm_config["target_hidden_size"] = config_dict["target_hidden_size"]
    vllm_config["norm_before_residual"] = config_dict.get("norm_before_residual", True)
    vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        vllm_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("eagle3_lc")
def update_eagle3_lc(config_dict: dict, vllm_config: dict) -> None:
    """Apply Eagle-3-LC specific configuration transformations.

    Extends Eagle-3 with configurable RoPE scaling controlled by the
    ``rope_method`` field in the speculators config:

    - "full"         — no RoPE changes (identical to eagle3); the verifier's
                       native ``rope_scaling`` is inherited unchanged.
    - "yarn"         — sets ``rope_scaling`` from ``rope_scaling_config``
                       (static YaRN applied at all context lengths).
    - "dynamic_yarn" — no ``rope_scaling`` override; the verifier's native
                       ``rope_scaling`` is inherited unchanged, identical to
                       "full".  vLLM has no dynamic-switching RoPE, so Llama3
                       frequencies are used for all positions.  This matches
                       training behaviour: the speculators
                       ``DynamicYaRNRotaryEmbedding`` uses the verifier's base
                       RoPE for sequences up to ``original_max_position_embeddings``
                       (which equals ``total_seq_len`` in practice, so YaRN is
                       never triggered during training).
    - "llama3"       — sets ``rope_scaling`` from ``rope_scaling_config``
                       (explicit Llama-3.1 scaling parameters).
    - "partial"      — sets ``partial_rotary_factor`` from ``rope_partial_factor``
                       (Qwen3-style partial rotation).

    For yarn/llama3 the ``rope_scaling_config`` dict is written to
    ``vllm_config["rope_scaling"]`` so that ``patch_rope_parameters`` in
    ``vllm.transformers_utils.config`` converts it to ``rope_parameters``
    for ``get_rope()``.

    For partial the ``rope_partial_factor`` float is written to
    ``vllm_config["partial_rotary_factor"]`` so that ``patch_rope_parameters``
    includes it in ``rope_parameters`` for ``get_rope()``.
    """
    vllm_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        vllm_config["target_hidden_size"] = config_dict["target_hidden_size"]
    vllm_config["norm_before_residual"] = config_dict.get("norm_before_residual", True)
    vllm_config["architectures"] = ["Eagle3LCLlamaForCausalLM"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        vllm_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]

    rope_method = config_dict.get("rope_method", "full")
    if rope_method in ("yarn", "llama3"):
        rope_scaling_config = config_dict.get("rope_scaling_config")
        if rope_scaling_config is not None:
            vllm_config["rope_scaling"] = rope_scaling_config
    elif rope_method == "partial":
        rope_partial_factor = config_dict.get("rope_partial_factor", 0.25)
        vllm_config["partial_rotary_factor"] = rope_partial_factor
    # "full" and "dynamic_yarn" leave rope_scaling unchanged, inheriting the
    # verifier's native scaling (Llama3 for Llama-3.1-8B).
