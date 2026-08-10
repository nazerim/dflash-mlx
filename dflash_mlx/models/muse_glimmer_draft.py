# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Muse Glimmer DFlash drafter (MuseGlimmerAssistantModel checkpoints).

Meta's assistant checkpoint fits the stock ``DFlashDraftModel`` geometry
(per-head q/k RMSNorm, SwiGLU MLP, all-sliding window 2048, drafter-own
KV heads), but its config diverges from the layout the base classes
expect:

- ``target_layer_ids`` / ``mask_token_id`` / ``block_size`` sit at the
  config ROOT — there is no ``dflash_config`` block. The base class
  would silently fall back to ``build_target_layer_ids()`` and
  ``mask_token_id=0`` (acceptance collapses with no error), so both keys
  are hard-required here.
- ``rope_theta`` is nested under ``rope_parameters``.
- ``vocab_size`` / ``num_target_layers`` / ``tie_word_embeddings`` are
  absent; they are required positional fields on the base dataclass but
  unused once ``target_layer_ids`` is explicit, so they default to
  inert values.
- Weights name the target-hidden projection ``encoder.fc`` /
  ``encoder.output_norm_enc`` (native names: ``fc`` / ``hidden_norm``).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs

MUSE_GLIMMER_DRAFT_MODEL_TYPE = "muse_glimmer_assistant"

_ENCODER_KEY_REMAP = (
    ("encoder.fc.", "fc."),
    ("encoder.output_norm_enc.", "hidden_norm."),
)


def is_muse_glimmer_draft_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return (
        str(config.get("model_type", "")).lower() == MUSE_GLIMMER_DRAFT_MODEL_TYPE
    )


def _normalized_draft_params(params: dict[str, Any]) -> dict[str, Any]:
    data = dict(params)

    rope_parameters = data.get("rope_parameters")
    if "rope_theta" not in data and isinstance(rope_parameters, dict):
        theta = rope_parameters.get("rope_theta")
        if theta is not None:
            data["rope_theta"] = float(theta)

    dflash_config = dict(data.get("dflash_config") or {})
    for key in ("target_layer_ids", "mask_token_id", "block_size"):
        if key not in dflash_config and key in data:
            dflash_config[key] = data[key]
    for key in ("target_layer_ids", "mask_token_id"):
        if dflash_config.get(key) is None:
            # The base class falls back to build_target_layer_ids() /
            # mask_token_id=0 for these — a silent acceptance collapse,
            # not an error. Refuse to load instead.
            raise ValueError(
                f"Muse Glimmer draft config is missing '{key}'; refusing the "
                "silent DFlash default"
            )
    data["dflash_config"] = dflash_config

    data.setdefault("vocab_size", 0)
    data.setdefault("num_target_layers", 0)
    data.setdefault("tie_word_embeddings", False)
    return data


class MuseGlimmerDraftModelArgs(DFlashDraftModelArgs):
    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "MuseGlimmerDraftModelArgs":
        data = _normalized_draft_params(params)
        # The base from_dict filters on cls.__annotations__, which only
        # holds a subclass's OWN annotations; merge the MRO instead.
        annotations: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))
        return cls(
            **{key: value for key, value in data.items() if key in annotations}
        )


class MuseGlimmerDraftModel(DFlashDraftModel):
    def __init__(self, args: MuseGlimmerDraftModelArgs):
        super().__init__(args)
        self.model_type = "dflash_muse_glimmer"

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        family = target_ops.family(target_model)
        if family != "muse_glimmer_swa":
            raise ValueError(
                "Muse Glimmer draft requires a muse_glimmer target, got "
                f"family={family!r}"
            )
        text_model = target_ops.text_model(target_model)
        num_layers = len(text_model.layers)
        if max(self.target_layer_ids) >= num_layers:
            raise ValueError(
                f"target_layer_ids {self.target_layer_ids} exceed the target's "
                f"{num_layers} layers"
            )
        target_hidden = int(getattr(text_model.args, "hidden_size", 0) or 0)
        if target_hidden != int(self.args.hidden_size):
            raise ValueError(
                "Muse Glimmer draft hidden size must match the target: "
                f"draft={self.args.hidden_size}, target={target_hidden}"
            )
        super().bind_target_model(target_model, target_ops=target_ops)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            for prefix, replacement in _ENCODER_KEY_REMAP:
                if key.startswith(prefix):
                    key = replacement + key[len(prefix) :]
                    break
            if key.startswith("encoder."):
                raise ValueError(
                    f"Unrecognized Muse Glimmer draft encoder weight: {key}"
                )
            sanitized[key] = value
        return sanitized
