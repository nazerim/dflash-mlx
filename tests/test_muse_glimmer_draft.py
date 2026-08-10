# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file

from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.engine.config import resolve_draft_window
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs
from dflash_mlx.models.muse_glimmer_draft import (
    MuseGlimmerDraftModel,
    MuseGlimmerDraftModelArgs,
    is_muse_glimmer_draft_config,
)
from dflash_mlx.runtime.loading import _get_dflash_model_classes

# The real assistant checkpoint's config layout: dflash keys at the ROOT
# (no dflash_config block), rope_theta nested, vocab_size /
# num_target_layers / tie_word_embeddings absent.
REAL_ASSISTANT_CONFIG = {
    "architectures": ["MuseGlimmerAssistantModel"],
    "attention_dropout": 0,
    "block_size": 16,
    "bos_token_id": 200000,
    "dtype": "bfloat16",
    "eos_token_id": 200001,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "layer_types": ["sliding_attention"] * 5,
    "mask_token_id": 201818,
    "max_position_embeddings": 131072,
    "model_type": "muse_glimmer_assistant",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "pad_token_id": 200018,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "sliding_window": 2048,
    "target_layer_ids": [1, 13, 25, 37, 49],
}


def _tiny_params(**overrides):
    params = {
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "layer_types": ["sliding_attention", "sliding_attention"],
        "sliding_window": 16,
        "block_size": 4,
        "target_layer_ids": [1, 5],
        "mask_token_id": 99,
    }
    params.update(overrides)
    return params


class TestFromDict:
    def test_real_config_normalizes(self):
        args = MuseGlimmerDraftModelArgs.from_dict(dict(REAL_ASSISTANT_CONFIG))
        assert args.dflash_config["target_layer_ids"] == [1, 13, 25, 37, 49]
        assert args.dflash_config["mask_token_id"] == 201818
        assert args.rope_theta == 500000.0
        assert args.block_size == 16
        assert args.sliding_window == 2048
        assert args.layer_types == ("sliding_attention",) * 5
        # Absent required base fields default to inert values.
        assert args.vocab_size == 0
        assert args.num_target_layers == 0
        assert args.tie_word_embeddings is False

    def test_model_picks_up_root_keys(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        assert model.target_layer_ids == [1, 5]
        assert model.mask_token_id == 99
        assert model.block_size == 4
        assert model.model_type == "dflash_muse_glimmer"

    def test_missing_target_layer_ids_raises(self):
        params = _tiny_params()
        del params["target_layer_ids"]
        with pytest.raises(ValueError, match="target_layer_ids"):
            MuseGlimmerDraftModelArgs.from_dict(params)

    def test_missing_mask_token_id_raises(self):
        params = _tiny_params()
        del params["mask_token_id"]
        with pytest.raises(ValueError, match="mask_token_id"):
            MuseGlimmerDraftModelArgs.from_dict(params)

    def test_existing_dflash_config_wins(self):
        params = _tiny_params()
        params["dflash_config"] = {"target_layer_ids": [0, 1], "mask_token_id": 7}
        args = MuseGlimmerDraftModelArgs.from_dict(params)
        assert args.dflash_config["target_layer_ids"] == [0, 1]
        assert args.dflash_config["mask_token_id"] == 7


class TestDispatch:
    def test_muse_config_dispatches_subclass(self):
        model_cls, args_cls = _get_dflash_model_classes(
            {"model_type": "muse_glimmer_assistant"}
        )
        assert model_cls is MuseGlimmerDraftModel
        assert args_cls is MuseGlimmerDraftModelArgs

    def test_other_config_keeps_base(self):
        model_cls, args_cls = _get_dflash_model_classes({"model_type": "qwen3"})
        assert model_cls is DFlashDraftModel
        assert args_cls is DFlashDraftModelArgs

    def test_is_muse_glimmer_draft_config(self):
        assert is_muse_glimmer_draft_config({"model_type": "muse_glimmer_assistant"})
        assert not is_muse_glimmer_draft_config({"model_type": "gemma4_assistant"})
        assert not is_muse_glimmer_draft_config(None)


class TestDraftWindow:
    def test_all_sliding_draft_uses_config_window(self):
        args = MuseGlimmerDraftModelArgs.from_dict(
            dict(REAL_ASSISTANT_CONFIG)
        )
        model = MuseGlimmerDraftModel(args)
        runtime_config = SimpleNamespace(draft_sink_size=64, draft_window_size=1024)
        sink, window = resolve_draft_window(runtime_config, model)
        assert sink == 64
        assert window == 2048


class TestSanitize:
    def test_encoder_keys_remap(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        weights = {
            "encoder.fc.weight": mx.zeros((32, 64)),
            "encoder.output_norm_enc.weight": mx.zeros((32,)),
            "layers.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
            "norm.weight": mx.zeros((32,)),
        }
        sanitized = model.sanitize(weights)
        assert "fc.weight" in sanitized
        assert "hidden_norm.weight" in sanitized
        assert "layers.0.self_attn.q_proj.weight" in sanitized
        assert "norm.weight" in sanitized
        assert not any(key.startswith("encoder.") for key in sanitized)

    def test_quantized_suffixes_survive(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        weights = {
            "encoder.fc.weight": mx.zeros((1,)),
            "encoder.fc.scales": mx.zeros((1,)),
            "encoder.fc.biases": mx.zeros((1,)),
        }
        sanitized = model.sanitize(weights)
        assert {"fc.weight", "fc.scales", "fc.biases"} == set(sanitized)

    def test_unknown_encoder_key_raises(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        with pytest.raises(ValueError, match="encoder"):
            model.sanitize({"encoder.mystery.weight": mx.zeros((1,))})


class TestBindValidation:
    def _target_ops(self, family="muse_glimmer_swa", num_layers=8, hidden=32):
        text_model = SimpleNamespace(
            layers=[object()] * num_layers,
            args=SimpleNamespace(hidden_size=hidden),
        )
        return SimpleNamespace(
            family=lambda model: family,
            text_model=lambda model: text_model,
        )

    def test_bind_accepts_matching_target(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        model.bind_target_model(object(), target_ops=self._target_ops())
        assert model.embed_scale == 1.0

    def test_bind_rejects_wrong_family(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        with pytest.raises(ValueError, match="family"):
            model.bind_target_model(
                object(), target_ops=self._target_ops(family="gemma4_swa")
            )

    def test_bind_rejects_layer_id_overflow(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        with pytest.raises(ValueError, match="target_layer_ids"):
            model.bind_target_model(
                object(), target_ops=self._target_ops(num_layers=4)
            )

    def test_bind_rejects_hidden_mismatch(self):
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        with pytest.raises(ValueError, match="hidden"):
            model.bind_target_model(
                object(), target_ops=self._target_ops(hidden=64)
            )


class TestProjection:
    def test_projection_is_norm_after_fc(self):
        mx.random.seed(0)
        model = MuseGlimmerDraftModel(
            MuseGlimmerDraftModelArgs.from_dict(_tiny_params())
        )
        x = mx.random.normal((1, 3, 2 * 32))
        expected = model.hidden_norm(model.fc(x))
        actual = model.project_target_hidden(x)
        assert bool(mx.allclose(actual, expected))
