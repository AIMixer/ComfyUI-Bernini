import json

import torch

from engine.native_quant import (
    has_native_quant_metadata,
    is_native_quant_layer,
    logical_linear_shape,
    read_layer_quant_conf,
)


def _make_int8_layer_sd(prefix="blocks.0.self_attn.q."):
    meta = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }
    meta_tensor = torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)
    return {
        prefix + "weight": torch.zeros(64, 32, dtype=torch.int8),
        prefix + "weight_scale": torch.tensor(0.01, dtype=torch.float32),
        prefix + "comfy_quant": meta_tensor,
    }


def test_has_native_quant_metadata_int8():
    sd = _make_int8_layer_sd()
    assert has_native_quant_metadata(sd)
    assert is_native_quant_layer(sd, "blocks.0.self_attn.q.")
    conf = read_layer_quant_conf(sd, "blocks.0.self_attn.q.")
    assert conf["format"] == "int8_tensorwise"
    assert conf["convrot"] is True
    assert conf["convrot_groupsize"] == 256


def test_int8_weight_shape_unchanged():
    sd = _make_int8_layer_sd()
    assert logical_linear_shape(sd, "blocks.0.self_attn.q.weight") == (64, 32)


def test_int8_scale_weight_alias():
    sd = _make_int8_layer_sd()
    prefix = "blocks.0.self_attn.q."
    sd[prefix + "scale_weight"] = sd.pop(prefix + "weight_scale")
    from engine.native_quant import build_quantized_parameter

    param, runtime = build_quantized_parameter(sd, prefix, torch.device("cpu"), torch.bfloat16)
    assert runtime["quant_format"] == "int8_tensorwise"
    assert param.shape == (64, 32)


def test_non_quant_state_dict():
    sd = {"blocks.0.self_attn.q.weight": torch.randn(64, 32)}
    assert not has_native_quant_metadata(sd)
