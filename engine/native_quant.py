"""Bridge Bernini WanVideo weight loading to ComfyUI native quantized checkpoints.

ComfyUI stores per-layer ``*.comfy_quant`` JSON plus ``weight`` / ``weight_scale`` keys.
Loading logic follows ``comfy.ops._load_quantized_module`` and ``comfy.quant_ops``.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

try:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
except ImportError as e:
    QUANT_ALGOS = {}
    QuantizedTensor = None
    get_layout_class = None
    _IMPORT_ERROR = str(e)
else:
    _IMPORT_ERROR = None

try:
    from comfy.utils import detect_layer_quantization
except ImportError:
    def detect_layer_quantization(state_dict, prefix=""):  # type: ignore
        for key in state_dict:
            if key.startswith(prefix) and key.endswith(".comfy_quant"):
                return {"mixed_ops": True}
        return None


def require_native_quant_backend():
    if QuantizedTensor is None or not QUANT_ALGOS:
        detail = _IMPORT_ERROR or "comfy.quant_ops unavailable"
        raise RuntimeError(
            "This checkpoint uses ComfyUI native quantization (.comfy_quant), but the "
            f"running ComfyUI build does not expose quant_ops ({detail}). "
            "Update ComfyUI to a build with comfy_kitchen / int8_tensorwise support."
        )


def has_native_quant_metadata(state_dict) -> bool:
    if state_dict is None:
        return False
    if detect_layer_quantization(state_dict, "") is not None:
        return True
    return any(key.endswith(".comfy_quant") for key in state_dict)


def _normalize_prefix(prefix: str) -> str:
    if prefix.endswith("weight"):
        prefix = prefix[: -len("weight")]
    if prefix and not prefix.endswith("."):
        prefix += "."
    return prefix


def _prefix_candidates(prefix: str) -> list[str]:
    prefix = _normalize_prefix(prefix)
    out: list[str] = []

    def add(candidate: str):
        candidate = _normalize_prefix(candidate)
        if candidate not in out:
            out.append(candidate)

    add(prefix)
    for wrapper in ("diffusion_model.", "model.diffusion_model.", "model."):
        if prefix.startswith(wrapper):
            add(prefix[len(wrapper) :])
        else:
            add(wrapper + prefix)
    return out


def resolve_layer_prefix(state_dict, prefix: str) -> str | None:
    for candidate in _prefix_candidates(prefix):
        if f"{candidate}comfy_quant" in state_dict:
            return candidate
    return None


def is_native_quant_layer(state_dict, prefix: str) -> bool:
    return resolve_layer_prefix(state_dict, prefix) is not None


def is_native_quant_weight_key(state_dict, weight_key: str) -> bool:
    if not (weight_key == "weight" or weight_key.endswith(".weight")):
        return False
    layer_prefix = weight_key[: -len("weight")]
    return is_native_quant_layer(state_dict, layer_prefix)


def read_layer_quant_conf(state_dict, layer_prefix: str) -> dict | None:
    matched = resolve_layer_prefix(state_dict, layer_prefix)
    if matched is None:
        return None
    raw = state_dict.get(f"{matched}comfy_quant")
    if raw is None:
        return None
    return json.loads(raw.numpy().tobytes())


def logical_linear_shape(state_dict, weight_key: str) -> tuple[int, int]:
    shape = tuple(state_dict[weight_key].shape)
    layer_prefix = weight_key[: -len("weight")]
    conf = read_layer_quant_conf(state_dict, layer_prefix)
    if conf and conf.get("format") == "nvfp4":
        return shape[0], shape[1] * 2
    return shape


def _int8_layout_scales(layer_conf: dict, state_dict, layer_prefix: str, device) -> dict:
    scale_key = f"{layer_prefix}weight_scale"
    scale = state_dict.get(scale_key)
    if scale is None:
        scale = state_dict.get(f"{layer_prefix}scale_weight")
    if scale is None:
        raise ValueError(f"Missing INT8 weight scale for layer {layer_prefix.rstrip('.')}")
    scales = {"scale": scale.to(device=device)}
    params_conf = layer_conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    if layer_conf.get("convrot", params_conf.get("convrot", False)):
        scales["convrot"] = True
        scales["convrot_groupsize"] = int(
            layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
        )
    return scales


def _pop_scale(state_dict, layer_prefix: str, name: str, device, dtype=None):
    key = f"{layer_prefix}{name}"
    value = state_dict.get(key)
    if value is None and name == "weight_scale":
        value = state_dict.get(f"{layer_prefix}scale_weight")
    if value is None:
        return None
    value = value.to(device=device)
    if dtype is not None:
        value = value.view(dtype=dtype)
    return value


def build_quantized_parameter(
    state_dict,
    layer_prefix: str,
    device,
    compute_dtype,
) -> tuple[torch.nn.Parameter, dict]:
    """Build one QuantizedTensor Parameter — mirrors comfy.ops._load_quantized_module."""
    require_native_quant_backend()

    matched = resolve_layer_prefix(state_dict, layer_prefix)
    if matched is None:
        raise ValueError(f"No native quant metadata for {layer_prefix}")

    weight = state_dict[f"{matched}weight"]
    layer_conf = read_layer_quant_conf(state_dict, matched)
    if layer_conf is None:
        raise ValueError(f"Missing comfy_quant entry for {matched}")

    quant_format = layer_conf.get("format")
    if quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unsupported native quant format: {quant_format}")

    qconfig = QUANT_ALGOS[quant_format]
    layout_type = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_type)
    out_features, in_features = logical_linear_shape(state_dict, f"{matched}weight")

    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scales = {"scale": _pop_scale(state_dict, matched, "weight_scale", device)}
    elif quant_format == "mxfp8":
        block_scale = _pop_scale(state_dict, matched, "weight_scale", device, torch.float8_e8m0fnu)
        if block_scale is None:
            raise ValueError(f"Missing MXFP8 block scales for {matched}")
        scales = {"scale": block_scale}
    elif quant_format == "nvfp4":
        tensor_scale = _pop_scale(state_dict, matched, "weight_scale_2", device)
        block_scale = _pop_scale(state_dict, matched, "weight_scale", device, torch.float8_e4m3fn)
        if tensor_scale is None or block_scale is None:
            raise ValueError(f"Missing NVFP4 scales for {matched}")
        scales = {"scale": tensor_scale, "block_scale": block_scale}
    elif quant_format == "int8_tensorwise":
        scales = _int8_layout_scales(layer_conf, state_dict, matched, device)
    else:
        raise ValueError(f"Unsupported native quant format: {quant_format}")

    params = layout_cls.Params(
        **scales,
        orig_dtype=compute_dtype,
        orig_shape=(out_features, in_features),
    )
    qt = QuantizedTensor(
        weight.to(device=device, dtype=qconfig["storage_t"]),
        layout_type,
        params,
    )
    runtime = {
        "quant_format": quant_format,
        "layout_type": layout_type,
        "orig_shape": (out_features, in_features),
    }
    return torch.nn.Parameter(qt, requires_grad=False), runtime


def bind_native_quant_runtime(module: nn.Module, runtime: dict):
    module.quant_format = runtime["quant_format"]
    module.layout_type = runtime["layout_type"]
    module._orig_shape = tuple(runtime["orig_shape"])
    module._full_precision_mm = bool(runtime.get("full_precision_mm", False))


def bind_native_quant_runtime_from_state_dict(module: nn.Module, state_dict, layer_prefix: str, compute_dtype, device):
    matched = resolve_layer_prefix(state_dict, layer_prefix)
    if matched is None:
        return False
    conf = read_layer_quant_conf(state_dict, matched)
    if conf is None:
        return False
    quant_format = conf.get("format")
    if quant_format not in QUANT_ALGOS:
        return False
    out_features, in_features = logical_linear_shape(state_dict, f"{matched}weight")
    bind_native_quant_runtime(
        module,
        {
            "quant_format": quant_format,
            "layout_type": QUANT_ALGOS[quant_format]["comfy_tensor_layout"],
            "orig_shape": (out_features, in_features),
            "full_precision_mm": conf.get("full_precision_matrix_mult", False),
        },
    )
    module.out_features, module.in_features = out_features, in_features
    return True


def _weight_is_materialized(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.device.type == "meta":
        return False
    return isinstance(weight, QuantizedTensor) or torch.is_floating_point(weight)


def materialize_native_quant_linears(
    model: nn.Module,
    state_dict,
    compute_dtype,
    load_device,
    prefix: str = "",
    device_resolver=None,
) -> int:
    """Assign QuantizedTensor weights to nn.Linear layers that carry .comfy_quant metadata."""
    if state_dict is None:
        return 0
    if prefix == "":
        require_native_quant_backend()

    materialized = 0
    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        materialized += materialize_native_quant_linears(
            module,
            state_dict,
            compute_dtype,
            load_device,
            child_prefix,
            device_resolver,
        )

        if not isinstance(module, nn.Linear) or "loras" in child_prefix:
            continue
        if not is_native_quant_layer(state_dict, child_prefix):
            continue
        if _weight_is_materialized(module):
            target = device_resolver(child_prefix) if device_resolver else None
            if target is not None and getattr(module.weight, "device", None) != target:
                module.to(target)
            continue

        target_device = device_resolver(child_prefix) if device_resolver else load_device
        if compute_dtype is None or target_device is None:
            raise RuntimeError("Native quant materialization requires compute dtype and device")

        param, runtime = build_quantized_parameter(state_dict, child_prefix, target_device, compute_dtype)
        module.weight = param
        bind_native_quant_runtime(module, runtime)
        module.out_features, module.in_features = runtime["orig_shape"]

        bias_key = f"{child_prefix}bias"
        if module.bias is not None and bias_key in state_dict:
            module.bias = nn.Parameter(
                state_dict[bias_key].to(device=target_device, dtype=compute_dtype),
                requires_grad=False,
            )
        materialized += 1

    return materialized


def dequantize_weight_for_fallback(weight, compute_dtype):
    if hasattr(weight, "dequantize"):
        return weight.dequantize().to(compute_dtype)
    return weight.to(compute_dtype)
