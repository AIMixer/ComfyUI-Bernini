import torch
import torch.nn as nn
from accelerate import init_empty_weights
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor


def _align_block_scale(scale: torch.Tensor, column_dim: int) -> torch.Tensor:
    """Expand MXFP8 block-compressed scale rows to match weight width."""
    if scale.ndim < 2 or scale.shape[-1] == column_dim:
        return scale
    blocks = column_dim // scale.shape[-1]
    if blocks > 1 and blocks * scale.shape[-1] == column_dim:
        return scale.repeat_interleave(blocks, dim=-1)
    return scale


@torch.library.custom_op("bernini::apply_lora", mutates_args=())
def apply_lora(
    weight: torch.Tensor,
    lora_diff_0: torch.Tensor,
    lora_diff_1: torch.Tensor,
    lora_diff_2: float,
    lora_strength: torch.Tensor,
) -> torch.Tensor:
    patch_diff = torch.mm(
        lora_diff_0.flatten(start_dim=1),
        lora_diff_1.flatten(start_dim=1),
    ).reshape(weight.shape)
    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    return weight + patch_diff * lora_strength * alpha


@apply_lora.register_fake
def _apply_lora_meta(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
    return weight.clone()


@torch.library.custom_op("bernini::apply_single_lora", mutates_args=())
def apply_single_lora(
    weight: torch.Tensor,
    lora_diff: torch.Tensor,
    lora_strength: torch.Tensor,
) -> torch.Tensor:
    return weight + lora_diff * lora_strength


@apply_single_lora.register_fake
def _apply_single_lora_meta(weight, lora_diff, lora_strength):
    return weight.clone()


@torch.library.custom_op("bernini::linear_forward", mutates_args=())
def linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.nn.functional.linear(input, weight, bias)


@linear_forward.register_fake
def _linear_forward_meta(input, weight, bias):
    out_features = weight.shape[0]
    return input.new_empty(list(input.shape[:-1]) + [out_features])


#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None, modules_to_not_convert=[]):

    has_children = list(model.children())
    if not has_children:
        return

    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args, modules_to_not_convert)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix and "dual_controller" not in module_prefix and name not in modules_to_not_convert:
            weight_key = module_prefix + "weight"
            if weight_key not in state_dict:
                continue

            in_features = state_dict[weight_key].shape[1]
            out_features = state_dict[weight_key].shape[0]

            is_gguf = isinstance(state_dict[weight_key], GGUFParameter)

            scale_weight = None
            if not is_gguf and scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"
                scale_weight = scale_weights.get(scale_key)

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weight,
                    allow_compile=allow_compile,
                    is_gguf=is_gguf
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu")):
    remove_lora_from_module(module)
    for name, child in module.named_children():
        params = list(child.parameters())
        device = params[0].device if params else torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.set_lora_strengths(lora_strengths, device=device)
            module._step.fill_(0)


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False,
        is_gguf=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile
        self.is_gguf = is_gguf
        self._bernini_ops = torch.ops.bernini

        if allow_compile:
            self._merge_lora = self._merge_lora_eager
            self._merge_single = self._merge_single_eager
            self._matmul = self._matmul_eager
        else:
            self._merge_lora = self._merge_lora_op
            self._merge_single = self._merge_single_op
            self._matmul = self._matmul_op

    def _merge_lora_eager(self, weight, lora_a, lora_b, rank_scale, strength):
        delta = torch.mm(lora_a.flatten(start_dim=1), lora_b.flatten(start_dim=1)).reshape(weight.shape)
        norm = rank_scale / lora_b.shape[0] if rank_scale != 0.0 else 1.0
        return weight + delta * strength * norm

    def _merge_single_eager(self, weight, delta, strength):
        return weight + delta * strength

    def _matmul_eager(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    def _merge_lora_op(self, weight, lora_a, lora_b, rank_scale, strength):
        return self._bernini_ops.apply_lora(
            weight, lora_a, lora_b,
            float(rank_scale) if rank_scale is not None else 0.0,
            strength,
        )

    def _merge_single_op(self, weight, delta, strength):
        return self._bernini_ops.apply_single_lora(weight, delta, strength)

    def _matmul_op(self, input, weight, bias):
        return self._bernini_ops.linear_forward(input, weight, bias)

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self.lora_diffs = []
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

    def set_lora_strengths(self, lora_strengths, device=torch.device("cpu")):
        self._lora_strength_is_scheduled = []
        self._step = self._step.to(device)
        for i, strength in enumerate(lora_strengths):
            scheduled = isinstance(strength, list)
            values = strength if scheduled else [strength]
            tensor = torch.tensor(values, dtype=self.compute_dtype, device=device)
            self.register_buffer(f"_lora_strength_{i}", tensor)
            self._lora_strength_is_scheduled.append(scheduled)

    def _strength_at(self, idx):
        buf = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return buf.index_select(0, self._step).squeeze(0)
        return buf[0]

    def _fold_lora_into_weight(self, weight):
        if not hasattr(self, "lora_diff_0_0"):
            return weight
        for idx, spec in enumerate(self.lora_diffs):
            strength = self._strength_at(idx)
            if isinstance(spec, tuple):
                a = getattr(self, spec[0])
                b = getattr(self, spec[1])
                rank_scale = getattr(self, spec[2])
                weight = self._merge_lora(
                    weight, a, b,
                    float(rank_scale) if rank_scale is not None else 0.0,
                    strength,
                )
            else:
                weight = self._merge_single(weight, getattr(self, spec), strength)
        return weight

    def _resolve_weight(self, input):
        if self.is_gguf:
            return dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        return self.weight.to(input)

    def forward(self, input):
        weight = self._resolve_weight(input)
        bias = self.bias.to(input if not self.is_gguf else self.compute_dtype) if self.bias is not None else None

        if not self.is_gguf and self.scale_weight is not None:
            sw = _align_block_scale(self.scale_weight, weight.shape[-1])
            if weight.numel() < input.numel():
                weight = weight * sw
            else:
                input = input * sw

        weight = self._fold_lora_into_weight(weight)
        out = self._matmul(input, weight, bias)
        del weight, input, bias
        return out

def update_lora_step(module, step):
    for submodule in module.modules():
        if isinstance(submodule, CustomLinear) and hasattr(submodule, "_step"):
            submodule._step.fill_(step)

def remove_lora_from_module(module):
    for submodule in module.modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                for suffix in ("_0", "_1", "_2"):
                    attr = f"lora_diff_{i}{suffix}"
                    if hasattr(submodule, attr):
                        delattr(submodule, attr)
