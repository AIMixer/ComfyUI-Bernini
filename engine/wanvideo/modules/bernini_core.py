"""Dedicated Bernini Wan 2.2 transformer forward — no optional WanVideo feature branches."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ...cache_methods.cache_methods import relative_l1_distance
from ...utils import log
from comfy import model_management as mm
from .model import sinusoidal_embedding_1d, _bernini_text_cache_key, _module_to_if_needed


def _embed_context_latents(model, x_list, context_latents, seq_len):
    """Patch-embed Bernini in-context reference tokens and append to the main stream."""
    if not context_latents:
        return x_list, seq_len, None

    x = x_list
    context_frame_shapes = []
    for lat in context_latents:
        if not isinstance(lat, torch.Tensor):
            log.warning("Skipping invalid Bernini context latent (expected Tensor, got %s)", type(lat))
            continue
        if lat.ndim == 3:
            lat = lat.unsqueeze(1)
        elif lat.ndim != 4:
            log.warning("Skipping Bernini context latent with unexpected rank %d", lat.ndim)
            continue
        lat = lat.to(device=x[0].device, dtype=x[0].dtype)
        p_t, p_h, p_w = model.patch_size
        pad_t = (p_t - (lat.shape[1] % p_t)) % p_t
        pad_h = (p_h - (lat.shape[2] % p_h)) % p_h
        pad_w = (p_w - (lat.shape[3] % p_w)) % p_w
        if pad_t or pad_h or pad_w:
            lat = F.pad(lat, (0, pad_w, 0, pad_h, 0, pad_t))
        cl = model.original_patch_embedding(lat.unsqueeze(0).float()).to(x[0].dtype)
        cl = cl.flatten(2).transpose(1, 2)
        x = [torch.cat([u, cl], dim=1) for u in x]
        seq_len = max(seq_len, x[0].shape[1])
        context_frame_shapes.append(lat.shape[1:4])
    return x, seq_len, context_frame_shapes


def _embed_text_context(model, context, device):
    """Pad, embed, and cache T5 context for cross-attention (reused each denoise step)."""
    if not hasattr(model, "text_embedding") or not isinstance(context, list) or not context:
        return context

    text_embed_dtype = model.text_embedding[0].weight.dtype
    if text_embed_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        text_embed_dtype = model.base_dtype

    cache_key = _bernini_text_cache_key(context, text_embed_dtype, device, model.text_len)
    text_cache = getattr(model, "_bernini_text_embed_cache", None)
    if text_cache is None:
        text_cache = {}
        model._bernini_text_embed_cache = text_cache

    embedded = text_cache.get(cache_key)
    if embedded is None:
        _module_to_if_needed(model.text_embedding, device)
        padded = torch.stack([
            torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
            for u in context
        ]).to(text_embed_dtype)
        embedded = model.text_embedding(padded.to(device))
        if len(text_cache) >= 4:
            text_cache.clear()
        text_cache[cache_key] = embedded
    return embedded


def _embed_nag_context(model, nag_context, device, text_embed_dtype):
    if nag_context is None or not isinstance(nag_context, list) or not nag_context:
        return None
    _module_to_if_needed(model.text_embedding, device)
    padded = torch.stack([
        torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
        for u in nag_context
    ]).to(text_embed_dtype)
    return model.text_embedding(padded.to(device))


def _resolve_rope(model, F, H, W, context_frame_shapes, context_window_start, x):
    cache_key = (
        F, H, W,
        model.rope_embedder.k,
        tuple(tuple(s) for s in context_frame_shapes) if context_frame_shapes else None,
        context_window_start,
    )
    if getattr(model, "cached_freqs", None) is not None and getattr(model, "cached_key", None) == cache_key:
        return model.cached_freqs
    freqs = model.rope_encode_comfy(
        F, H, W,
        context_frame_shapes=context_frame_shapes,
        context_window_start=context_window_start,
        device=x.device,
        dtype=x.dtype,
    )
    model.cached_freqs = freqs
    model.cached_key = cache_key
    return freqs


def _time_embed(model, t, F, grid_sizes, x, device):
    time_projection_enabled = hasattr(model, "time_projection")
    if hasattr(model, "time_projection"):
        time_embed_dtype = model.time_embedding[0].weight.dtype
        if time_embed_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            time_embed_dtype = model.base_dtype
    else:
        time_embed_dtype = model.time_embedding.mlp[0].weight.dtype
        if time_embed_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            time_embed_dtype = model.base_dtype

    cached_time = getattr(model, "_bernini_time_embed_cache", None)
    if (
        cached_time is not None
        and cached_time[0] is t
        and cached_time[1] == device
        and cached_time[2] == time_embed_dtype
        and cached_time[3] == time_projection_enabled
        and cached_time[4] == F
    ):
        return cached_time[5]

    _module_to_if_needed(model.time_embedding, device)
    if time_projection_enabled:
        _module_to_if_needed(model.time_projection, device)
        e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t.flatten()).to(time_embed_dtype))
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
    else:
        if len(t.shape) == 1:
            t = t.unsqueeze(1).expand(-1, F)
        model.time_embedding.to(torch.float32)
        e0 = model.time_embedding(t.float().flatten(), dtype=torch.float32)
        e0 = e0.reshape(1, F, -1)

    if t.dim() == 2 and not model.is_longcat:
        b, f = t.shape
        e0 = e0.view(b, f, 1, 1, 6, model.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], 6, model.dim)
        e0 = e0.flatten(1, 3).transpose(1, 2)
        if not e0.is_contiguous():
            e0 = e0.contiguous()

    model._bernini_time_embed_cache = (t, device, time_embed_dtype, time_projection_enabled, F, e0)
    return e0


def _teacache_should_skip(model, e0, current_step, pred_id, device):
    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
    if pred_id is None:
        pred_id = model.teacache_state.new_prediction(cache_device=model.cache_device)
        return True, pred_id, accumulated_rel_l1_distance, None

    state = model.teacache_state.get(pred_id)
    previous_modulated_input = state["previous_modulated_input"].to(device)
    previous_residual = state["previous_residual"]
    accumulated_rel_l1_distance = state["accumulated_rel_l1_distance"]

    if model.teacache_use_coefficients:
        rescale_func = np.poly1d(model.teacache_coefficients[model.teacache_mode])
        temb = e0 if model.teacache_mode == "e" else e0
        accumulated_rel_l1_distance += rescale_func((
            (temb.to(device) - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
        ).cpu().item())
    else:
        accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + relative_l1_distance(
            previous_modulated_input, e0
        )

    if accumulated_rel_l1_distance < model.rel_l1_thresh:
        return False, pred_id, accumulated_rel_l1_distance.to(model.cache_device), previous_residual

    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
    return True, pred_id, accumulated_rel_l1_distance.to(model.cache_device), None


def _magcache_should_skip(model, current_step, total_steps, pred_id, x, device):
    if pred_id is None:
        pred_id = model.magcache_state.new_prediction(cache_device=model.cache_device)
        return True, pred_id, x, None

    state = model.magcache_state.get(pred_id)
    accumulated_ratio = state["accumulated_ratio"]
    accumulated_err = state["accumulated_err"]
    accumulated_steps = state["accumulated_steps"]

    calibration_len = len(model.magcache_ratios) // 2
    cur_mag_ratio = model.magcache_ratios[int((current_step * (calibration_len / max(total_steps, 1))))]
    accumulated_ratio *= cur_mag_ratio
    accumulated_err += abs(1 - accumulated_ratio)
    accumulated_steps += 1

    model.magcache_state.update(
        pred_id,
        accumulated_ratio=accumulated_ratio,
        accumulated_steps=accumulated_steps,
        accumulated_err=accumulated_err,
    )

    if accumulated_err <= model.magcache_thresh and accumulated_steps <= model.magcache_K:
        residual = model.magcache_state.get(pred_id)["residual_cache"]
        if residual is not None:
            x = x + residual.to(x.device)
        model.magcache_state.get(pred_id)["skipped_steps"].append(current_step)
        return False, pred_id, x, None

    model.magcache_state.update(
        pred_id,
        accumulated_ratio=1.0,
        accumulated_steps=0,
        accumulated_err=0,
    )
    return True, pred_id, x, x.clone().to(model.cache_device)


def _apply_step_cache(model, e0, x, current_step, total_steps, pred_id, device):
    """TeaCache / MagCache gate before the block loop. Returns (should_calc, pred_id, x, original_x)."""
    should_calc = True
    original_x = None
    previous_modulated_input = None
    accumulated_rel_l1_distance = None

    if model.enable_teacache and model.teacache_start_step <= current_step <= model.teacache_end_step:
        should_calc, pred_id, accumulated_rel_l1_distance, previous_residual = _teacache_should_skip(
            model, e0, current_step, pred_id, device
        )
        previous_modulated_input = e0.to(model.cache_device).clone()
        if not should_calc and previous_residual is not None:
            x = x.to(previous_residual.dtype) + previous_residual.to(x.device)
            model.teacache_state.update(pred_id, accumulated_rel_l1_distance=accumulated_rel_l1_distance)
            model.teacache_state.get(pred_id)["skipped_steps"].append(current_step)
            return False, pred_id, x, None, previous_modulated_input, accumulated_rel_l1_distance
        should_calc = True
        original_x = x.clone().to(model.cache_device)

    if model.enable_magcache and model.magcache_start_step <= current_step <= model.magcache_end_step:
        should_calc, pred_id, x, mag_original = _magcache_should_skip(
            model, current_step, total_steps, pred_id, x, device
        )
        if not should_calc:
            return False, pred_id, x, None, previous_modulated_input, accumulated_rel_l1_distance
        if mag_original is not None:
            original_x = mag_original

    return should_calc, pred_id, x, original_x, previous_modulated_input, accumulated_rel_l1_distance


def forward(
    model,
    x,
    t,
    context,
    seq_len,
    freqs=None,
    *,
    context_latents=None,
    context_window_start=0,
    is_uncond=False,
    current_step=0,
    last_step=0,
    total_steps=50,
    pred_id=None,
    device=None,
    transformer_options=None,
    nag_params=None,
    nag_context=None,
    enhance_enabled=False,
    **_,
):
    """Bernini-only diffusion step — patch embed, context tokens, blocks, head."""
    device = device or model.main_device
    transformer_options = transformer_options or {}

    if model.lora_scheduling_enabled:
        from ...custom_linear import update_lora_step
        update_lora_step(model, current_step)

    _, F, H, W = x[0].shape
    suffix_frames = x[0].shape[1]
    prefix_frames = 0

    model.original_patch_embedding.to(model.main_device)
    x = [model.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(model.base_dtype) for u in x]

    grid_sizes = torch.stack([torch.tensor(u.shape[2:], device=device, dtype=torch.long) for u in x])
    original_grid_sizes = grid_sizes.clone()
    x = [u.flatten(2).transpose(1, 2) for u in x]
    model.original_seq_len = x[0].shape[1]

    x, seq_len, context_frame_shapes = _embed_context_latents(model, x, context_latents, seq_len)
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32)
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    if freqs is None and "comfy" in model.rope_func:
        freqs = _resolve_rope(model, F, H, W, context_frame_shapes, context_window_start, x)
    if freqs is not None and freqs.device != device:
        freqs = freqs.to(device)

    text_embed_dtype = None
    if context is not None:
        context = _embed_text_context(model, context, device)
        if hasattr(model, "text_embedding"):
            text_embed_dtype = model.text_embedding[0].weight.dtype
            if text_embed_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                text_embed_dtype = model.base_dtype
        if nag_context is not None and not is_uncond:
            nag_context = _embed_nag_context(model, nag_context, device, text_embed_dtype)
        if getattr(model, "offload_txt_emb", False):
            model.text_embedding.to(model.offload_device, non_blocking=getattr(model, "use_non_blocking", False))

    e0 = _time_embed(model, t, F, grid_sizes, x, device)
    x = x.to(model.base_dtype)
    e0 = e0.to(model.base_dtype)

    should_calc, pred_id, x, original_x, previous_modulated_input, accumulated_rel_l1_distance = _apply_step_cache(
        model, e0, x, current_step, total_steps, pred_id, device
    )

    block_kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        clip_embed=None,
        current_step=torch.tensor(current_step),
        last_step=torch.tensor(last_step, dtype=torch.bool),
        num_latent_frames=F,
        frame_tokens=x.shape[1] // F,
        original_seq_len=model.original_seq_len,
        enhance_enabled=enhance_enabled,
        nag_params=nag_params or {},
        nag_context=nag_context if not is_uncond else None,
        transformer_options=transformer_options,
    )

    swap_start_idx = len(model.blocks) - model.blocks_to_swap if model.blocks_to_swap > 0 else len(model.blocks)
    use_non_blocking = getattr(model, "use_non_blocking", False)
    prefetch_blocks = getattr(model, "prefetch_blocks", 0)
    swap_stream = getattr(model, "swap_cuda_stream", None)
    events = (
        [torch.cuda.Event() for _ in model.blocks]
        if prefetch_blocks > 0 and swap_stream is None and torch.cuda.is_available()
        else None
    )

    if should_calc:
        for b, block in enumerate(model.blocks):
            mm.throw_exception_if_processing_interrupted()
            if b >= swap_start_idx and model.blocks_to_swap > 0:
                block.to(model.main_device, non_blocking=use_non_blocking)
            if prefetch_blocks > 0:
                if swap_stream is not None:
                    for pf in range(1, min(prefetch_blocks, len(model.blocks) - b - 1) + 1):
                        next_b = b + pf
                        if next_b >= swap_start_idx and model.blocks_to_swap > 0:
                            with torch.cuda.stream(swap_stream):
                                model.blocks[next_b].to(model.main_device, non_blocking=use_non_blocking)
                else:
                    for prefetch_offset in range(1, prefetch_blocks + 1):
                        prefetch_idx = b + prefetch_offset
                        if (
                            prefetch_idx < len(model.blocks)
                            and model.blocks_to_swap > 0
                            and prefetch_idx >= swap_start_idx
                        ):
                            model.blocks[prefetch_idx].to(model.main_device, non_blocking=use_non_blocking)
                            if events is not None:
                                events[prefetch_idx].record()
            if b >= swap_start_idx and model.blocks_to_swap > 0 and events is not None:
                if not events[b].query():
                    events[b].synchronize()
            if model.slg_blocks is not None and b in model.slg_blocks and is_uncond:
                step_pct = current_step / max(total_steps, 1)
                if model.slg_start_percent <= step_pct <= model.slg_end_percent:
                    continue
            x, _, _, _ = block(x, x_ip=None, lynx_ref_feature=None, x_ovi=None, attention_mode_override=None, **block_kwargs)
            if prefetch_blocks > 0 and swap_stream is not None:
                torch.cuda.current_stream().wait_stream(swap_stream)
            if b >= swap_start_idx and model.blocks_to_swap > 0:
                block.to(model.offload_device, non_blocking=use_non_blocking)

        if (
            model.enable_teacache
            and model.teacache_start_step <= current_step <= model.teacache_end_step
            and pred_id is not None
            and original_x is not None
        ):
            model.teacache_state.update(
                pred_id,
                previous_residual=(x.to(original_x.device) - original_x),
                accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                previous_modulated_input=previous_modulated_input,
            )
        elif (
            model.enable_magcache
            and model.magcache_start_step <= current_step <= model.magcache_end_step
            and pred_id is not None
            and original_x is not None
        ):
            model.magcache_state.update(
                pred_id,
                residual_cache=(x.to(original_x.device) - original_x),
            )

    x = x[:, : model.original_seq_len]
    x = model.head(x, e0.to(x.device), temp_length=F)
    x = model.unpatchify(x, original_grid_sizes)
    x = [u[:, prefix_frames:suffix_frames, ...].float() for u in x]
    return (x, None, pred_id) if pred_id is not None else (x, None, None)
