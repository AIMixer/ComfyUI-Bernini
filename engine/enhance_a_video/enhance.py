import torch
from einops import rearrange
from .globals import get_enhance_weight, get_num_frames


@torch.compiler.disable()
def get_feta_scores(query, key):
    img_q, img_k = query, key

    num_frames = get_num_frames()

    B, S, N, C = img_q.shape

    spatial_dim = S // num_frames

    query_image = img_q.reshape(B, spatial_dim, num_frames, N, C)
    key_image = img_k.reshape(B, spatial_dim, num_frames, N, C)

    query_image = query_image.expand(-1, -1, num_frames, -1, -1)
    key_image = key_image.expand(-1, -1, num_frames, -1, -1)

    query_image = rearrange(query_image, "b s t n c -> (b s) n t c")
    key_image = rearrange(key_image, "b s t n c -> (b s) n t c")

    return feta_score(query_image, key_image, C, num_frames)


@torch.compiler.disable()
def feta_score(query_image, key_image, head_dim, num_frames):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + get_enhance_weight())
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores
