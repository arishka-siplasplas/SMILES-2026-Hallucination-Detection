from __future__ import annotations

import torch
import torch.nn.functional as F

SELECTED_LAYERS = (8, 12, 16, 20)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    last = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    feats = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        masked = layer[mask]
        feats.append(masked.mean(dim=0))
        feats.append(masked.std(dim=0) if masked.shape[0] > 1 else torch.zeros(layer.shape[-1]))
        feats.append(layer[last])
    return torch.cat(feats, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    pooled = hidden_states[:, mask, :].mean(dim=1)
    norms = pooled.norm(dim=1)
    drift = F.cosine_similarity(pooled[:-1], pooled[1:], dim=1)
    length = mask.sum().float().reshape(1)
    return torch.cat([norms, drift, length], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    feats = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        feats = torch.cat([feats, geo], dim=0)
    return feats
