# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/omni.py`; body unmodified except as recorded in ../SOURCE.md.

"""Neural components for the optional governed Omni training objective."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class OmniEvidenceBridge(nn.Module):
    """Project optional modality slots into the existing lattice hidden stream."""

    def __init__(
        self,
        hidden_size: int,
        feature_size: int,
        feature_quality_size: int,
        modality_count: int = 4,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(feature_size, hidden_size)
        self.modality_embedding = nn.Embedding(modality_count, hidden_size)
        self.time_projection = nn.Linear(1, hidden_size)
        self.quality_projection = nn.Linear(feature_quality_size, hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        feature_slots: torch.Tensor | None,
        feature_mask: torch.Tensor | None,
        feature_type: torch.Tensor | None = None,
        feature_time: torch.Tensor | None = None,
        feature_quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if feature_slots is None:
            return None
        if feature_slots.ndim != 3:
            raise ValueError("omni_feature_slots must have shape [batch, slots, feature_size]")
        batch, slots, _ = feature_slots.shape
        device = feature_slots.device
        dtype = self.feature_projection.weight.dtype
        feature_slots = feature_slots.to(device=device, dtype=dtype)
        if feature_mask is None:
            feature_mask = torch.ones((batch, slots), device=device, dtype=dtype)
        else:
            feature_mask = feature_mask.to(device=device, dtype=dtype)
        if feature_type is None:
            feature_type = torch.zeros((batch, slots), device=device, dtype=torch.long)
        else:
            feature_type = feature_type.to(device=device, dtype=torch.long).clamp_min(0)
        if feature_time is None:
            feature_time = torch.zeros((batch, slots), device=device, dtype=dtype)
        else:
            feature_time = feature_time.to(device=device, dtype=dtype)
        if feature_quality is None:
            feature_quality = torch.zeros(
                (batch, slots, self.quality_projection.in_features),
                device=device,
                dtype=dtype,
            )
        else:
            feature_quality = feature_quality.to(device=device, dtype=dtype)

        slots_hidden = self.feature_projection(feature_slots)
        slots_hidden = slots_hidden + self.modality_embedding(
            feature_type.clamp_max(self.modality_embedding.num_embeddings - 1)
        )
        slots_hidden = slots_hidden + self.time_projection(feature_time.unsqueeze(-1))
        slots_hidden = slots_hidden + self.quality_projection(feature_quality)
        slots_hidden = self.context_norm(slots_hidden)
        mask = feature_mask.unsqueeze(-1).clamp(0.0, 1.0)
        denom = mask.sum(dim=1).clamp_min(1.0)
        context = (slots_hidden * mask).sum(dim=1) / denom
        context = self.context_gate(context)
        return {
            "slots": slots_hidden,
            "mask": mask.squeeze(-1),
            "context": context,
        }


class OmniStateHead(nn.Module):
    """Predict the next governed state and fixed semantic boundary labels."""

    def __init__(
        self,
        hidden_size: int,
        recurrent_state_size: int,
        state_size: int,
        response_mode_count: int,
        ambiguity_count: int,
        claim_count: int,
    ) -> None:
        super().__init__()
        latent_size = hidden_size
        self.state_size = state_size
        self.before_state_projection = nn.Linear(state_size, hidden_size)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + recurrent_state_size + hidden_size, latent_size),
            nn.GELU(),
            nn.LayerNorm(latent_size),
        )
        self.next_state = nn.Linear(latent_size, state_size)
        self.state_delta = nn.Linear(latent_size, state_size)
        self.response_mode = nn.Linear(latent_size, response_mode_count)
        self.ambiguity = nn.Linear(latent_size, ambiguity_count)
        self.claim = nn.Linear(latent_size, claim_count)
        self.confidence = nn.Linear(latent_size, 1)
        self.uncertainty = nn.Linear(latent_size, 1)
        self.dissent = nn.Linear(latent_size, 1)
        self.state_hidden = nn.Linear(latent_size, hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_before: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = hidden.mean(dim=1)
        if state_before is None:
            state_before = pooled.new_zeros((pooled.size(0), self.state_size))
        else:
            state_before = state_before.to(device=pooled.device, dtype=pooled.dtype)
        before_hidden = self.before_state_projection(state_before)
        latent = self.fuse(torch.cat([pooled, recurrent_state, before_hidden], dim=-1))
        return {
            "latent": latent,
            "next_state": self.next_state(latent),
            "state_delta": self.state_delta(latent),
            "response_mode_logits": self.response_mode(latent),
            "ambiguity_logits": self.ambiguity(latent),
            "claim_logits": self.claim(latent),
            "confidence": torch.sigmoid(self.confidence(latent)).squeeze(-1),
            "uncertainty": torch.sigmoid(self.uncertainty(latent)).squeeze(-1),
            "dissent_logits": self.dissent(latent).squeeze(-1),
            "state_hidden": self.state_hidden(latent),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if values.ndim == 0:
        return values
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = (values * mask).sum()
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return numerator / denominator


def _pairwise_alignment_loss(
    slots: torch.Tensor | None,
    mask: torch.Tensor | None,
    feature_type: torch.Tensor | None,
) -> torch.Tensor | None:
    if slots is None or mask is None or feature_type is None:
        return None
    losses: list[torch.Tensor] = []
    normalized = F.normalize(slots, dim=-1)
    for batch_index in range(slots.size(0)):
        valid = mask[batch_index] > 0.5
        for left in range(slots.size(1)):
            if not bool(valid[left]):
                continue
            for right in range(left + 1, slots.size(1)):
                if not bool(valid[right]):
                    continue
                if int(feature_type[batch_index, left]) == int(feature_type[batch_index, right]):
                    continue
                losses.append(1.0 - (normalized[batch_index, left] * normalized[batch_index, right]).sum())
    if not losses:
        return slots.new_zeros(())
    return torch.stack(losses).mean()


def compute_omni_losses(
    predictions: dict[str, torch.Tensor],
    *,
    targets: dict[str, torch.Tensor | None],
    feature_slots: torch.Tensor | None,
    feature_mask: torch.Tensor | None,
    feature_type: torch.Tensor | None,
    transition_weight: float,
    semantic_weight: float,
    calibration_weight: float,
    alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute finite, independently visible Omni losses."""

    valid = targets.get("omni_valid")
    if valid is None:
        valid = predictions["next_state"].new_ones(predictions["next_state"].size(0))
    valid = valid.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)

    state_after = targets.get("omni_state_after")
    state_delta = targets.get("omni_state_delta")
    if state_after is None or state_delta is None:
        raise ValueError("Omni state_after and state_delta targets are required")
    state_after = state_after.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
    state_delta = state_delta.to(device=predictions["state_delta"].device, dtype=predictions["state_delta"].dtype)
    next_state_error = F.smooth_l1_loss(predictions["next_state"], state_after, reduction="none").mean(dim=-1)
    delta_error = F.smooth_l1_loss(predictions["state_delta"], state_delta, reduction="none").mean(dim=-1)
    state_before = targets.get("omni_state_before")
    if state_before is not None:
        state_before = state_before.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
        consistency_error = F.smooth_l1_loss(
            predictions["next_state"] - state_before,
            predictions["state_delta"],
            reduction="none",
        ).mean(dim=-1)
    else:
        consistency_error = next_state_error.new_zeros(next_state_error.shape)
    transition_loss = _masked_mean(0.5 * (next_state_error + delta_error) + 0.1 * consistency_error, valid)

    response_target = targets.get("omni_response_mode")
    claim_target = targets.get("omni_claim")
    ambiguity_target = targets.get("omni_ambiguity")
    semantic_terms: list[torch.Tensor] = []
    if response_target is not None:
        response_target = response_target.to(device=predictions["next_state"].device, dtype=torch.long)
        semantic_terms.append(F.cross_entropy(predictions["response_mode_logits"], response_target, reduction="none"))
    if claim_target is not None:
        claim_target = claim_target.to(device=predictions["next_state"].device, dtype=torch.long)
        semantic_terms.append(F.cross_entropy(predictions["claim_logits"], claim_target, reduction="none"))
    if ambiguity_target is not None:
        ambiguity_target = ambiguity_target.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
        ambiguity_error = F.binary_cross_entropy_with_logits(
            predictions["ambiguity_logits"], ambiguity_target, reduction="none"
        ).mean(dim=-1)
        semantic_terms.append(ambiguity_error)
    semantic_loss = _masked_mean(torch.stack(semantic_terms, dim=0).mean(dim=0), valid) if semantic_terms else transition_loss.new_zeros(())

    calibration_terms: list[torch.Tensor] = []
    confidence_target = targets.get("omni_confidence")
    uncertainty_target = targets.get("omni_uncertainty")
    dissent_target = targets.get("omni_dissent")
    if confidence_target is not None:
        confidence_target = confidence_target.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
        calibration_terms.append((predictions["confidence"] - confidence_target).square())
    if uncertainty_target is not None:
        uncertainty_target = uncertainty_target.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
        calibration_terms.append((predictions["uncertainty"] - uncertainty_target).square())
    if dissent_target is not None:
        dissent_target = dissent_target.to(device=predictions["next_state"].device, dtype=predictions["next_state"].dtype)
        calibration_terms.append(F.binary_cross_entropy_with_logits(
            predictions["dissent_logits"], dissent_target, reduction="none"
        ))
    calibration_loss = _masked_mean(torch.stack(calibration_terms, dim=0).mean(dim=0), valid) if calibration_terms else transition_loss.new_zeros(())

    alignment_mask = feature_mask
    if alignment_mask is not None:
        alignment_mask = alignment_mask.to(device=valid.device, dtype=valid.dtype) * valid.unsqueeze(-1)
    alignment_loss = _pairwise_alignment_loss(feature_slots, alignment_mask, feature_type)
    if alignment_loss is None:
        alignment_loss = transition_loss.new_zeros(())
    total = (
        transition_loss * float(transition_weight)
        + semantic_loss * float(semantic_weight)
        + calibration_loss * float(calibration_weight)
        + alignment_loss * float(alignment_weight)
    )
    metrics = {
        "omni_total": total.detach(),
        "omni_transition": transition_loss.detach(),
        "omni_state_consistency": _masked_mean(consistency_error, valid).detach(),
        "omni_semantic": semantic_loss.detach(),
        "omni_calibration": calibration_loss.detach(),
        "omni_alignment": alignment_loss.detach(),
        "omni_valid_fraction": valid.detach().mean(),
    }
    return total, metrics
