from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAColor(nn.Module):
    def __init__(
        self,
        palette_rgb: torch.Tensor,
        embedding_dim: int = 70,
        hidden_dim: int = 128,
        aux_hidden_dim: int = 128,
        temperature: float = 0.1,
        aux_positive_output: bool = True,
        aux_output_activation: str = "softplus",
        aux_feature_mode: str = "symmetric",
        fusion_mode: str = "basic",
        embedding_mode: str = "rgb_id",
        prior_mode: str = "none",
        prior_weight: float = 1.0,
        neural_weight: float = 1.0,
        cooccurrence_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.aux_hidden_dim = aux_hidden_dim
        self.aux_positive_output = aux_positive_output
        self.aux_output_activation = aux_output_activation
        self.aux_feature_mode = aux_feature_mode
        self.fusion_mode = fusion_mode
        self.embedding_mode = embedding_mode
        self.prior_mode = prior_mode
        self.prior_weight = float(prior_weight)
        self.neural_weight = float(neural_weight)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if aux_feature_mode not in {"symmetric", "concatenate"}:
            raise ValueError(f"Unsupported aux_feature_mode: {aux_feature_mode}")
        if aux_output_activation not in {"relu", "softplus", "linear"}:
            raise ValueError(f"Unsupported auxiliary output activation: {aux_output_activation}")
        if embedding_mode not in {"rgb", "rgb_id"}:
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}")
        if prior_mode not in {"none", "cooccurrence"}:
            raise ValueError(f"Unsupported prior mode: {prior_mode}")
        if prior_weight < 0 or neural_weight < 0:
            raise ValueError("prior_weight and neural_weight must be non-negative")

        self.register_buffer("palette_rgb", palette_rgb.float() / 255.0)
        if prior_mode == "cooccurrence":
            if cooccurrence_prior is None:
                cooccurrence_prior = torch.zeros(
                    (int(palette_rgb.shape[0]), int(palette_rgb.shape[0])), dtype=torch.float32
                )
            if tuple(cooccurrence_prior.shape) != (int(palette_rgb.shape[0]), int(palette_rgb.shape[0])):
                raise ValueError("cooccurrence_prior must have shape [palette_size, palette_size]")
            self.register_buffer("cooccurrence_prior", cooccurrence_prior.float())

        self.embed_fc1 = nn.Linear(3, hidden_dim)
        self.embed_fc2 = nn.Linear(hidden_dim, embedding_dim)
        if embedding_mode == "rgb_id":
            self.id_embeddings = nn.Embedding(int(palette_rgb.shape[0]), embedding_dim)
            nn.init.normal_(self.id_embeddings.weight, mean=0.0, std=0.02)
        if fusion_mode in {"basic", "mean"}:
            self.query_fc = nn.Linear(embedding_dim, embedding_dim)
        elif fusion_mode == "enhanced":
            self.fusion_norm = nn.LayerNorm(embedding_dim)
            self.pool_proj = nn.Linear(embedding_dim, 1)
            self.query_fc = nn.Linear(embedding_dim * 2, embedding_dim)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.aux_fc1 = nn.Linear(embedding_dim * 2, aux_hidden_dim)
        self.aux_fc2 = nn.Linear(aux_hidden_dim, 1)

    def color_embeddings(self, color_ids: torch.Tensor) -> torch.Tensor:
        rgb = self.palette_rgb[color_ids]
        hidden = F.relu(self.embed_fc1(rgb))
        embeddings = self.embed_fc2(hidden)
        if self.embedding_mode == "rgb_id":
            embeddings = embeddings + self.id_embeddings(color_ids)
        return F.relu(embeddings)

    def encode_query(self, query_ids: torch.Tensor, query_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeds = self.color_embeddings(query_ids.clamp_min(0))
        query_embeds = query_embeds * query_mask.unsqueeze(-1)

        if self.fusion_mode == "mean":
            pooled = query_embeds.sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1)
            query_vector = F.relu(self.query_fc(pooled))
            return query_vector, query_embeds

        scores = torch.matmul(query_embeds, query_embeds.transpose(1, 2)) / math.sqrt(self.embedding_dim)
        key_mask = query_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(key_mask, attention, torch.zeros_like(attention))
        attended = torch.matmul(attention, query_embeds)
        attended = attended * query_mask.unsqueeze(-1)

        if self.fusion_mode == "basic":
            pooled = attended.sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            contextual = self.fusion_norm(query_embeds + attended)
            pool_logits = self.pool_proj(contextual).squeeze(-1)
            pool_logits = pool_logits.masked_fill(~query_mask, float("-inf"))
            pool_weights = torch.softmax(pool_logits, dim=-1)
            pool_weights = torch.where(query_mask, pool_weights, torch.zeros_like(pool_weights))
            weighted_pool = torch.einsum("bq,bqd->bd", pool_weights, contextual)
            masked_contextual = contextual.masked_fill(~query_mask.unsqueeze(-1), float("-inf"))
            max_pool = masked_contextual.max(dim=1).values
            pooled = torch.cat([weighted_pool, max_pool], dim=-1)
        query_vector = F.relu(self.query_fc(pooled))
        return query_vector, query_embeds

    def predict_auxiliary(self, query_embeds: torch.Tensor, query_mask: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        target_embeds = self.color_embeddings(target_ids).unsqueeze(1).expand_as(query_embeds)
        if self.aux_feature_mode == "symmetric":
            combined = torch.cat([(query_embeds - target_embeds).abs(), query_embeds * target_embeds], dim=-1)
        else:
            combined = torch.cat([query_embeds, target_embeds], dim=-1)
        hidden = F.relu(self.aux_fc1(combined))
        predictions = self.aux_fc2(hidden).squeeze(-1)
        activation = self.aux_output_activation if self.aux_positive_output else "linear"
        if activation == "relu":
            predictions = F.relu(predictions)
        elif activation == "softplus":
            predictions = F.softplus(predictions)
        return predictions * query_mask

    @staticmethod
    def cosine_scores(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        return torch.matmul(left, right.transpose(-1, -2))

    def score_candidates(self, query_ids: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        query_vector, _ = self.encode_query(query_ids, query_mask)
        all_color_ids = torch.arange(self.palette_rgb.shape[0], device=query_ids.device)
        all_embeddings = self.color_embeddings(all_color_ids)
        neural_scores = self.cosine_scores(query_vector, all_embeddings) / self.temperature
        return self.combine_with_prior(neural_scores, query_ids, query_mask)

    def combine_with_prior(
        self,
        neural_scores: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.neural_weight * neural_scores
        if self.prior_mode == "cooccurrence":
            prior_rows = self.cooccurrence_prior[query_ids.clamp_min(0)]
            prior_rows = prior_rows * query_mask.unsqueeze(-1)
            prior_scores = prior_rows.sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1)
            scores = scores + self.prior_weight * prior_scores
        return scores

    def config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "aux_hidden_dim": self.aux_hidden_dim,
            "temperature": self.temperature,
            "aux_positive_output": self.aux_positive_output,
            "aux_output_activation": self.aux_output_activation,
            "aux_feature_mode": self.aux_feature_mode,
            "fusion_mode": self.fusion_mode,
            "embedding_mode": self.embedding_mode,
            "prior_mode": self.prior_mode,
            "prior_weight": self.prior_weight,
            "neural_weight": self.neural_weight,
            "palette_size": int(self.palette_rgb.shape[0]),
        }
