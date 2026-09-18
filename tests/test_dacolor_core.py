import pytest
import torch
import torch.nn.functional as F

from dacolor.data import PairDataset, SchemeBatchSampler, pair_collate_fn
from dacolor.model import DAColor
from dacolor.train import sample_negative_ids, training_step
from dacolor.prior import build_log_cooccurrence_prior


def test_negative_sampling_never_uses_real_color_zero_from_padded_schemes():
    torch.manual_seed(7)
    scheme_ids = torch.tensor([[0, 4, -1, -1], [1, 2, 3, 4]])
    scheme_mask = scheme_ids.ge(0)
    for _ in range(20):
        negatives = sample_negative_ids(scheme_ids, scheme_mask, palette_size=8, negative_size=4)
        assert not set(negatives[0].tolist()) & {0, 4}
        assert not set(negatives[1].tolist()) & {1, 2, 3, 4}


def test_symmetric_auxiliary_features_give_same_prediction_in_both_directions():
    torch.manual_seed(11)
    palette_rgb = torch.tensor([[10, 20, 30], [200, 150, 100]], dtype=torch.float32)
    model = DAColor(palette_rgb, embedding_dim=8, hidden_dim=12, aux_hidden_dim=10, aux_feature_mode="symmetric")
    mask = torch.ones((1, 1), dtype=torch.bool)
    first = model.color_embeddings(torch.tensor([[0]]))
    second = model.color_embeddings(torch.tensor([[1]]))

    forward = model.predict_auxiliary(first, mask, torch.tensor([1]))
    reverse = model.predict_auxiliary(second, mask, torch.tensor([0]))
    assert torch.allclose(forward, reverse, atol=1e-7)


def test_scheme_batch_sampler_keeps_all_pairs_for_each_selected_scheme(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"scheme_id": 1, "query_ids": [0], "target_id": 1, "scheme_color_ids": [0, 1], "query_size": 1}',
                '{"scheme_id": 1, "query_ids": [1], "target_id": 0, "scheme_color_ids": [0, 1], "query_size": 1}',
                '{"scheme_id": 2, "query_ids": [2], "target_id": 3, "scheme_color_ids": [2, 3], "query_size": 1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PairDataset(path)
    batches = list(SchemeBatchSampler(dataset, schemes_per_batch=1, shuffle=False, seed=0))
    assert batches == [[0, 1], [2]]


def test_training_main_loss_gives_each_scheme_equal_weight():
    torch.manual_seed(19)
    model = DAColor(torch.randint(0, 256, (8, 3), dtype=torch.float32), embedding_dim=6, hidden_dim=9)
    samples = [
        {"scheme_id": 10, "query_ids": [0], "target_id": 1, "scheme_color_ids": [0, 1], "query_size": 1},
        {"scheme_id": 20, "query_ids": [2], "target_id": 3, "scheme_color_ids": [2, 3, 4, 5], "query_size": 1},
        {"scheme_id": 20, "query_ids": [2], "target_id": 4, "scheme_color_ids": [2, 3, 4, 5], "query_size": 1},
        {"scheme_id": 20, "query_ids": [2], "target_id": 5, "scheme_color_ids": [2, 3, 4, 5], "query_size": 1},
    ]
    batch = pair_collate_fn(samples)

    torch.manual_seed(23)
    negatives = sample_negative_ids(batch["scheme_color_ids"], batch["scheme_mask"], 8, 2)
    query_vector, _ = model.encode_query(batch["query_ids"], batch["query_mask"])
    positive_embeddings = model.color_embeddings(batch["target_ids"])
    negative_embeddings = model.color_embeddings(negatives)
    logits = torch.cat(
        [
            (query_vector * positive_embeddings).sum(dim=-1, keepdim=True) / model.temperature,
            torch.einsum("bd,bnd->bn", query_vector, negative_embeddings) / model.temperature,
        ],
        dim=1,
    )
    per_pair = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long), reduction="none")
    expected = (per_pair[0] + per_pair[1:].mean()) / 2

    torch.manual_seed(23)
    loss, details = training_step(
        model,
        batch,
        ciede2000=torch.zeros((8, 8)),
        alpha=1.0,
        negative_size=2,
        aux_scale=torch.tensor(1.0),
        aux_scale_mode="max",
        aux_reduction="token_mean",
        contrastive_similarity="inner_product",
        negative_mode="random",
    )
    assert torch.allclose(loss, expected, atol=1e-7)
    assert details["loss_main"] == pytest.approx(float(expected.detach()), abs=1e-7)


def test_corrected_model_defaults_are_safe_and_query_consistent():
    model = DAColor(torch.randint(0, 256, (6, 3), dtype=torch.float32), embedding_dim=5, hidden_dim=7)

    assert model.aux_feature_mode == "symmetric"
    assert model.aux_output_activation == "softplus"
    assert model.embedding_mode == "rgb_id"
    assert hasattr(model, "id_embeddings")


def test_softplus_auxiliary_head_has_gradient_when_preactivation_is_negative():
    torch.manual_seed(31)
    model = DAColor(torch.randint(0, 256, (4, 3), dtype=torch.float32), embedding_dim=5, hidden_dim=7)
    with torch.no_grad():
        model.aux_fc2.weight.zero_()
        model.aux_fc2.bias.fill_(-10.0)
    query_ids = torch.tensor([[0]])
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)
    query_embeds = model.color_embeddings(query_ids)

    prediction = model.predict_auxiliary(query_embeds, query_mask, torch.tensor([1]))
    prediction.sum().backward()

    assert prediction.item() > 0.0
    assert model.aux_fc2.bias.grad is not None
    assert model.aux_fc2.bias.grad.item() > 0.0


def test_full_candidate_loss_masks_other_true_scheme_colors():
    torch.manual_seed(37)
    model = DAColor(
        torch.randint(0, 256, (7, 3), dtype=torch.float32),
        embedding_dim=6,
        hidden_dim=9,
    )
    samples = [
        {
            "scheme_id": 5,
            "query_ids": [0],
            "target_id": 1,
            "scheme_color_ids": [0, 1, 2],
            "query_size": 1,
        }
    ]
    batch = pair_collate_fn(samples)
    query_vector, _ = model.encode_query(batch["query_ids"], batch["query_mask"])
    all_ids = torch.arange(7)
    logits = torch.matmul(
        F.normalize(query_vector, dim=-1),
        F.normalize(model.color_embeddings(all_ids), dim=-1).T,
    ) / model.temperature
    logits[:, [0, 2]] = -torch.inf
    expected = F.cross_entropy(logits, batch["target_ids"])

    actual, details = training_step(
        model,
        batch,
        ciede2000=torch.zeros((7, 7)),
        alpha=1.0,
        negative_size=2,
        aux_scale=torch.tensor(1.0),
        aux_scale_mode="max",
        aux_reduction="token_mean",
    )

    assert torch.allclose(actual, expected, atol=1e-7)
    assert details["loss_main"] == pytest.approx(float(expected.detach()), abs=1e-7)


def test_cooccurrence_prior_uses_training_scheme_counts_only():
    prior = build_log_cooccurrence_prior([[0, 1], [0, 1], [0, 2]], palette_size=3, epsilon=1.0)
    assert prior.shape == (3, 3)
    assert prior[0, 1] > prior[0, 2]
    assert prior[1, 0] > prior[1, 2]


def test_model_cooccurrence_prior_is_averaged_over_query_colors():
    palette = torch.tensor([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=torch.float32)
    prior = torch.tensor([[0.0, 2.0, 4.0], [2.0, 0.0, 8.0], [4.0, 8.0, 0.0]])
    model = DAColor(
        palette,
        embedding_dim=4,
        hidden_dim=6,
        prior_mode="cooccurrence",
        prior_weight=1.0,
        neural_weight=0.0,
        cooccurrence_prior=prior,
    )
    scores = model.score_candidates(torch.tensor([[0, 1]]), torch.tensor([[True, True]]))
    assert torch.allclose(scores[0], torch.tensor([1.0, 1.0, 6.0]))
