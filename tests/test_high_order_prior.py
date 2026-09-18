import numpy as np

from dacolor.high_order_prior import HighOrderCooccurrencePrior


def test_exact_query_component_prefers_observed_three_color_context():
    scorer = HighOrderCooccurrencePrior.fit(
        [[0, 1, 2], [0, 1, 2], [0, 1, 3], [0, 3, 4]],
        palette_size=5,
    )
    _, exact, _, eligible = scorer.components([0, 1])
    assert exact[2] > exact[3]
    assert not eligible[0]
    assert not eligible[1]


def test_popularity_penalty_can_demote_ubiquitous_candidate():
    scorer = HighOrderCooccurrencePrior.fit(
        [[0, 1], [0, 1], [0, 2], [2, 3], [2, 4]],
        palette_size=5,
    )
    no_penalty = scorer.score([0], exact_weight=0.0, popularity_penalty=0.0)
    with_penalty = scorer.score([0], exact_weight=0.0, popularity_penalty=1.0)
    assert np.isneginf(with_penalty[0])
    assert with_penalty[1] - with_penalty[2] > no_penalty[1] - no_penalty[2]


def test_hierarchical_middle_component_uses_two_color_subsets():
    schemes = [[0, 1, 2, 3], [0, 1, 4], [0, 2, 4], [1, 2, 5]]
    scorer = HighOrderCooccurrencePrior.fit(schemes, palette_size=6)
    pairwise, middle, exact, popularity, eligible = scorer.hierarchical_components([0, 1, 2])
    assert pairwise.shape == middle.shape == exact.shape == popularity.shape == (6,)
    assert eligible.tolist() == [False, False, False, True, True, True]
    assert np.isfinite(middle[eligible]).all()
