"""Unit tests for the published monosemanticity metrics.

All are checked against a literal transcription of the paper equations (or, for
`PachMonosemanticityScore`, of the official reference code), computed by brute force over all
N(N-1) ordered pairs -- the closed-form / bit-packed implementations exist only for speed, so a
naive reference is the right oracle for them.
"""

from __future__ import annotations

import numpy as np
import torch

from cbm_msae_lab.metrics.monosemanticity import MonosemanticityScore, PachMonosemanticityScore
from cbm_msae_lab.metrics.tversky_ms import TverskyMS


def _brute_force_ms(acts: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Pach et al. Eq. 7-9, transcribed literally: minmax over the N stimuli,
    r_nm = w_n * w_m, and MS^k = (1 / N(N-1)) * sum_{n != m} r^k_nm * s_nm."""
    similarity = emb @ emb.t()  # [N, N]
    n_stimuli = acts.shape[0]
    scores = []
    for k in range(acts.shape[1]):
        a = acts[:, k]
        span = a.max() - a.min()
        w = (a - a.min()) / span if span > 0 else torch.ones_like(a)  # Eq. 7
        total = sum(
            w[n] * w[m] * similarity[n, m]  # Eq. 8 x s_nm
            for n in range(n_stimuli)
            for m in range(n_stimuli)
            if n != m
        )
        scores.append(total / (n_stimuli * (n_stimuli - 1)))  # Eq. 9
    return torch.tensor(scores)


def _brute_force_ms_pach(acts: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """`ExplainableML/sae-for-vlm/metric.py::main()`, transcribed literally: same Eq. 7-8
    weights and per-pair similarity as `_brute_force_ms`, but MS^k = sum(r^k_nm * s_nm) /
    sum(r^k_nm) -- a weighted average, not a division by the fixed pair count."""
    similarity = emb @ emb.t()  # [N, N]
    n_stimuli = acts.shape[0]
    scores = []
    for k in range(acts.shape[1]):
        a = acts[:, k]
        span = a.max() - a.min()
        w = (a - a.min()) / span if span > 0 else torch.ones_like(a)
        weighted_sum = sum(
            w[n] * w[m] * similarity[n, m] for n in range(n_stimuli) for m in range(n_stimuli) if n != m
        )
        weight_sum = sum(w[n] * w[m] for n in range(n_stimuli) for m in range(n_stimuli) if n != m)
        scores.append(weighted_sum / weight_sum if weight_sum != 0 else torch.tensor(float("nan")))
    return torch.tensor(scores)


def _random_embeddings(n: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1)


def test_ms_closed_form_matches_the_published_equation() -> None:
    """The O(N*D) identity ||sum_n w_n e_n||^2 - sum_n w_n^2 must reproduce the
    O(N^2) double sum exactly -- that identity is the whole implementation."""
    embeddings = _random_embeddings(40, 12, seed=0)
    generator = torch.Generator().manual_seed(1)
    activations = torch.rand(40, 5, generator=generator)
    activations[:35, 0] = 0.0  # a sparse concept, firing on 5 of 40 stimuli

    metric = MonosemanticityScore(dict_size=5)
    metric.update(activations, embeddings)

    assert torch.allclose(metric.compute(), _brute_force_ms(activations, embeddings), atol=1e-5)


def test_ms_of_a_perfectly_monosemantic_concept_approaches_its_cluster_similarity() -> None:
    """A concept that fires on one tight cluster and nowhere else: the only pairs
    carrying weight are within-cluster, so MS is that cluster's similarity scaled
    by the pair fraction the concept covers (Eq. 9 divides by *all* N(N-1) pairs)."""
    n_stimuli, cluster = 30, 6
    direction = torch.nn.functional.normalize(torch.ones(8), dim=0)
    embeddings = _random_embeddings(n_stimuli, 8, seed=2)
    embeddings[:cluster] = direction  # an exactly-identical cluster: s_nm = 1

    activations = torch.zeros(n_stimuli, 1)
    activations[:cluster, 0] = 1.0

    metric = MonosemanticityScore(dict_size=1)
    metric.update(activations, embeddings)
    expected = cluster * (cluster - 1) / (n_stimuli * (n_stimuli - 1))
    assert abs(metric.compute().item() - expected) < 1e-5


def test_ms_is_nan_for_a_constant_concept() -> None:
    """Eq. 7 is 0/0 when a concept never varies. Returning `nan` keeps such a
    concept out of `.nanmean()`; the alternative fallback w = 1 would score it at
    the *dataset's* mean pairwise similarity, which on a single-domain set is
    higher than any live concept reaches and would dominate the dictionary mean."""
    n_stimuli = 25
    embeddings = _random_embeddings(n_stimuli, 8, seed=3)
    activations = torch.zeros(n_stimuli, 2)
    activations[:6, 1] = 1.0  # concept 1 is alive, concept 0 never fires

    metric = MonosemanticityScore(dict_size=2)
    metric.update(activations, embeddings)
    scores = metric.compute()

    assert torch.isnan(scores[0])
    assert torch.isfinite(scores[1])
    assert scores.nanmean() == scores[1]
    assert metric.dead_fraction() == 0.5


def test_ms_pach_closed_form_matches_the_official_repo_formula() -> None:
    """The closed-form weighted-average identity must reproduce the O(N^2) double sum
    of the official reference code exactly."""
    embeddings = _random_embeddings(40, 12, seed=0)
    generator = torch.Generator().manual_seed(1)
    activations = torch.rand(40, 5, generator=generator)
    activations[:35, 0] = 0.0  # a sparse concept, firing on 5 of 40 stimuli

    metric = PachMonosemanticityScore(dict_size=5)
    metric.update(activations, embeddings)

    assert torch.allclose(metric.compute(), _brute_force_ms_pach(activations, embeddings), atol=1e-5)


def test_ms_pach_of_a_perfectly_monosemantic_concept_equals_its_cluster_similarity() -> None:
    """Unlike `MonosemanticityScore` (which scales this down by the pair fraction the concept
    covers), the Pach weighted-average score of a concept firing on one tight cluster and
    nowhere else is exactly that cluster's own similarity -- undiluted by how many of the
    N(N-1) pairs the concept actually touches."""
    n_stimuli, cluster = 30, 6
    direction = torch.nn.functional.normalize(torch.ones(8), dim=0)
    embeddings = _random_embeddings(n_stimuli, 8, seed=2)
    embeddings[:cluster] = direction  # an exactly-identical cluster: s_nm = 1

    activations = torch.zeros(n_stimuli, 1)
    activations[:cluster, 0] = 1.0

    metric = PachMonosemanticityScore(dict_size=1)
    metric.update(activations, embeddings)
    assert abs(metric.compute().item() - 1.0) < 1e-5


def test_ms_pach_is_nan_for_a_constant_concept() -> None:
    """Same nan handling as `MonosemanticityScore` for a concept that never varies."""
    n_stimuli = 25
    embeddings = _random_embeddings(n_stimuli, 8, seed=3)
    activations = torch.zeros(n_stimuli, 2)
    activations[:6, 1] = 1.0  # concept 1 is alive, concept 0 never fires

    metric = PachMonosemanticityScore(dict_size=2)
    metric.update(activations, embeddings)
    scores = metric.compute()

    assert torch.isnan(scores[0])
    assert torch.isfinite(scores[1])
    assert scores.nanmean() == scores[1]
    assert metric.dead_fraction() == 0.5


def _brute_force_jaccard(binary: np.ndarray, i: int, j: int) -> float:
    """Filus & Pokucinski Eq. 1 with alpha = beta = 1, on raw boolean rows."""
    intersection = float((binary[i] & binary[j]).sum())
    union = float((binary[i] | binary[j]).sum())
    return intersection / union if union > 0 else 0.0


def test_tversky_bit_packing_reproduces_a_plain_jaccard() -> None:
    """The uint64 packing + popcount path must agree with set arithmetic on the
    same boolean rows, for every pair -- including rows that cross a word boundary."""
    from cbm_msae_lab.metrics.tversky_ms import _pack_bitset, _tversky_index

    generator = np.random.default_rng(0)
    binary = generator.random((12, 150)) > 0.7  # 150 latents -> 3 uint64 words, last one partial
    packed = _pack_bitset(binary)

    rows = np.arange(12)
    for i in rows:
        got = _tversky_index(packed, np.full(12, i), rows, alpha=1.0, beta=1.0)
        expected = [_brute_force_jaccard(binary, i, j) for j in rows]
        assert np.allclose(got, expected)


def test_tversky_ms_is_one_when_every_active_stimulus_has_the_same_latent_pattern() -> None:
    """Identical binarized rows have Jaccard 1 with each other, so a dictionary
    whose latents all fire together scores exactly 1."""
    activations = torch.zeros(20, 4)
    activations[:8, :] = 1.0  # the same 8 stimuli activate all 4 latents

    metric = TverskyMS(dict_size=4, max_pairs=50)
    metric.update(activations)
    assert metric.compute().item() == 1.0


def test_tversky_ms_ignores_latents_that_never_activate() -> None:
    """A dead latent has an empty active set; the paper scores it 0 and reports
    over the non-zero values, so it must not drag the model-level mean down."""
    activations = torch.zeros(20, 3)
    activations[:8, 0] = 1.0  # only latent 0 is ever active

    metric = TverskyMS(dict_size=3, max_pairs=50)
    metric.update(activations)
    assert metric.compute().item() == 1.0
