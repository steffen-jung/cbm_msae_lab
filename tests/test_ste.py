from __future__ import annotations

import torch

from cbm_msae_lab.losses.ste import ste_binarize


def test_forward_value_is_hard_binary_gate() -> None:
    torch.manual_seed(0)
    s = torch.rand(5, 7).abs()  # >= 0, some exactly 0 after the next line
    s[0, 0] = 0.0
    out = ste_binarize(s)
    expected = (s > 0).to(s.dtype)
    # hard + s - s.detach() is only numerically (not bit-exactly) equal to
    # `hard`, since floating-point addition/subtraction isn't associative.
    assert torch.allclose(out, expected, atol=1e-6)


def test_zero_maps_to_zero_gate() -> None:
    s = torch.zeros(4)
    assert torch.equal(ste_binarize(s), torch.zeros(4))


def test_backward_gradient_saturates_for_large_s() -> None:
    """A plain identity-passthrough STE (`hard + s - s.detach()`) gives
    `d/ds == 1` everywhere -- mathematically indistinguishable from raw L1 on
    non-negative `s`, which is exactly what let the group-sparsity/exclusivity
    losses keep shrinking activation magnitude toward zero (see module
    docstring). The tanh-surrogate version must NOT do that: gradient near
    the decision boundary (small s) should track the identity-STE case
    closely, but clearly active latents (large s) must get near-zero
    gradient so this loss stops pushing them down."""
    s_small = torch.tensor([1e-3, 1e-2], requires_grad=True)
    ste_binarize(s_small).sum().backward()
    assert s_small.grad is not None
    assert torch.all(s_small.grad > 0.9)  # near-identity close to the boundary

    s_large = torch.tensor([5.0, 10.0], requires_grad=True)
    ste_binarize(s_large).sum().backward()
    assert s_large.grad is not None
    assert torch.all(s_large.grad < 1e-3)  # saturated: no more shrink pressure


def test_backward_gradient_matches_tanh_derivative() -> None:
    torch.manual_seed(1)
    s = torch.rand(10).abs().requires_grad_()
    ste_binarize(s).sum().backward()
    assert s.grad is not None
    expected = 1 - torch.tanh(s.detach()) ** 2
    assert torch.allclose(s.grad, expected, atol=1e-6)
