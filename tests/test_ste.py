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


def test_backward_gradient_is_identity() -> None:
    torch.manual_seed(1)
    s = torch.rand(10).abs().requires_grad_()
    ste_binarize(s).sum().backward()
    assert s.grad is not None
    assert torch.allclose(s.grad, torch.ones_like(s))
