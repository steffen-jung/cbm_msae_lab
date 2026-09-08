"""Dead-feature revival loss.

Thin wrapper around ``ComposableLossTrainer.get_auxiliary_loss`` (ported
unchanged from ``dictionary_learning``): encourages dictionary features that
haven't fired in a long time to reconstruct the current residual, which is
what prevents them from staying permanently dead. The actual bookkeeping
(``num_tokens_since_fired``) is stateful across steps, so it lives on the
trainer rather than on this stateless ``Loss`` wrapper.
"""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.losses.base import Loss, LossContext


class AuxKLoss(Loss):
    def compute(self, ctx: LossContext) -> Tensor:
        residual = (ctx.x - ctx.x_hat).detach()  # [N, activation_dim]
        return ctx.trainer.get_auxiliary_loss(residual, ctx.post_act)
