"""The concept dictionary: a Matryoshka-BatchTopK SAE with a swappable activation.

Subclasses ``dictionary_learning``'s ``MatryoshkaBatchTopKSAE`` (see
https://github.com/saprmarks/dictionary_learning, the base CFM itself
depends on) rather than reimplementing it -- per the user's explicit choice,
the proven encoder/decoder/Matryoshka-grouping logic stays exactly as-is; only
the encoder nonlinearity is made configurable, following the same pattern
``cfm_finegrained``'s ``ActivationMatryoshkaSAE`` already validated.

Note on the encoder/decoder "shape": ``W_enc``/``W_dec`` here are plain
``nn.Parameter`` matrices used as a manual linear map (``x @ W_enc + b_enc``),
not an ``nn.Conv2d`` -- the SAE always operates on flattened per-token feature
vectors ``[N, activation_dim]``. Any spatial structure (which patch a token
came from) is handled entirely outside the SAE, by ``ConceptPipeline``
flattening/reshaping around it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from torch import Tensor

from cbm_msae_lab.sae.activations import (
    ACTIVATION_BY_ID,
    ACTIVATION_IDS,
    Activation,
    activation_fn,
)


class ConfigurableActivationSAE(MatryoshkaBatchTopKSAE):
    """``MatryoshkaBatchTopKSAE`` with ReLU swapped for a configurable nonlinearity.

    ``encode`` is copied from the parent class rather than wrapped, because
    the parent applies the nonlinearity and BatchTopK selection in one
    expression with no seam to override just the nonlinearity.
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        group_sizes: list[int],
        activation: Activation = "relu",
    ) -> None:
        super().__init__(activation_dim, dict_size, k, group_sizes)
        if activation not in ACTIVATION_IDS:
            raise ValueError(f"unknown activation {activation!r}; expected one of {list(ACTIVATION_IDS)}")
        self.activation: Activation = activation
        self._act: Callable[[Tensor], Tensor] = activation_fn(activation)
        self.register_buffer("activation_id", torch.tensor(ACTIVATION_IDS[activation], dtype=torch.int))

    def encode(self, x: Tensor, return_active: bool = False, use_threshold: bool = True):
        """x: [N, activation_dim] -> latents [N, dict_size] (+ optionally active-mask, pre-topk activations)."""
        post_act_BF = self._act((x - self.b_dec) @ self.W_enc + self.b_enc)  # [N, dict_size]

        if use_threshold:
            encoded_acts_BF = post_act_BF * (post_act_BF > self.threshold)
        else:
            flattened_acts = post_act_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)
            encoded_acts_BF = (
                torch.zeros_like(post_act_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_act_BF.shape)
            )

        # Matryoshka nesting: zero out every feature beyond the currently-active groups.
        max_act_index = self.group_indices[self.active_groups]
        encoded_acts_BF[:, max_act_index:] = 0

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_act_BF
        return encoded_acts_BF

    @classmethod
    def from_state_dict(cls, state_dict: dict, k: int | None = None, device=None) -> ConfigurableActivationSAE:
        """Reconstructs an instance purely from a `state_dict` -- every shape
        (activation_dim, dict_size, group_sizes) and the activation function
        are read back out of it, so no separate config is needed to load a
        checkpoint correctly (used directly by `checkpointing.load_for_reproduction`)."""
        activation_dim, dict_size = state_dict["W_enc"].shape
        if k is None:
            k = state_dict["k"].item()
        group_sizes = state_dict["group_sizes"].tolist()
        activation = (
            ACTIVATION_BY_ID[int(state_dict["activation_id"].item())] if "activation_id" in state_dict else "relu"
        )

        autoencoder = cls(
            activation_dim,
            dict_size,
            k=k,
            group_sizes=group_sizes,
            activation=activation,
        )
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

    @classmethod
    def from_pretrained(cls, path, k=None, device=None, **kwargs) -> ConfigurableActivationSAE:
        state_dict = torch.load(path, map_location=device)
        return cls.from_state_dict(state_dict, k=k, device=device)
