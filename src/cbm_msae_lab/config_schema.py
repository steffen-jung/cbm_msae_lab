"""Structured Hydra configs for every axis of the pipeline.

Each dataclass here mirrors the constructor arguments of the class its
``_target_`` points to, so ``hydra.utils.instantiate(cfg.encoder)`` (etc.)
builds the object directly from the resolved config. Registering these with
Hydra's ``ConfigStore`` gives two things for free: (1) the YAML files under
``conf/`` are type- and typo-checked against these fields, and (2) Pylance can
see every config field's type when you write ``cfg.sae.dict_size`` in code.

The top-level ``Config`` dataclass is what ``scripts/train.py`` receives from
``@hydra.main``. Its sub-configs are intentionally typed as ``Any`` where a
group has multiple mutually-exclusive implementations (e.g. `encoder` can be
`ClipDinoiserEncoderConfig` or `DINOv3EncoderConfig`) -- Hydra picks the
concrete one at compose time based on ``conf/config.yaml``'s defaults list or
a CLI override like ``encoder=dinov3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# --------------------------------------------------------------------------
# Encoder configs (frozen backbones)
# --------------------------------------------------------------------------


@dataclass
class ClipDinoiserEncoderConfig:
    """CLIP-DINOiser ("DINOised CLIP"), ported from CFM. Always frozen."""

    _target_: str = "cbm_msae_lab.encoders.clip_dinoiser.ClipDinoiserEncoder"
    checkpoint_path: str = "checkpoints/clip_dinoiser/laion2b.pt"
    clip_backbone: str = "maskclip"  # selects cfm's maskclip.yaml (laion2b_s34b_b88k weights)
    vit_arch: str = "vit_base"
    vit_patch_size: int = 16
    feats_idx: int = -3  # which CLIP ViT block's dense features to hook for pooling
    enc_type_feats: str = "v"  # "v" = value vectors from self-attention, matches CFM
    gamma: float = 0.2  # DINO-correlation threshold used in the weighted pooling
    delta: float = 0.99
    output_dim: int = 512  # native channel dim of the pooled dense features (C_backbone)
    image_size: int = 224
    extract_attention: bool = False  # see EncoderOutput.attention; off by default, costs nothing when unused


@dataclass
class DINOv3EncoderConfig:
    """DINOv3 ViT, loaded through timm (auto-downloads, no gated Meta approval)."""

    _target_: str = "cbm_msae_lab.encoders.dinov3.DINOv3Encoder"
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    output_dim: int = 768  # ViT-B/16 embedding dim; must match model_name's actual width
    image_size: int = 224
    patch_size: int = 16
    extract_attention: bool = False  # see EncoderOutput.attention; off by default, costs nothing when unused


# --------------------------------------------------------------------------
# Projection config (the trainable, kernel-size-configurable head)
# --------------------------------------------------------------------------


@dataclass
class ProjectionConfig:
    """Maps frozen backbone features [B, C_backbone, H, W] -> [B, C_sae, H, W].

    ``kernel_size=1`` is mathematically identical to a per-token ``nn.Linear``
    (documented in ``projection.py``); ``kernel_size=3`` additionally mixes in
    each token's 3x3 spatial neighbourhood before the SAE ever sees it.
    """

    _target_: str = "cbm_msae_lab.projection.EncoderProjection"
    in_channels: int = "${encoder.output_dim}"  # type: ignore[assignment]  # OmegaConf interpolation
    out_channels: int = 512
    kernel_size: int = 3


# --------------------------------------------------------------------------
# Upsampler configs
# --------------------------------------------------------------------------


@dataclass
class IdentityUpsamplerConfig:
    """No-op: keeps the encoder's native spatial resolution. `stage` isn't a
    constructor argument here (`IdentityUpsampler` always hardcodes
    `stage="none"`) -- there is nothing to configure."""

    _target_: str = "cbm_msae_lab.upsampling.identity.IdentityUpsampler"


@dataclass
class BilinearUpsamplerConfig:
    _target_: str = "cbm_msae_lab.upsampling.bilinear.BilinearUpsampler"
    target_resolution: int = 64
    stage: str = "features"  # "features" (before the SAE) or "latents" (after SAE.encode)


@dataclass
class AnyUpUpsamplerConfig:
    """Image-guided universal feature upsampler (Wimmer et al., 2025)."""

    _target_: str = "cbm_msae_lab.upsampling.anyup.AnyUpUpsampler"
    target_resolution: int = 64
    stage: str = "features"
    torch_hub_repo: str = "wimmerth/anyup"
    torch_hub_model: str = "anyup_multi_backbone"
    use_natten: bool = False


# --------------------------------------------------------------------------
# SAE config
# --------------------------------------------------------------------------


@dataclass
class SAEConfig:
    """Hyperparameters for ``ConfigurableActivationSAE`` + ``ComposableLossTrainer``."""

    activation_dim: int = "${projection.out_channels}"  # type: ignore[assignment]
    dict_size: int = 8192
    k: int = 12
    group_fractions: list[float] = field(default_factory=lambda: [0.008, 0.03, 0.06, 0.12, 0.24, 0.542])
    activation: str = "relu"  # "relu" | "sigmoid"
    lr: float | None = None  # None -> dictionary_learning's 1/sqrt(dict_size) auto-scaling
    steps: int = 100_000  # total optimizer steps, for the LR warmup/decay schedule
    warmup_steps: int = 1_000
    decay_start: int | None = None
    threshold_beta: float = 0.999
    threshold_start_step: int = 1_000
    seed: int = 0


# --------------------------------------------------------------------------
# Loss configs -- one dataclass per Loss class, each independently toggleable
# via its `weight` (0.0 disables it and skips its compute() call entirely)
# --------------------------------------------------------------------------


@dataclass
class ReconstructionLossConfig:
    """CFM's original nested per-Matryoshka-group L2 reconstruction loss."""

    weight: float = 1.0


@dataclass
class AuxKLossConfig:
    """Dead-feature revival loss (ported unchanged from dictionary_learning)."""

    weight: float = 0.03125  # CFM's default auxk_alpha = 1/32


@dataclass
class ScaleSpatialLossConfig:
    """The user's own scale/spatial loss, ported from cfm_finegrained/scale_loss.py.

    L = sum_s w_s * L_s,
    L_s = (1/|G_s|) sum_g (1/d) || Pi^-1(z_g[:b_s]) - F_bar_g ||^2,
    b_s = ceil(dict_size * s^(-2*alpha)), clamped to [b_min, dict_size].
    """

    weight: float = 0.0
    scales: list[int] = field(default_factory=lambda: [1, 2, 7, 14])
    scale_weights: list[float] | None = None  # None -> uniform 1/len(scales)
    alpha: float = 1.0
    b_min: int = 1


@dataclass
class AttentionGroupingConfig:
    """Hyperparameters for the S2AE-style attention+spatial-proximity patch
    clustering (see `attention_grouping.py`), shared by `group_sparsity` and
    `exclusivity` whenever either uses `grouping="attention"` -- kept as one
    shared config (rather than duplicated per-loss) so the two losses can't
    silently disagree about which clustering they're grouping by.

    Defaults (`n_clusters=20`, `spatial_coeff=0.02`) match the values found in
    S2AE's own public repo (github.com/liaoweiduo/s2ae) and README.
    """

    n_clusters: int = 20
    spatial_coeff: float = 0.02  # alpha in `distance = d_attention * (d_spatial ** alpha)`


@dataclass
class GroupSparsityLossConfig:
    """S2AE group sparsity loss (arXiv:2607.08605, Eq. 9): L_gs = mean_g ||s^g||_1.

    `grouping="tile"` (default) uses s x s spatial tiles (`losses/tiling.py`),
    identical for every image -- simple and requires no caching.
    `grouping="attention"` instead uses S2AE's own approach: patches are
    clustered per-image by combining the encoder's self-attention with
    spatial proximity (`attention_grouping.py`). This requires the chosen
    encoder to have been run with `extract_attention=True` and a group-label
    cache built via `scripts/extract_attention_groups.py` beforehand.
    """

    weight: float = 0.0
    grouping: str = "tile"  # "tile" | "attention"
    tile_size: int = 2


@dataclass
class ExclusivityLossConfig:
    """S2AE exclusive sparsity loss (arXiv:2607.08605, Eq. 10):
    L_es = (1/N) sum_j (sum_g |s^g_j|)^2. See `GroupSparsityLossConfig` for
    what `grouping` selects between.
    """

    weight: float = 0.0
    grouping: str = "tile"  # "tile" | "attention"
    tile_size: int = 2


@dataclass
class LossConfig:
    reconstruction: ReconstructionLossConfig = field(default_factory=ReconstructionLossConfig)
    auxk: AuxKLossConfig = field(default_factory=AuxKLossConfig)
    scale_spatial: ScaleSpatialLossConfig = field(default_factory=ScaleSpatialLossConfig)
    group_sparsity: GroupSparsityLossConfig = field(default_factory=GroupSparsityLossConfig)
    exclusivity: ExclusivityLossConfig = field(default_factory=ExclusivityLossConfig)
    attention_grouping: AttentionGroupingConfig = field(default_factory=AttentionGroupingConfig)


# --------------------------------------------------------------------------
# Dataset configs
# --------------------------------------------------------------------------


@dataclass
class CUBDatasetConfig:
    _target_: str = "cbm_msae_lab.data.cub.CUBDataset"
    root: str = "/ceph/faroesch/datasets/CUB_200_2011"
    image_size: int = 224
    val_fraction: float = 0.1
    seed: int = 0


# --------------------------------------------------------------------------
# Training-loop config (batching, caching, eval cadence, wandb)
# --------------------------------------------------------------------------


@dataclass
class CacheConfig:
    """See caching.py. `mode="live"` never touches disk (encoder + projection
    both run live every step). `"cache_encoder"` caches only the frozen
    encoder's raw output (scripts/extract_raw_features.py, built once, fails
    loudly if missing) and still runs the trainable projection live, with
    gradients, every step -- the projection trains normally, just without
    paying for the encoder forward pass every epoch."""

    mode: str = "live"  # "live" | "cache_encoder"
    dir: str = "cache"


@dataclass
class EvalConfig:
    """Expensive metrics (Monosemanticity/Tversky-MS) need a full pass over many
    stimuli, so they run on their own cadence instead of every epoch."""

    every_n_epochs: int = 5
    enable_expensive_metrics: bool = False
    max_eval_samples: int = 2_000


@dataclass
class WandbConfig:
    project: str = "cbm-msae-lab"
    mode: str = "online"  # "online" | "offline" | "disabled"
    run_name: str | None = None


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 50
    device: str = "cuda"
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_every_n_epochs: int = 5
    log_every_n_steps: int = 50
    cache: CacheConfig = field(default_factory=CacheConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# --------------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Every field below is filled by `conf/config.yaml`'s defaults list (e.g.
    `encoder: clip_dinoiser`), not by a default here. Using `MISSING` (rather
    than a concrete value, or a `field(default_factory=...)`) matters: Hydra
    validates the defaults-list composition result against this dataclass as
    a schema, and a *concrete* default here (e.g. `IdentityUpsamplerConfig`)
    would make an `upsampler=anyup` override fail to validate -- Hydra would
    complain that `AnyUpUpsamplerConfig is not a subclass of
    IdentityUpsamplerConfig` instead of accepting the override. `MISSING`
    means "the defaults list must supply this," which is exactly the
    intended contract.
    """

    encoder: Any = MISSING
    projection: Any = MISSING
    upsampler: Any = MISSING
    sae: Any = MISSING
    loss: Any = MISSING
    dataset: Any = MISSING
    train: Any = MISSING
    seed: int = 0


def register_configs() -> None:
    """Register every structured config with Hydra's ConfigStore.

    Must run (import this module) before ``@hydra.main`` composes the config,
    so that e.g. ``encoder=dinov3`` on the CLI resolves to `DINOv3EncoderConfig`.

    Every schema is stored under a ``base_``-prefixed name, deliberately
    *not* matching its corresponding ``conf/**/*.yaml`` file's own name (e.g.
    ``base_clip_dinoiser`` vs. ``conf/encoder/clip_dinoiser.yaml``): each YAML
    file instead pulls its schema in explicitly via its own ``defaults:``
    list (``- base_clip_dinoiser``). Relying on same-name auto-matching
    instead is deprecated as of Hydra 1.1 (see
    https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/automatic_schema_matching);
    this is that migration's recommended fix, "Option 1: rename the
    structured config."
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    cs.store(group="encoder", name="base_clip_dinoiser", node=ClipDinoiserEncoderConfig)
    cs.store(group="encoder", name="base_dinov3", node=DINOv3EncoderConfig)

    cs.store(group="projection", name="base_k1", node=ProjectionConfig(kernel_size=1))
    cs.store(group="projection", name="base_k3", node=ProjectionConfig(kernel_size=3))

    cs.store(group="upsampler", name="base_none", node=IdentityUpsamplerConfig)
    cs.store(group="upsampler", name="base_bilinear", node=BilinearUpsamplerConfig)
    cs.store(group="upsampler", name="base_anyup", node=AnyUpUpsamplerConfig)

    cs.store(group="sae", name="base_matryoshka_batch_topk", node=SAEConfig)

    cs.store(group="loss", name="base_default", node=LossConfig)

    cs.store(group="dataset", name="base_cub", node=CUBDatasetConfig)

    cs.store(group="train", name="base_default", node=TrainConfig)
