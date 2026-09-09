# cbm_msae_lab

A flexible research codebase for training concept sparse autoencoders (a
Matryoshka BatchTopK SAE, following [CFM](https://arxiv.org/abs/2601.13798))
on vision-encoder features, built for experimenting freely along several
axes that CFM itself hardcodes:

- **Encoder**: frozen "DINOised CLIP" (CLIP-DINOiser, ported from CFM) or DINOv3 (via `timm`).
- **Projection**: a trainable Conv2d between the encoder and the SAE, with a configurable kernel size (`1` = per-token linear, `3` = spatially mixed).
- **Upsampler** (optional): none, bilinear, or [AnyUp](https://github.com/wimmerth/anyup) -- applied either to the features *before* the SAE (changing the SAE's own training resolution) or to the SAE's latents *after* encoding (for higher-resolution concept maps only).
- **SAE activation**: ReLU (default) or Sigmoid.
- **Losses** (each independently toggleable via its weight): CFM's nested Matryoshka reconstruction loss, the dead-feature-revival auxiliary loss, a custom scale/spatial loss, and the two S2AE structured-sparsity losses (group sparsity, exclusivity) -- the latter two can group patches either by a fixed spatial tile (default) or, optionally, by S2AE's own attention+spatial-proximity clustering (`loss.group_sparsity.grouping=attention`, see "Attention-based patch grouping" below).
- **Dataset**: CUB-200-2011 to start, via a small registry so more can be added later.

See `/home/faroesch/.claude/plans/ich-m-chte-eine-neue-zippy-quiche.md` for
the full design rationale and provenance of every ported piece (CFM,
`cfm_finegrained`, `us-spatial-cbm`).

## Setup

```bash
uv sync
uv run scripts/download_checkpoints.py   # copies the CLIP-DINOiser checkpoint from a CFM checkout
```

DINOv3 (via `timm`) and AnyUp (via `torch.hub`) auto-download on first use --
no separate step needed for those.

## Training

```bash
# CFM-equivalent baseline: CLIP-DINOiser, 3x3 projection, no upsampling, ReLU, reconstruction+auxk only.
uv run scripts/train.py

# Swap the encoder, activation, and enable the scale/spatial loss:
uv run scripts/train.py encoder=dinov3 sae.activation=sigmoid loss.scale_spatial.weight=1.0

# Upsample features to 64x64 with AnyUp before the SAE ever sees them:
uv run scripts/train.py upsampler=anyup upsampler.target_resolution=64

# Pre-build a cache of the frozen encoder's raw output once, then skip the
# expensive encoder forward every epoch. The projection still trains live,
# with gradients, on top of the cache:
uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val
uv run scripts/train.py train.cache.mode=cache_encoder
```

Example: comparing CFM's own reconstruction loss against the scale/spatial
loss and the S2AE losses, everything else fixed (1x1 projection, no
upsampling, ReLU). Note that `reconstruction` and `scale_spatial` are
*alternatives*, not additive -- `scale_spatial`'s finest scale (`s=1`) is
already the same full-patch reconstruction as `reconstruction`'s own finest
Matryoshka-group term (see `scale_spatial.py`'s docstring), so running both
at once would double-weight it. `group_sparsity`/`exclusivity`, in contrast,
are pure regularizers and stack fine on top of the standard reconstruction:

```bash
uv run scripts/train.py projection=k1                                          # CFM vanilla
uv run scripts/train.py projection=k1 loss.reconstruction.weight=0.0 \
  loss.scale_spatial.weight=1.0                                                # spatial loss instead
uv run scripts/extract_attention_groups.py --split train                       # once, before the S2AE run
uv run scripts/train.py projection=k1 \
  loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.1 \
  loss.exclusivity.grouping=attention loss.exclusivity.weight=0.1              # + S2AE losses
```

Every checkpoint (`outputs/checkpoints/epoch_XXXX.pt` by default) is a single
file containing the projection + SAE weights, both optimizers' states, and
the *entire* resolved Hydra config -- `checkpointing.load_for_reproduction`
rebuilds the exact pipeline from that one file.

The training loop's progress bar (and wandb, every `train.log_every_n_steps`)
also reports whether the DataLoader is the bottleneck: `train/dataloader_wait_ms`
(time spent waiting for the next batch) vs. `train/dataloader_compute_ms` (time
spent on the actual training step) -- see `timing.py::TimedLoader`. Every
active loss term is logged individually too (`train/loss_reconstruction`,
`train/loss_scale_spatial`, ...), not just the weighted total.

## Downstream task accuracy (post-hoc, after training)

`scripts/evaluate_task_accuracy.py` measures how much of CUB's 200-way
classification signal survives the SAE's concept bottleneck, by training a
linear probe (CFM paper's own protocol, App. B.5: AdamW, learning-rate and
L1-sparsity sweep, best checkpoint by held-out accuracy -- see `probing.py`)
on three per-image-pooled representations extracted from one checkpoint:
the raw projected encoder feature (`f_plus`, the reconstruction target, an
un-bottlenecked ceiling), the SAE's dense activation (`a`), and its
inference-time thresholded activation (`z`, what CFM calls its "concepts").

```bash
uv run scripts/evaluate_task_accuracy.py --checkpoint outputs/checkpoints/epoch_0049.pt
```

Only ever run this on a *finished* checkpoint -- it reads the checkpoint's own
stored config to rebuild the exact encoder/projection/SAE, so it never needs
its own Hydra config. Writes a JSON report (top-1/top-5 per representation,
per seed) to `outputs/task_accuracy/<checkpoint-stem>.json`.

## Attention-based patch grouping (optional)

Both encoders can optionally expose their own self-attention
(`EncoderOutput.attention`, averaged over every head and transformer block,
`encoder.extract_attention=true`). This feeds an adaptation of the S2AE paper's
(arXiv:2607.08605) patch-clustering pipeline -- symmetrize the attention,
convert it to a distance, combine with spatial proximity, and run
`AgglomerativeClustering` -- as an alternative to the default fixed-tile
grouping for `group_sparsity`/`exclusivity`. See `attention_grouping.py`'s
module docstring for exactly what differs from S2AE's own approach (they
cluster using an LLM decoder's attention; this codebase substitutes the ViT
encoder's own self-attention, since there's no LLM here).

Since group labels depend only on the frozen encoder (never the trainable
projection or SAE), they're always precomputed and cached, never clustered
on the fly inside the training loop:

```bash
uv run scripts/extract_attention_groups.py
uv run scripts/train.py loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.3
```

## Tests

```bash
uv run pytest tests/
```

Fast tests use a dummy encoder (no downloads). A few integration tests
against the real encoders/upsampler are skipped unless their
checkpoint/network dependency is actually available.

## Notes

- No `sbatch`/Slurm submission happens from this codebase -- training is
  always invoked directly, left entirely to you.
- This repo's git history is yours to manage: nothing here commits on your
  behalf.
- `cache/`, `checkpoints/`, and `outputs/` are symlinks into
  `/ceph/faroesch/cbm_msae_lab_*` (Home has a much smaller quota than `/ceph`)
  -- created once, transparent to every path in the config, nothing to set up
  per-run.
