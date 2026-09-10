# cbm_msae_lab

A flexible research codebase for training concept sparse autoencoders (a
Matryoshka BatchTopK SAE, following [CFM](https://arxiv.org/abs/2601.13798))
on vision-encoder features, built for experimenting freely along several
axes that CFM itself hardcodes:

- **Encoder**: frozen "DINOised CLIP" (CLIP-DINOiser, ported from CFM) or DINOv3 (via `timm`). Its patch features go straight into the SAE -- nothing trainable sits in between, so the SAE's reconstruction target is fixed, exactly as in CFM. `sae.activation_dim` therefore follows `encoder.output_dim` (512 / 768).
- **Upsampler** (optional): none, bilinear, or [AnyUp](https://github.com/wimmerth/anyup) -- applied either to the features *before* the SAE (changing the SAE's own training resolution) or to the SAE's latents *after* encoding (for higher-resolution concept maps only).
- **SAE activation**: ReLU (default) or Sigmoid.
- **Losses** (each independently toggleable via its weight): CFM's nested Matryoshka reconstruction loss, the dead-feature-revival auxiliary loss, a custom scale/spatial loss, and the two S2AE structured-sparsity losses (group sparsity, exclusivity) -- the latter two can group patches by a fixed spatial tile (default), by S2AE's own attention+spatial-proximity clustering (`grouping=attention`), or by raw encoder-feature similarity + spatial proximity (`grouping=feature`), see "Patch grouping for group-sparsity/exclusivity" below.
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
# CFM-equivalent baseline: CLIP-DINOiser, no upsampling, ReLU, reconstruction+auxk only.
uv run scripts/train.py

# Swap the encoder, activation, and enable the scale/spatial loss:
uv run scripts/train.py encoder=dinov3 sae.activation=sigmoid loss.scale_spatial.weight=1.0

# Upsample features to 64x64 with AnyUp before the SAE ever sees them:
uv run scripts/train.py upsampler=anyup upsampler.target_resolution=64

# Pre-build a cache of the frozen encoder's raw output once, then skip the
# expensive encoder forward every epoch. Nothing trainable sits between the
# encoder and the SAE, so the cache holds exactly what the SAE consumes:
uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val
uv run scripts/train.py train.cache.mode=cache_encoder
```

Example: comparing CFM's own reconstruction loss against the scale/spatial
loss and the S2AE losses, everything else fixed (no upsampling, ReLU). Note that `reconstruction` and `scale_spatial` are
*alternatives*, not additive -- `scale_spatial`'s finest scale (`s=1`) is
already the same full-patch reconstruction as `reconstruction`'s own finest
Matryoshka-group term (see `scale_spatial.py`'s docstring), so running both
at once would double-weight it. `group_sparsity`/`exclusivity`, in contrast,
are pure regularizers and stack fine on top of the standard reconstruction:

```bash
uv run scripts/train.py                                                        # CFM vanilla
uv run scripts/train.py loss.reconstruction.weight=0.0 \
  loss.scale_spatial.weight=1.0                                                # spatial loss instead
uv run scripts/extract_attention_groups.py --split train                       # once, before the S2AE run
uv run scripts/train.py \
  loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.1 \
  loss.exclusivity.grouping=attention loss.exclusivity.weight=0.1              # + S2AE losses
```

Every checkpoint (`outputs/checkpoints/epoch_XXXX.pt` by default) is a single
file containing the SAE weights, the optimizer state, and
the *entire* resolved Hydra config -- `checkpointing.load_for_reproduction`
rebuilds the exact pipeline from that one file.

The training loop's progress bar (and wandb, every `train.log_every_n_steps`)
also reports whether the DataLoader is the bottleneck: `train/dataloader_wait_ms`
(time spent waiting for the next batch) vs. `train/dataloader_compute_ms` (time
spent on the actual training step) -- see `timing.py::TimedLoader`. Every
active loss term is logged individually too (`train/loss_reconstruction`,
`train/loss_scale_spatial`, ...), not just the weighted total.

`train.epochs=200` is a hard cap, not a target: `train.early_stopping` (on by
default) stops once `train.early_stopping.monitor` (`val/fvu` by default)
hasn't improved by >= `min_delta` (a relative fraction) for `patience`
consecutive eval checks -- checks happen every
`train.eval.every_n_epochs` epochs, so `patience` counts *checks*, not
epochs. See `early_stopping.py::EarlyStopping`. Disable with
`train.early_stopping.enabled=false` to always train the full `epochs` cap.

When `train.eval.enable_expensive_metrics=true` (the default), every eval
also computes Monosemanticity Score and Tversky-MS. MS needs embeddings from
a genuinely *external* encoder (see `metrics/monosemanticity.py`'s
docstring) -- not the same encoder the SAE is trained to reconstruct, which
would just measure how well the SAE learned to copy that encoder's own
geometry. `external_encoder.py` always pairs the training encoder with the
*other* one this codebase supports: CLIP-DINOiser's external encoder is
DINOv3 and vice versa, each loaded with its own default config, frozen.

Whenever the monitored metric improves, `<checkpoint_dir>/best.pt` is
overwritten (in
addition to the regular `epoch_XXXX.pt` cadence) -- always the checkpoint you
want for downstream evaluation. Every checkpoint also stores the wandb run id
it was written from (`torch.load(path)["wandb_run_id"]`), to trace a
checkpoint back to its run. Once training ends (either the epoch cap or early
stopping), `scripts/evaluate_task_accuracy.py` is run automatically on
`best.pt` -- no separate manual step needed; see the next section for what it
reports.

## Downstream task accuracy (post-hoc, after training)

Runs automatically on `best.pt` at the end of every `scripts/train.py` call
(see above) -- re-run manually only to evaluate a different checkpoint, or
with different seeds/probe settings.

`scripts/evaluate_task_accuracy.py` measures how much of CUB's 200-way
classification signal survives the SAE's concept bottleneck, by training a
linear probe (CFM paper's own protocol, App. B.5: AdamW, learning-rate and
L1-sparsity sweep, best checkpoint by held-out accuracy -- see `probing.py`)
on three per-image-pooled representations extracted from one checkpoint:
the raw projected encoder feature (`f_plus`, the reconstruction target, an
un-bottlenecked ceiling), the SAE's dense activation (`a`), and its
inference-time thresholded activation (`z`, what CFM calls its "concepts").

```bash
uv run scripts/evaluate_task_accuracy.py --checkpoint outputs/checkpoints/best.pt
```

Only ever run this on a *finished* checkpoint -- it reads the checkpoint's own
stored config to rebuild the exact encoder/SAE, so it never needs
its own Hydra config. Writes a JSON report (top-1/top-5 per representation,
per seed) to `outputs/task_accuracy/<checkpoint-dir-name>_<checkpoint-stem>.json`
(e.g. `cub_spatial_best.json`) -- prefixed with the checkpoint's parent dir
name since every run names its best checkpoint `best.pt`, and a bare
`<checkpoint-stem>.json` would let different runs overwrite each other's
report. Override with `--out` for a different path.

## Reconstruction/concept-quality evaluation (post-hoc, after training)

`scripts/evaluate_checkpoint.py` re-runs the same metrics
`scripts/train.py::run_eval` computes during training -- reconstruction
MSE/cosine-sim, FVU, Monosemanticity Score, Tversky-MS -- standalone, on any
checkpoint and any split, without re-running training:

```bash
uv run scripts/evaluate_checkpoint.py --checkpoint outputs/checkpoints/best.pt
uv run scripts/evaluate_checkpoint.py --checkpoint outputs/checkpoints/best.pt --split test
```

Like `evaluate_task_accuracy.py`, it needs no Hydra config of its own -- the
checkpoint's own stored config rebuilds the exact pipeline. `--skip-expensive`
skips MS/Tversky-MS (and the external-encoder load they need) if you only
want the cheap reconstruction/FVU numbers. Writes a JSON report to
`outputs/eval/<checkpoint-stem>_<split>.json`.

## Patch grouping for group-sparsity/exclusivity (optional)

Two clustering-based alternatives to the default fixed-tile grouping, both
implemented in `attention_grouping.py`:

- `grouping=attention`: both encoders can optionally expose their own
  self-attention (`EncoderOutput.attention`, averaged over every head and
  transformer block, `encoder.extract_attention=true`). This feeds an
  adaptation of the S2AE paper's (arXiv:2607.08605) patch-clustering
  pipeline -- symmetrize the attention, convert it to a distance, combine
  with spatial proximity, and run `AgglomerativeClustering`
  (`cluster_patches`). See `attention_grouping.py`'s module docstring for
  exactly what differs from S2AE's own approach (they cluster using an LLM
  decoder's attention; this codebase substitutes the ViT encoder's own
  self-attention, since there's no LLM here).
- `grouping=feature`: same clustering machinery, but similarity comes from
  the encoder's raw per-patch features (cosine similarity) instead of
  attention (`cluster_patches_by_features`). Added after attention-based
  clusters turned out not to be granular enough on the object itself in
  practice (see `notebooks/attention_maps.ipynb`'s visual comparison) --
  worth trying both and comparing.

Since group labels depend only on the frozen encoder (never the SAE),
they're always precomputed and cached, never clustered
on the fly inside the training loop. The two methods use separate cache
slots (`compute_group_cache_key`'s `method` argument), so building one
doesn't invalidate or overwrite the other:

```bash
uv run scripts/extract_attention_groups.py --method attention
uv run scripts/train.py loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.3

uv run scripts/extract_attention_groups.py --method feature
uv run scripts/train.py loss.group_sparsity.grouping=feature loss.group_sparsity.weight=0.3
```

Every grouping consumer -- `group_sparsity`, `exclusivity`,
`participation_ratio` and `group_topk` -- shares one cached `group_labels`
tensor per training step, so any of them that use clustered grouping (not
`"tile"`) must use the *same* method; `scripts/train.py` raises if they
disagree. A consumer that is switched off (`weight: 0.0`, or
`group_topk.enabled: false`) is ignored by that check and pulls in no cache
requirement.

Both cache builders (`extract_raw_features.py`, `extract_attention_groups.py`)
skip an already-complete cache by default, so sbatch jobs that build the same
cache can be submitted together; `--overwrite` forces a rebuild.

## Structural mechanisms: Group-TopK and the participation-ratio loss

Two additions on top of the existing BatchTopK sparsity, aimed at making
features *spatially* specialised rather than merely few:

- **Group-TopK** (`sae/group_topk.py`) -- a hard selection rule, not a loss.
  Within each region and each Matryoshka block, only the `k_l` features with the
  largest mean pre-activation over that region's patches may be active. It
  contributes no gradient and therefore no shrinkage pressure. It *intersects*
  with BatchTopK rather than replacing it: BatchTopK decides which activations
  survive at all, Group-TopK which features a region may use, and Group-TopK can
  never revive what BatchTopK dropped. `k_group` is one total budget, split
  across the Matryoshka levels in proportion to their size.
  **Applied during training only** -- inference keeps the learned BatchTopK
  threshold and needs no group labels, so only the `train` split needs a cache.
  That is a deliberate train/inference asymmetry and belongs in any write-up of
  results obtained with it.
- **Participation-ratio loss** (`participation_ratio.py`,
  `losses/participation_ratio.py`) -- `PR_j = 1 / sum_g m_gj^2`, the effective
  number of regions feature `j` spreads its activation mass over: 1 for a single
  region, 2 for two equal ones, and so on. Minimising it concentrates each
  feature spatially. Unlike `exclusivity`, PR is homogeneous of degree 0
  (`L(alpha*z) = L(z)`), so by Euler `<grad L, z> = 0`: it has no radial gradient
  component and cannot systematically shrink activations, which is why it needs
  no straight-through binarisation. The normalisation is written without an
  epsilon (inactive features are masked out instead), because an epsilon would
  break that invariance. Both properties are pinned in
  `tests/test_participation_ratio.py`.

```bash
# Group-TopK on a fixed 2x2 tile partition (needs no cache):
uv run scripts/train.py loss.group_topk.enabled=true loss.group_topk.k_group=48

# PR loss on attention clusters:
uv run scripts/extract_attention_groups.py --method attention
uv run scripts/train.py loss.participation_ratio.weight=0.1 \
    loss.participation_ratio.grouping=attention
```

The four-arm ablation (baseline / +PR / +Group-TopK / both) lives in
`scripts/run_structure_ablation.sh` and in `scripts/sbatch/cub_struct_*.sbatch`.
Each arm reports `val/fvu`, `val/l0`, `val/monosemanticity_score`,
`val/tversky_ms`, `val/region_pr`, `val/region_dominant_share` and
`val/activation_frequency_*`. `train.eval.region_grouping` (default `"auto"`)
measures region consistency on whatever patch clusters the run's own
structural loss actually trains with (`grouping.resolve_region_grouping`),
falling back to a fixed 2x2 tile partition only when no structural loss uses
clustering at all (e.g. the plain baseline arm) -- a fixed tile would
otherwise measure something the loss was never asked to optimize (see
`participation_ratio.py`'s and `group_topk.py`'s docstrings for why a
feature-similarity cluster need not be spatially contiguous, so shrinking it
doesn't shrink an unrelated tile's footprint). Pass `region_grouping=tile` /
`attention` / `feature` explicitly to force one fixed partition across every
arm instead, at the cost of measuring some arms against a partition their own
loss never optimized. The hypothesis under test is not that the structural
terms improve FVU, but that at controlled reconstruction quality they improve
spatial specialisation and monosemanticity.

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
