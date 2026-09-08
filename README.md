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

# Pre-build an activation cache once, then train many SAE variants against it
# without re-running the encoder every time:
uv run scripts/extract_features.py
uv run scripts/train.py train.cache.mode=cache
```

Every checkpoint (`outputs/checkpoints/epoch_XXXX.pt` by default) is a single
file containing the projection + SAE weights, both optimizers' states, and
the *entire* resolved Hydra config -- `checkpointing.load_for_reproduction`
rebuilds the exact pipeline from that one file.

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
