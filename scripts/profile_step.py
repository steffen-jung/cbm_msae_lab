"""Breaks one training step down into named phases (data, encoder,
SAE encode split into matmul+activation vs. BatchTopK's own sort/scatter,
decode, loss, backward, optimizer steps) so the actual bottleneck can be read
off directly instead of guessed at.

Duplicates `ConfigurableActivationSAE.encode` and
`MatryoshkaBatchTopKTrainer.update` inline (rather than calling them) purely
so a timer can sit between their sub-steps -- see model.py / dictionary_learning's
matryoshka_batch_top_k.py for the originals this mirrors.

Usage:
    uv run scripts/profile_step.py                              # live encoder
    uv run scripts/profile_step.py train.cache.mode=cache_encoder
    uv run scripts/profile_step.py --steps 50 --warmup 10
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import hydra.utils
import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.utils.data import Dataset

from cbm_msae_lab.activation_loader import build_activation_loader
from cbm_msae_lab.config_schema import register_configs
from cbm_msae_lab.encoders.base import Encoder
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.sae.trainer import ComposableLossTrainer
from cbm_msae_lab.upsampling.base import Upsampler

register_configs()

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())


def resolve_device(requested: str) -> str:
    return requested if requested == "cuda" and torch.cuda.is_available() else "cpu"


def build_pipeline_shell(cfg: DictConfig, device: str) -> tuple[Encoder, Upsampler]:
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    upsampler = hydra.utils.instantiate(cfg.upsampler).to(device)
    return encoder, upsampler


@torch.no_grad()
def infer_grid_shape(
    encoder: Encoder, upsampler: Upsampler, image_size: int, device: str
) -> tuple[int, int]:
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = encoder(dummy).features
    if upsampler.stage == "features":
        features = upsampler(dummy, features)
    return features.shape[-2], features.shape[-1]


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.dataset, split="val")
    return train_dataset, val_dataset


@contextmanager
def timed(bucket: dict[str, float], name: str, device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if device == "cuda":
        torch.cuda.synchronize()
    bucket[name] += time.perf_counter() - t0


def profile(cfg: DictConfig, num_steps: int, num_warmup: int) -> None:
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.seed)

    train_dataset, _val_dataset = build_datasets(cfg)
    encoder, upsampler = build_pipeline_shell(cfg, device)
    grid_shape = infer_grid_shape(encoder, upsampler, cfg.dataset.image_size, device)

    trainer = ComposableLossTrainer(
        grid_shape=grid_shape,
        loss_config=cfg.loss,
        activation=cfg.sae.activation,
        steps=num_steps + num_warmup,
        activation_dim=cfg.encoder.output_dim,
        dict_size=cfg.sae.dict_size,
        k=cfg.sae.k,
        layer=0,
        lm_name="vision-encoder",
        group_fractions=list(cfg.sae.group_fractions),
        lr=cfg.sae.lr,
        warmup_steps=cfg.sae.warmup_steps,
        decay_start=cfg.sae.decay_start,
        threshold_beta=cfg.sae.threshold_beta,
        threshold_start_step=cfg.sae.threshold_start_step,
        seed=cfg.sae.seed,
        device=device,
    )
    pipeline = ConceptPipeline(encoder, upsampler, trainer.ae).to(device)
    train_loader = build_activation_loader(cfg, "train", train_dataset, pipeline, device)
    ae = trainer.ae

    times: dict[str, float] = defaultdict(float)
    it = iter(train_loader)
    pipeline.train()

    for step in range(num_warmup + num_steps):
        bucket = times if step >= num_warmup else defaultdict(float)  # discard warmup timings

        with timed(bucket, "data", device):
            x_img, _labels, _idx = next(it)
        B, P, D = x_img.shape
        x = x_img.reshape(B * P, D)

        if step == 0:
            trainer.ae.b_dec.data = trainer.geometric_median(x_img)

        # -- SAE encode, split into the two things the user asked to tell apart --
        with timed(bucket, "sae_encode_matmul", device):
            centered = (x - ae.b_dec) @ ae.W_enc + ae.b_enc
            post_act = ae._act(centered)

        with timed(bucket, "sae_encode_batchtopk_sort", device):
            flattened = post_act.flatten()
            post_topk = flattened.topk(ae.k * x.size(0), sorted=False, dim=-1)
            f = (
                torch.zeros_like(flattened)
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_act.shape)
            )
            max_act_index = ae.group_indices[ae.active_groups]
            f[:, max_act_index:] = 0

        with timed(bucket, "sae_decode", device):
            x_hat = ae.decode(f)

        with timed(bucket, "loss_terms", device):
            from cbm_msae_lab.losses.base import LossContext

            ctx = LossContext(
                sae=ae,
                trainer=trainer,
                x=x,
                f=f,
                x_hat=x_hat,
                post_act=post_act,
                x_img=x_img,
                f_img=f.reshape(B, P, -1),
                x_hat_img=x_hat.reshape(B, P, D),
                grid=trainer.grid_shape,
                step=step,
                group_labels=None,
            )
            total = torch.zeros((), device=x.device, dtype=x.dtype)
            for loss_obj in trainer.losses.values():
                if loss_obj.weight == 0.0:
                    continue
                total = total + loss_obj.weight * loss_obj.compute(ctx)

        with timed(bucket, "backward", device):
            total.backward()

        with timed(bucket, "decoder_grad_projection_and_clip", device):
            ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
                ae.W_dec.T, ae.W_dec.grad.T, ae.activation_dim, ae.dict_size
            ).T
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)

        with timed(bucket, "sae_optimizer_step", device):
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.scheduler.step()

        with timed(bucket, "decoder_renorm", device):
            ae.W_dec.data = set_decoder_norm_to_unit_norm(ae.W_dec.T, ae.activation_dim, ae.dict_size).T

    total_time = sum(times.values())
    print(f"\n{'phase':38s} {'total_s':>10s} {'per_step_ms':>12s} {'share':>7s}")
    for name, secs in sorted(times.items(), key=lambda kv: -kv[1]):
        print(f"{name:38s} {secs:10.3f} {secs / num_steps * 1000:12.2f} {secs / total_time:6.1%}")
    print(f"{'TOTAL':38s} {total_time:10.3f} {total_time / num_steps * 1000:12.2f} {100.0:6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. encoder=dinov3 train.cache.mode=cache_encoder")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=args.overrides)

    profile(cfg, num_steps=args.steps, num_warmup=args.warmup)


if __name__ == "__main__":
    main()
