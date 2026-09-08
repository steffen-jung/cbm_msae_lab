"""Builds a MaskCLIP/CLIP_DINOiser module from an OmegaConf ``_target_`` config.

Ported from CFM (cfm/clip_dinoiser_backbone/builder.py). Uses Hydra's
``instantiate`` purely as a small factory (no ``@hydra.main`` involved here).
Unlike CFM's version, this does not also eagerly import
``clip_dinoiser``/``maskclip`` at module load time: `instantiate` resolves
``_target_`` dotted paths itself via its own dynamic import, so pre-importing
them here was redundant -- and, depending on which module a caller imports
first, created a circular-import failure between this file and
``clip_dinoiser.py`` (which itself imports ``build_model`` from here).
"""

from hydra.utils import instantiate


def build_model(config, class_names):
    return instantiate(config, class_names=class_names)
