"""Image generation helpers for text-to-image and image-to-image pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from src.config import AVAILABLE_MODELS


def make_pipe(repo_id: str, kind: str, device: str, dtype: Optional[torch.dtype] = None):
    """Create and configure a Diffusers pipeline for the requested model."""
    if kind == "DiffusionPipeline":
        pipe = DiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionImg2ImgPipeline":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
    else:
        raise ValueError(f"Unknown pipeline kind: {kind}")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if device == "cuda":
        if dtype is None and hasattr(torch, "float16"):
            dtype = torch.float16
        try:
            pipe = pipe.to(device, torch_dtype=dtype)
        except Exception:
            pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass
    return pipe


@torch.inference_mode()
def generate_images(
    model_key: str,
    prompts: list[str],
    out_dir: Path,
    device: str,
    max_count: Optional[int] = None,
    seed: int = 1024,
    steps: int = 30,
    guidance: float = 7.5,
) -> None:
    """Generate images for prompts with a configured text-to-image model."""
    repo_id, kind = AVAILABLE_MODELS[model_key]
    if kind == "StableDiffusionImg2ImgPipeline":
        kind = "StableDiffusionPipeline"
    pipe = make_pipe(repo_id, kind, device)
    generator = torch.Generator(device=device).manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(prompts) if max_count is None else min(max_count, len(prompts))
    for i in tqdm(range(count), desc=f"Generating with {model_key}"):
        image = pipe(
            prompts[i],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        image.save(out_dir / f"{i}.png")


@torch.inference_mode()
def img2img_from_folder(
    base_model_key: str,
    src_dir: Path,
    prompts: list[str],
    out_dir: Path,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    seed: int = 1024,
    device: str = "cuda",
    max_count: Optional[int] = None,
) -> None:
    """Generate modified images by applying img2img to a folder of source PNGs."""
    repo_id, _ = AVAILABLE_MODELS[base_model_key]
    pipe = make_pipe(repo_id, "StableDiffusionImg2ImgPipeline", device)
    generator = torch.Generator(device=device).manual_seed(seed)

    paths = sorted(src_dir.glob("*.png"), key=_natural_sort_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(paths) if max_count is None else min(max_count, len(paths))
    for i in tqdm(range(count), desc=f"Img2Img {base_model_key} <= {src_dir.name}"):
        init_image = Image.open(paths[i]).convert("RGB")
        image = pipe(
            prompt=prompts[i],
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save(out_dir / f"{i}.png")


def _natural_sort_key(path: Path) -> list[Union[int, str]]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path.stem)]
