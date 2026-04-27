"""Configuration constants for the fingerprinting pipeline."""

from pathlib import Path


AVAILABLE_MODELS = {
    # base
    "ldm-text2im-large-256": ("CompVis/ldm-text2im-large-256", "DiffusionPipeline"),
    "waifu-diffusion": ("hakurei/waifu-diffusion", "DiffusionPipeline"),
    "realistic-vision-v2": ("SG161222/Realistic_Vision_V2.0", "StableDiffusionPipeline"),
    "portraitplus": ("wavymulder/portraitplus", "StableDiffusionPipeline"),
    "sd-2-1-base": ("stabilityai/stable-diffusion-2-1-base", "StableDiffusionPipeline"),
    "openjourney-v2": ("prompthero/openjourney-v2", "StableDiffusionPipeline"),
    # modified examples
    "waifu-diffusers": ("Nilaier/Waifu-Diffusers", "StableDiffusionPipeline"),
}

BASE_GENERATION_SETS = [
    ("ldm-text2im-large-256", Path("outputs/queries/ldm-text2im-large-256-images")),
    ("waifu-diffusion", Path("outputs/queries/waifu-diffusion")),
    ("realistic-vision-v2", Path("outputs/queries/Realistic_Vision_V2.0")),
    ("portraitplus", Path("outputs/queries/portraitplus")),
    ("sd-2-1-base", Path("outputs/queries/stable-diffusion-2-1-base")),
    ("openjourney-v2", Path("outputs/queries/openjourney-v2")),
]

METRIC_COMPARISON_PAIRS = [
    (Path("outputs/queries/stable-diffusion-2-1-base"), Path("outputs/queries/Realistic_Vision_V2.0")),
    (Path("outputs/modifiedqueries/Waifu-Diffusers"), Path("outputs/queries/waifu-diffusion")),
    (
        Path("outputs/modifiedqueries/input_modified_base_realistic_gen_stablediff"),
        Path("outputs/queries/Realistic_Vision_V2.0"),
    ),
    (Path("outputs/modifiedqueries/modifiedportraitplus"), Path("outputs/queries/portraitplus")),
]

DEFAULT_PROMPTS_CSV = Path("data/1k.csv")
DEFAULT_MODIFIED_PROMPTS_CSV = Path("data/modified_1k.csv")
DEFAULT_METRICS_CSV = Path("outputs/metrics_results.csv")
