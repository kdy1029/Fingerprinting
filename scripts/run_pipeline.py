"""Command-line entrypoint for the fingerprinting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BASE_GENERATION_SETS,
    DEFAULT_METRICS_CSV,
    DEFAULT_MODIFIED_PROMPTS_CSV,
    DEFAULT_PROMPTS_CSV,
    METRIC_COMPARISON_PAIRS,
)
from src.io_utils import load_prompts, save_results_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser while preserving the original arguments."""
    parser = argparse.ArgumentParser(description="T2I Fingerprinting (local GPU)")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--prompts_csv", default=str(DEFAULT_PROMPTS_CSV))
    parser.add_argument("--modified_prompts_csv", default=str(DEFAULT_MODIFIED_PROMPTS_CSV))
    parser.add_argument("--max_count", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)

    parser.add_argument("--phash_thresh", type=int, default=30)
    parser.add_argument("--template_thresh", type=float, default=0.80)
    parser.add_argument("--orb_nfeatures", type=int, default=1500)
    parser.add_argument("--orb_ratio_thresh", type=float, default=0.7)
    parser.add_argument("--orb_ransac_reproj", type=float, default=3.0)

    parser.add_argument("--do_base", action="store_true")
    parser.add_argument("--do_modified", action="store_true")
    parser.add_argument("--do_metrics", action="store_true", help="Calculate all metrics")
    parser.add_argument("--csv_out", default=str(DEFAULT_METRICS_CSV), help="CSV output file path")
    return parser


def default_device() -> str:
    """Return the original CUDA-first default when PyTorch is available."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def main() -> None:
    """Run the selected generation and evaluation stages."""
    args = build_parser().parse_args()

    device = args.device
    prompts = load_prompts(Path(args.prompts_csv))
    modified_prompts_csv = Path(args.modified_prompts_csv)
    modified_prompts = load_prompts(modified_prompts_csv) if modified_prompts_csv.exists() else prompts

    if args.do_base:
        run_base_generation(args, prompts, device)

    if args.do_modified:
        run_modified_generation(args, prompts, modified_prompts, device)

    if args.do_metrics:
        from src.metrics import run_all_metrics

        results = run_all_metrics(
            METRIC_COMPARISON_PAIRS,
            phash_thresh=args.phash_thresh,
            orb_nfeatures=args.orb_nfeatures,
            orb_ratio_thresh=args.orb_ratio_thresh,
            orb_ransac_reproj=args.orb_ransac_reproj,
            template_thresh=args.template_thresh,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        save_results_csv(results, args.csv_out)


def run_base_generation(args: argparse.Namespace, prompts: list[str], device: str) -> None:
    """Generate the configured base image sets."""
    from src.generation import generate_images

    for model_key, out_dir in BASE_GENERATION_SETS:
        try:
            generate_images(
                model_key,
                prompts,
                out_dir,
                device,
                args.max_count,
                seed=args.seed,
                steps=args.steps,
                guidance=args.guidance,
            )
        except Exception as exc:
            print(f"[WARN] {model_key}: {exc}")


def run_modified_generation(
    args: argparse.Namespace,
    prompts: list[str],
    modified_prompts: list[str],
    device: str,
) -> None:
    """Generate the configured modified image sets."""
    from src.generation import generate_images, img2img_from_folder

    try:
        generate_images(
            "waifu-diffusers",
            prompts,
            Path("outputs/modifiedqueries/Waifu-Diffusers"),
            device,
            args.max_count,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
        )
    except Exception as exc:
        print(f"[WARN] waifu-diffusers: {exc}")

    source_dir = Path("outputs/queries/Realistic_Vision_V2.0")
    if source_dir.exists() and any(source_dir.glob("*.png")):
        try:
            img2img_from_folder(
                "sd-2-1-base",
                source_dir,
                prompts,
                Path("outputs/modifiedqueries/input_modified_base_realistic_gen_stablediff"),
                strength=0.75,
                guidance_scale=args.guidance,
                seed=args.seed,
                device=device,
                max_count=args.max_count,
            )
        except Exception as exc:
            print(f"[WARN] img2img: {exc}")

    try:
        generate_images(
            "portraitplus",
            modified_prompts,
            Path("outputs/modifiedqueries/modifiedportraitplus"),
            device,
            args.max_count,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
        )
    except Exception as exc:
        print(f"[WARN] modifiedportraitplus: {exc}")


if __name__ == "__main__":
    main()
