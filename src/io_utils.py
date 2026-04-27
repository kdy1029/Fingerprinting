"""Input and output helpers for prompt and metrics files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Union


def load_prompts(csv_path: Path) -> list[str]:
    """Load prompt text from the first column of a CSV file."""
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                prompts.append(row[0].strip())
    return prompts


def save_results_csv(results: list[dict[str, Any]], out_path: Union[str, Path] = "outputs/metrics_results.csv") -> None:
    """Save metric rows to a CSV file using a stable column order."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        print("[WARN] No results to save.")
        return

    fieldnames = [
        "A",
        "B",
        "pHash_acc(%)",
        "pHash_dist_mean",
        "pHash_dist_std",
        "pHash_pairs",
        "pHash_thresh",
        "SSIM_mean(%)",
        "SSIM_samples",
        "HSV_hist_corr_mean(%)",
        "HSV_samples",
        "Template_match_hit(%)",
        "Template_thresh",
        "Template_samples",
        "ORB_inlier_ratio_mean(%)",
        "ORB_samples",
        "ORB_nfeatures",
        "ORB_ratio_thresh",
        "ORB_ransac_reproj",
        "Combined_acc(%)",
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Metrics saved to {out_file}")
