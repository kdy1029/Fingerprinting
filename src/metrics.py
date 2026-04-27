"""Similarity metrics for comparing generated image directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
    import imagehash
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
except Exception:
    cv2 = None
    imagehash = None
    np = None
    ssim = None


def metric_phash_with_stats(a: Path, b: Path, thresh: int = 30) -> dict[str, Any]:
    """Compute pHash threshold accuracy plus distance summary statistics."""
    if imagehash is None or np is None or Image is None:
        return {"pHash_acc(%)": float("nan"), "pHash_dist_mean": float("nan"), "pHash_dist_std": float("nan")}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"pHash_acc(%)": float("nan"), "pHash_dist_mean": float("nan"), "pHash_dist_std": float("nan")}
    distances = []
    hits = 0
    for path_a, path_b in pairs:
        hash_a = imagehash.phash(Image.open(path_a).convert("RGB"))
        hash_b = imagehash.phash(Image.open(path_b).convert("RGB"))
        distance = hash_a - hash_b
        distances.append(distance)
        hits += int(distance <= thresh)
    distances = np.array(distances, dtype=float)
    return {
        "pHash_acc(%)": 100.0 * hits / len(pairs),
        "pHash_dist_mean": float(np.mean(distances)),
        "pHash_dist_std": float(np.std(distances)),
        "pHash_pairs": len(pairs),
        "pHash_thresh": thresh,
    }


def metric_ssim(a: Path, b: Path) -> dict[str, Any]:
    """Compute mean structural similarity between paired grayscale images."""
    if ssim is None or cv2 is None or np is None:
        return {"SSIM_mean(%)": float("nan"), "SSIM_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"SSIM_mean(%)": float("nan"), "SSIM_samples": 0}
    values = []
    for path_a, path_b in pairs:
        image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            continue
        if image_a.shape != image_b.shape:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_AREA)
        image_a = image_a.astype(np.float64) / 255.0
        image_b = image_b.astype(np.float64) / 255.0
        values.append(float(ssim(image_a, image_b, data_range=1.0)))
    return {
        "SSIM_mean(%)": float(np.mean(values) * 100.0) if values else float("nan"),
        "SSIM_samples": len(values),
    }


def metric_hsv_hist_corr(a: Path, b: Path) -> dict[str, Any]:
    """Compute mean HSV histogram correlation for paired images."""
    if cv2 is None or np is None:
        return {"HSV_hist_corr_mean(%)": float("nan"), "HSV_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"HSV_hist_corr_mean(%)": float("nan"), "HSV_samples": 0}
    scores = []
    for path_a, path_b in pairs:
        image_a = cv2.imread(str(path_a))
        image_b = cv2.imread(str(path_b))
        if image_a is None or image_b is None:
            continue
        if image_a.shape[:2] != image_b.shape[:2]:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_AREA)
        hist_a = _hsv_hist(image_a)
        hist_b = _hsv_hist(image_b)
        scores.append(float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))
    return {
        "HSV_hist_corr_mean(%)": float(np.mean(scores) * 100.0) if scores else float("nan"),
        "HSV_samples": len(scores),
    }


def metric_template_match(a: Path, b: Path, thresh: float = 0.80) -> dict[str, Any]:
    """Compute the percent of image pairs whose template match exceeds a threshold."""
    if cv2 is None or np is None:
        return {"Template_match_hit(%)": float("nan"), "Template_thresh": thresh, "Template_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"Template_match_hit(%)": float("nan"), "Template_thresh": thresh, "Template_samples": 0}
    hits = 0
    used = 0
    for path_a, path_b in pairs:
        template_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        template_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if template_a is None or template_b is None:
            continue
        if template_a.shape[0] * template_a.shape[1] <= template_b.shape[0] * template_b.shape[1]:
            template, target = template_a, template_b
        else:
            template, target = template_b, template_a
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        max_value = float(result.max()) if result.size else 0.0
        hits += int(max_value >= thresh)
        used += 1
    return {
        "Template_match_hit(%)": 100.0 * hits / used if used else float("nan"),
        "Template_thresh": thresh,
        "Template_samples": used,
    }


def metric_orb_inlier_ratio(
    a: Path,
    b: Path,
    nfeatures: int = 1500,
    ratio_thresh: float = 0.7,
    ransac_reproj: float = 3.0,
) -> dict[str, Any]:
    """Compute ORB keypoint matching plus RANSAC homography inlier ratio."""
    if cv2 is None or np is None:
        return {"ORB_inlier_ratio_mean(%)": float("nan"), "ORB_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"ORB_inlier_ratio_mean(%)": float("nan"), "ORB_samples": 0}

    orb = cv2.ORB_create(nfeatures=nfeatures)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ratios = []
    used = 0
    for path_a, path_b in pairs:
        image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            continue
        if image_a.shape != image_b.shape:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_AREA)

        keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
            continue

        matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        if len(good_matches) < 8:
            continue

        src = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj)
        if mask is None:
            continue
        inliers = int(mask.sum())
        ratios.append(100.0 * inliers / max(1, len(good_matches)))
        used += 1

    return {
        "ORB_inlier_ratio_mean(%)": float(np.mean(ratios)) if ratios else float("nan"),
        "ORB_samples": used,
        "ORB_nfeatures": nfeatures,
        "ORB_ratio_thresh": ratio_thresh,
        "ORB_ransac_reproj": ransac_reproj,
    }


def combined_accuracy_weighted(result: dict[str, Any]) -> float:
    """Combine selected metrics into the existing weighted accuracy score."""
    weights = {
        "pHash_acc(%)": 0.40,
        "SSIM_mean(%)": 0.35,
        "HSV_hist_corr_mean(%)": 0.15,
        "ORB_inlier_ratio_mean(%)": 0.10,
    }
    score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        value = result.get(key)
        if value is not None and value == value:
            score += float(value) * weight
            total_weight += weight
    return score / total_weight if total_weight > 0 else float("nan")


def run_all_metrics(
    pairs: list[tuple[Path, Path]],
    phash_thresh: int = 30,
    orb_nfeatures: int = 1500,
    orb_ratio_thresh: float = 0.7,
    orb_ransac_reproj: float = 3.0,
    template_thresh: float = 0.80,
) -> list[dict[str, Any]]:
    """Run all configured metrics for each pair of image directories."""
    results = []
    for dir_a, dir_b in pairs:
        if not dir_a.exists() or not dir_b.exists():
            continue
        row: dict[str, Any] = {"A": str(dir_a), "B": str(dir_b)}
        row.update(metric_phash_with_stats(dir_a, dir_b, phash_thresh))
        row.update(metric_ssim(dir_a, dir_b))
        row.update(metric_hsv_hist_corr(dir_a, dir_b))
        row.update(metric_template_match(dir_a, dir_b, template_thresh))
        row.update(metric_orb_inlier_ratio(dir_a, dir_b, orb_nfeatures, orb_ratio_thresh, orb_ransac_reproj))
        row["Combined_acc(%)"] = combined_accuracy_weighted(
            {
                "pHash_acc(%)": row.get("pHash_acc(%)"),
                "SSIM_mean(%)": row.get("SSIM_mean(%)"),
                "HSV_hist_corr_mean(%)": row.get("HSV_hist_corr_mean(%)"),
                "ORB_inlier_ratio_mean(%)": row.get("ORB_inlier_ratio_mean(%)"),
            }
        )
        results.append(row)
    return results


def _natural_sort_key(path: Path) -> list[Union[int, str]]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path.stem)]


def _pair_files(a: Path, b: Path) -> list[tuple[Path, Path]]:
    files_a = sorted(a.glob("*.png"), key=_natural_sort_key)
    files_b = sorted(b.glob("*.png"), key=_natural_sort_key)
    count = min(len(files_a), len(files_b))
    return list(zip(files_a[:count], files_b[:count]))


def _hsv_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()
