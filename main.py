# main.py
import os, re, csv, argparse, json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
from PIL import Image
from tqdm import tqdm

# ===== optional deps (있으면 사용) =====
try:
    import numpy as np
    import cv2
    from skimage.metrics import structural_similarity as ssim
    import imagehash
except Exception:
    np = None
    cv2 = None
    ssim = None
    imagehash = None

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

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

# -------------------------
# IO helpers
# -------------------------
def load_prompts(csv_path: Path) -> List[str]:
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if row:
                prompts.append(row[0].strip())
    return prompts

def make_pipe(repo_id: str, kind: str, device: str, dtype: Optional[torch.dtype] = None):
    if kind == "DiffusionPipeline":
        pipe = DiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionImg2ImgPipeline":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
    else:
        raise ValueError(f"Unknown pipeline kind: {kind}")

    # optional perf tweaks
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if device == "cuda":
        if dtype is None and hasattr(torch, "float16"):
            dtype = torch.float16
        try:
            pipe = pipe.to(device, torch_dtype=dtype)
        except Exception:
            pipe = pipe.to(device)
    # 연구 목적: safety checker 비활성 (원하면 주석 해제)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass
    return pipe

@torch.inference_mode()
def generate_images(
    model_key: str,
    prompts: List[str],
    out_dir: Path,
    device: str,
    max_count: Optional[int] = None,
    seed: int = 1024,
    steps: int = 30,
    guidance: float = 7.5,
):
    repo_id, kind = AVAILABLE_MODELS[model_key]
    # Img2Img가 아닌 일반 파이프
    if kind == "StableDiffusionImg2ImgPipeline":
        kind = "StableDiffusionPipeline"
    pipe = make_pipe(repo_id, kind, device)
    g = torch.Generator(device=device).manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(prompts) if max_count is None else min(max_count, len(prompts))
    for i in tqdm(range(n), desc=f"Generating with {model_key}"):
        img = pipe(
            prompts[i],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g
        ).images[0]
        img.save(out_dir / f"{i}.png")

@torch.inference_mode()
def img2img_from_folder(
    base_model_key: str,
    src_dir: Path,
    prompts: List[str],
    out_dir: Path,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    seed: int = 1024,
    device: str = "cuda",
    max_count: Optional[int] = None,
):
    repo_id, _ = AVAILABLE_MODELS[base_model_key]
    pipe = make_pipe(repo_id, "StableDiffusionImg2ImgPipeline", device)
    rng = torch.Generator(device=device).manual_seed(seed)

    def _ns_key(p: Path):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.stem)]

    paths = sorted(src_dir.glob("*.png"), key=_ns_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(paths) if max_count is None else min(max_count, len(paths))
    for i in tqdm(range(n), desc=f"Img2Img {base_model_key} <= {src_dir.name}"):
        init = Image.open(paths[i]).convert("RGB")
        img = pipe(
            prompt=prompts[i],
            image=init,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=rng
        ).images[0]
        img.save(out_dir / f"{i}.png")

# -------------------------
# Metrics (pairwise dir A vs B)
# -------------------------
def _ns_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.stem)]

def _pair_files(a: Path, b: Path) -> List[Tuple[Path, Path]]:
    fa = sorted(a.glob("*.png"), key=_ns_key)
    fb = sorted(b.glob("*.png"), key=_ns_key)
    n = min(len(fa), len(fb))
    return list(zip(fa[:n], fb[:n]))

def metric_phash_with_stats(a: Path, b: Path, thresh: int = 30) -> Dict[str, Any]:
    if imagehash is None:
        return {"pHash_acc(%)": float("nan"), "pHash_dist_mean": float("nan"), "pHash_dist_std": float("nan")}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"pHash_acc(%)": float("nan"), "pHash_dist_mean": float("nan"), "pHash_dist_std": float("nan")}
    dists = []
    ok = 0
    for pa, pb in pairs:
        h1 = imagehash.phash(Image.open(pa).convert("RGB"))
        h2 = imagehash.phash(Image.open(pb).convert("RGB"))
        d = (h1 - h2)
        dists.append(d)
        ok += int(d <= thresh)
    dists = np.array(dists, dtype=float)
    return {
        "pHash_acc(%)": 100.0 * ok / len(pairs),
        "pHash_dist_mean": float(np.mean(dists)),
        "pHash_dist_std": float(np.std(dists)),
        "pHash_pairs": len(pairs),
        "pHash_thresh": thresh,
    }

def metric_ssim(a: Path, b: Path) -> Dict[str, Any]:
    if (ssim is None) or (cv2 is None) or (np is None):
        return {"SSIM_mean(%)": float("nan"), "SSIM_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"SSIM_mean(%)": float("nan"), "SSIM_samples": 0}
    vals = []
    for pa, pb in pairs:
        ia = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None:
            continue
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_AREA)
        ia = ia.astype(np.float64) / 255.0
        ib = ib.astype(np.float64) / 255.0
        score = ssim(ia, ib, data_range=1.0)
        vals.append(float(score))
    return {
        "SSIM_mean(%)": float(np.mean(vals) * 100.0) if vals else float("nan"),
        "SSIM_samples": len(vals),
    }

def _hsv_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [50,60,60], [0,180, 0,256, 0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def metric_hsv_hist_corr(a: Path, b: Path) -> Dict[str, Any]:
    if (cv2 is None) or (np is None):
        return {"HSV_hist_corr_mean(%)": float("nan"), "HSV_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"HSV_hist_corr_mean(%)": float("nan"), "HSV_samples": 0}
    scores = []
    for pa, pb in pairs:
        ia = cv2.imread(str(pa))
        ib = cv2.imread(str(pb))
        if ia is None or ib is None:
            continue
        # 해상도 통일 후 히스토그램
        if ia.shape[:2] != ib.shape[:2]:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_AREA)
        ha = _hsv_hist(ia); hb = _hsv_hist(ib)
        corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)  # [-1,1]
        scores.append(float(corr))
    return {
        "HSV_hist_corr_mean(%)": float(np.mean(scores) * 100.0) if scores else float("nan"),
        "HSV_samples": len(scores),
    }

def metric_template_match(a: Path, b: Path, thresh: float = 0.80) -> Dict[str, Any]:
    """TM_CCOEFF_NORMED에서 max값이 임계치 이상인 비율(%)"""
    if (cv2 is None) or (np is None):
        return {"Template_match_hit(%)": float("nan"), "Template_thresh": thresh, "Template_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"Template_match_hit(%)": float("nan"), "Template_thresh": thresh, "Template_samples": 0}
    hits = 0
    used = 0
    for pa, pb in pairs:
        ta = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        tb = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ta is None or tb is None:
            continue
        # 템플릿은 더 작은 쪽으로
        if ta.shape[0]*ta.shape[1] <= tb.shape[0]*tb.shape[1]:
            templ, target = ta, tb
        else:
            templ, target = tb, ta
        res = cv2.matchTemplate(target, templ, cv2.TM_CCOEFF_NORMED)
        m = float(res.max()) if res.size else 0.0
        hits += int(m >= thresh)
        used += 1
    return {
        "Template_match_hit(%)": 100.0 * hits / used if used else float("nan"),
        "Template_thresh": thresh,
        "Template_samples": used,
    }

def metric_orb_inlier_ratio(
    a: Path, b: Path,
    nfeatures: int = 1500,
    ratio_thresh: float = 0.7,
    ransac_reproj: float = 3.0
) -> Dict[str, Any]:
    """
    ORB 키포인트 매칭 + RANSAC 호모그래피 인라이어 비율(%)
    - 권장: 인라이어 / good_matches 로 보정
    """
    if (cv2 is None) or (np is None):
        return {"ORB_inlier_ratio_mean(%)": float("nan"), "ORB_samples": 0}
    pairs = _pair_files(a, b)
    if not pairs:
        return {"ORB_inlier_ratio_mean(%)": float("nan"), "ORB_samples": 0}

    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ratios = []
    used = 0
    for pa, pb in pairs:
        ia = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None:
            continue
        # 크기 차이 보정
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_AREA)

        ka, da = orb.detectAndCompute(ia, None)
        kb, db = orb.detectAndCompute(ib, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            continue

        matches = bf.knnMatch(da, db, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

        if len(good) < 8:
            continue

        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj)
        if mask is None:
            continue
        inliers = int(mask.sum())
        ratio = 100.0 * inliers / max(1, len(good))  # 분모: good matches
        ratios.append(ratio)
        used += 1

    return {
        "ORB_inlier_ratio_mean(%)": float(np.mean(ratios)) if ratios else float("nan"),
        "ORB_samples": used,
        "ORB_nfeatures": nfeatures,
        "ORB_ratio_thresh": ratio_thresh,
        "ORB_ransac_reproj": ransac_reproj,
    }

def combined_accuracy_weighted(res: dict) -> float:
    # 템플릿 매칭은 종합점수에서 제외(노이즈 경향)
    weights = {
        "pHash_acc(%)": 0.40,
        "SSIM_mean(%)": 0.35,
        "HSV_hist_corr_mean(%)": 0.15,
        "ORB_inlier_ratio_mean(%)": 0.10,
    }
    score, total_w = 0.0, 0.0
    for k, w in weights.items():
        v = res.get(k)
        if v is not None and v == v:  # not NaN
            score += float(v) * w
            total_w += w
    return score / total_w if total_w > 0 else float("nan")


def run_all_metrics(
    pairs: List[Tuple[Path, Path]],
    phash_thresh: int = 30,
    orb_nfeatures: int = 1500,
    orb_ratio_thresh: float = 0.7,
    orb_ransac_reproj: float = 3.0,
    template_thresh: float = 0.80,
) -> List[Dict[str, Any]]:
    results = []
    for A, B in pairs:
        if not A.exists() or not B.exists():
            continue
        row: Dict[str, Any] = {
            "A": str(A),
            "B": str(B),
        }
        # pHash
        ph = metric_phash_with_stats(A, B, phash_thresh)
        row.update(ph)
        # SSIM
        ss = metric_ssim(A, B)
        row.update(ss)
        # HSV
        hv = metric_hsv_hist_corr(A, B)
        row.update(hv)
        # Template (참고용)
        tm = metric_template_match(A, B, template_thresh)
        row.update(tm)
        # ORB
        ob = metric_orb_inlier_ratio(A, B, orb_nfeatures, orb_ratio_thresh, orb_ransac_reproj)
        row.update(ob)

        # combined
        row["Combined_acc(%)"] = combined_accuracy_weighted({
            "pHash_acc(%)": row.get("pHash_acc(%)"),
            "SSIM_mean(%)": row.get("SSIM_mean(%)"),
            "HSV_hist_corr_mean(%)": row.get("HSV_hist_corr_mean(%)"),
            "ORB_inlier_ratio_mean(%)": row.get("ORB_inlier_ratio_mean(%)"),
        })
        results.append(row)
    return results


def save_results_csv(results: List[Dict[str, Any]], out_path="outputs/metrics_results.csv"):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        print("⚠️ No results to save.")
        return

    # 고정 컬럼 순서 (CSV 해석 용이)
    fieldnames = [
        "A", "B",
        "pHash_acc(%)", "pHash_dist_mean", "pHash_dist_std", "pHash_pairs", "pHash_thresh",
        "SSIM_mean(%)", "SSIM_samples",
        "HSV_hist_corr_mean(%)", "HSV_samples",
        "Template_match_hit(%)", "Template_thresh", "Template_samples",
        "ORB_inlier_ratio_mean(%)", "ORB_samples", "ORB_nfeatures", "ORB_ratio_thresh", "ORB_ransac_reproj",
        "Combined_acc(%)",
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✅ Metrics saved to {out_file}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="T2I Fingerprinting (local GPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompts_csv", default="data/1k.csv")
    ap.add_argument("--modified_prompts_csv", default="data/modified_1k.csv")
    ap.add_argument("--max_count", type=int, default=None)

    # generation params
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)

    # metrics params
    ap.add_argument("--phash_thresh", type=int, default=30)
    ap.add_argument("--template_thresh", type=float, default=0.80)
    ap.add_argument("--orb_nfeatures", type=int, default=1500)
    ap.add_argument("--orb_ratio_thresh", type=float, default=0.7)
    ap.add_argument("--orb_ransac_reproj", type=float, default=3.0)

    ap.add_argument("--do_base", action="store_true")
    ap.add_argument("--do_modified", action="store_true")
    ap.add_argument("--do_metrics", action="store_true", help="모든 지표 계산")
    ap.add_argument("--csv_out", default="outputs/metrics_results.csv",
                    help="CSV output file path")
    args = ap.parse_args()

    device = args.device
    prompts = load_prompts(Path(args.prompts_csv))
    mod_prompts = load_prompts(Path(args.modified_prompts_csv)) if Path(args.modified_prompts_csv).exists() else prompts

    # 1) 기본 생성
    if args.do_base:
        base_sets = [
            ("ldm-text2im-large-256", Path("outputs/queries/ldm-text2im-large-256-images")),
            ("waifu-diffusion",        Path("outputs/queries/waifu-diffusion")),
            ("realistic-vision-v2",    Path("outputs/queries/Realistic_Vision_V2.0")),
            ("portraitplus",           Path("outputs/queries/portraitplus")),
            ("sd-2-1-base",            Path("outputs/queries/stable-diffusion-2-1-base")),
            ("openjourney-v2",         Path("outputs/queries/openjourney-v2")),
        ]
        for key, outdir in base_sets:
            try:
                generate_images(
                    key, prompts, outdir, device, args.max_count,
                    seed=args.seed, steps=args.steps, guidance=args.guidance
                )
            except Exception as e:
                print(f"[WARN] {key}: {e}")

    # 2) 변형 생성
    if args.do_modified:
        # fine-tuned 예시
        try:
            generate_images(
                "waifu-diffusers", prompts, Path("outputs/modifiedqueries/Waifu-Diffusers"),
                device, args.max_count, seed=args.seed, steps=args.steps, guidance=args.guidance
            )
        except Exception as e:
            print(f"[WARN] waifu-diffusers: {e}")

        # img2img: realistic-vision 결과를 sd-2-1-base 가이드로 변환
        src = Path("outputs/queries/Realistic_Vision_V2.0")
        if src.exists() and any(src.glob("*.png")):
            try:
                img2img_from_folder(
                    "sd-2-1-base", src, prompts,
                    Path("outputs/modifiedqueries/input_modified_base_realistic_gen_stablediff"),
                    strength=0.75, guidance_scale=args.guidance, seed=args.seed,
                    device=device, max_count=args.max_count
                )
            except Exception as e:
                print(f"[WARN] img2img: {e}")

        # modified prompts + portraitplus
        try:
            generate_images(
                "portraitplus", mod_prompts, Path("outputs/modifiedqueries/modifiedportraitplus"),
                device, args.max_count, seed=args.seed, steps=args.steps, guidance=args.guidance
            )
        except Exception as e:
            print(f"[WARN] modifiedportraitplus: {e}")

    # 3) 전 지표 계산
    if args.do_metrics:
        pairs = [
            # 예시 비교쌍
            (Path("outputs/queries/stable-diffusion-2-1-base"), Path("outputs/queries/Realistic_Vision_V2.0")),
            (Path("outputs/modifiedqueries/Waifu-Diffusers"), Path("outputs/queries/waifu-diffusion")),
            (Path("outputs/modifiedqueries/input_modified_base_realistic_gen_stablediff"), Path("outputs/queries/Realistic_Vision_V2.0")),
            (Path("outputs/modifiedqueries/modifiedportraitplus"), Path("outputs/queries/portraitplus")),
        ]
        results = run_all_metrics(
            pairs,
            phash_thresh=args.phash_thresh,
            orb_nfeatures=args.orb_nfeatures,
            orb_ratio_thresh=args.orb_ratio_thresh,
            orb_ransac_reproj=args.orb_ransac_reproj,
            template_thresh=args.template_thresh,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        save_results_csv(results, args.csv_out)

if __name__ == "__main__":
    main()
