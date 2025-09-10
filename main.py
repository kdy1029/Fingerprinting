# main.py
import os, csv, argparse, json
from pathlib import Path
from typing import List, Optional, Tuple
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

def make_pipe(repo_id: str, kind: str, device: str):
    if kind == "DiffusionPipeline":
        pipe = DiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(repo_id)
    elif kind == "StableDiffusionImg2ImgPipeline":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
    else:
        raise ValueError(f"Unknown pipeline kind: {kind}")
    if device == "cuda":
        pipe = pipe.to("cuda")
    return pipe

@torch.inference_mode()
def generate_images(model_key: str, prompts: List[str], out_dir: Path, device: str, max_count: Optional[int] = None):
    repo_id, kind = AVAILABLE_MODELS[model_key]
    # Img2Img가 아닌 일반 파이프
    if kind == "StableDiffusionImg2ImgPipeline":
        kind = "StableDiffusionPipeline"
    pipe = make_pipe(repo_id, kind, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(prompts) if max_count is None else min(max_count, len(prompts))
    for i in tqdm(range(n), desc=f"Generating with {model_key}"):
        img = pipe(prompts[i]).images[0]
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
    paths = sorted(src_dir.glob("*.png"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(paths) if max_count is None else min(max_count, len(paths))
    for i in tqdm(range(n), desc=f"Img2Img {base_model_key} <= {src_dir.name}"):
        init = Image.open(paths[i]).convert("RGB")
        img = pipe(prompt=prompts[i], image=init, strength=strength, guidance_scale=guidance_scale, generator=rng).images[0]
        img.save(out_dir / f"{i}.png")

# -------------------------
# Metrics (pairwise dir A vs B)
# -------------------------
def _pair_files(a: Path, b: Path) -> List[Tuple[Path, Path]]:
    fa = sorted(a.glob("*.png"))
    fb = sorted(b.glob("*.png"))
    n = min(len(fa), len(fb))
    return list(zip(fa[:n], fb[:n]))

def metric_phash(a: Path, b: Path, thresh: int = 30) -> float:
    if imagehash is None:
        return float("nan")
    pairs = _pair_files(a, b)
    if not pairs: return float("nan")
    ok = 0
    for pa, pb in pairs:
        h1 = imagehash.phash(Image.open(pa).convert("RGB"))
        h2 = imagehash.phash(Image.open(pb).convert("RGB"))
        ok += int((h1 - h2) <= thresh)
    return 100.0 * ok / len(pairs)  # % of pairs within distance threshold

def metric_ssim(a: Path, b: Path) -> float:
    if (ssim is None) or (cv2 is None) or (np is None):
        return float("nan")
    pairs = _pair_files(a, b)
    if not pairs: return float("nan")
    vals = []
    for pa, pb in pairs:
        ia = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None: continue
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
        score, _ = ssim(ia, ib, full=True)
        vals.append(float(score))
    return float(np.mean(vals))*100.0 if vals else float("nan")

def _hsv_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[50,60,60],[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def metric_hsv_hist_corr(a: Path, b: Path) -> float:
    if (cv2 is None) or (np is None):
        return float("nan")
    pairs = _pair_files(a, b)
    if not pairs: return float("nan")
    scores = []
    for pa, pb in pairs:
        ia = cv2.imread(str(pa))
        ib = cv2.imread(str(pb))
        if ia is None or ib is None: continue
        ha = _hsv_hist(ia); hb = _hsv_hist(ib)
        corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)  # [-1,1]
        scores.append(float(corr))
    return float(np.mean(scores))*100.0 if scores else float("nan")

def metric_template_match(a: Path, b: Path, thresh: float = 0.80) -> float:
    """
    TM_CCOEFF_NORMED에서 max값이 임계치 이상인 비율(%)의 평균.
    구조가 달라도 '패치 유사'가 강하면 잡힘. 값이 클수록 유사.
    """
    if (cv2 is None) or (np is None):
        return float("nan")
    pairs = _pair_files(a, b)
    if not pairs: return float("nan")
    hits = 0
    for pa, pb in pairs:
        ta = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        tb = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ta is None or tb is None: continue
        # 템플릿은 더 작은 쪽으로
        if ta.shape[0]*ta.shape[1] <= tb.shape[0]*tb.shape[1]:
            templ, target = ta, tb
        else:
            templ, target = tb, ta
        res = cv2.matchTemplate(target, templ, cv2.TM_CCOEFF_NORMED)
        m = float(res.max()) if res.size else 0.0
        hits += int(m >= thresh)
    return 100.0 * hits / len(pairs)

def metric_orb_inlier_ratio(a: Path, b: Path) -> float:
    """
    ORB 키포인트 매칭 + RANSAC 호모그래피 인라이어 비율(%)
    - 회전/스케일 변화에 비교적 강함
    """
    if (cv2 is None) or (np is None):
        return float("nan")
    pairs = _pair_files(a, b)
    if not pairs: return float("nan")

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ratios = []
    for pa, pb in pairs:
        ia = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None: continue
        # 크기 차이 보정(선택)
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))

        ka, da = orb.detectAndCompute(ia, None)
        kb, db = orb.detectAndCompute(ib, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            continue

        # knn 매칭 + ratio test
        matches = bf.knnMatch(da, db, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        if len(good) < 8:
            continue

        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            continue
        inliers = int(mask.sum())
        ratio = 100.0 * inliers / max(1, len(ka))  # 논문식 S = |inliers| / |K1|
        ratios.append(ratio)

    return float(np.mean(ratios)) if ratios else float("nan")

def combined_accuracy_weighted(res: dict) -> float:
    weights = {
        "pHash_acc(%)": 0.40,
        "SSIM_mean(%)": 0.35,
        "HSV_hist_corr_mean(%)": 0.15,
        "ORB_inlier_ratio_mean(%)": 0.10,
        # "Template_match_hit(%)": 0.00,  # 제외
    }
    score, total_w = 0.0, 0.0
    for k, w in weights.items():
        v = res.get(k)
        if v is not None and v == v:  # not NaN
            score += v * w
            total_w += w
    return score / total_w if total_w > 0 else float("nan")


def run_all_metrics(pairs):
    results = []
    for A, B in pairs:
        if not A.exists() or not B.exists():
            continue
        res = {
            "A": str(A),
            "B": str(B),
            "pHash_acc(%)": metric_phash(A, B),
            "SSIM_mean(%)": metric_ssim(A, B),
            "HSV_hist_corr_mean(%)": metric_hsv_hist_corr(A, B),
            "Template_match_hit(%)": metric_template_match(A, B),
            "ORB_inlier_ratio_mean(%)": metric_orb_inlier_ratio(A, B),
        }
        results.append(res)
    return results



def save_results_csv(results, out_path="outputs/metrics_results.csv"):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        print("⚠️ No results to save.")
        return
    keys = results[0].keys()
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
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
                generate_images(key, prompts, outdir, device, args.max_count)
            except Exception as e:
                print(f"[WARN] {key}: {e}")

    # 2) 변형 생성
    if args.do_modified:
        # fine-tuned 예시
        try:
            generate_images("waifu-diffusers", prompts, Path("outputs/modifiedqueries/Waifu-Diffusers"), device, args.max_count)
        except Exception as e:
            print(f"[WARN] waifu-diffusers: {e}")

        # img2img: realistic-vision 결과를 sd-2-1-base 가이드로 변환
        src = Path("outputs/queries/Realistic_Vision_V2.0")
        if src.exists() and any(src.glob("*.png")):
            try:
                img2img_from_folder(
                    "sd-2-1-base", src, prompts,
                    Path("outputs/modifiedqueries/input_modified_base_realistic_gen_stablediff"),
                    strength=0.75, guidance_scale=7.5, seed=1024, device=device, max_count=args.max_count
                )
            except Exception as e:
                print(f"[WARN] img2img: {e}")

        # modified prompts + portraitplus
        try:
            generate_images("portraitplus", mod_prompts, Path("outputs/modifiedqueries/modifiedportraitplus"), device, args.max_count)
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
        results = run_all_metrics(pairs)
        for r in results:
            r["Combined_acc(%)"] = combined_accuracy_weighted(r)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        save_results_csv(results, args.csv_out)

if __name__ == "__main__":
    main()
