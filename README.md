# Query-Based Fingerprinting of Text-to-Image Models

This repository implements a simple pipeline for query-based fingerprinting of text-to-image (T2I) models.

In this project, **fingerprinting** means generating comparable image sets from known prompts and measuring how similar those sets are. The resulting scores act like a behavioral signature for a model or model variant. The pipeline is useful for comparing base models, fine-tuned models, image-to-image modifications, and prompt-modified outputs.

## Pipeline

The workflow has three stages:

1. **Generate** base images from a prompt CSV using several Diffusers models.
2. **Modify** the query outputs with fine-tuned models, image-to-image generation, or modified prompts.
3. **Evaluate** paired image folders with similarity metrics:
   - Perceptual hash accuracy and distance statistics
   - Structural Similarity Index Measure (SSIM)
   - HSV histogram correlation
   - ORB feature matching inlier ratio
   - Template matching

The combined score preserves the original weighting used by the project: pHash, SSIM, HSV histogram correlation, and ORB inlier ratio. Template matching is still reported separately.

## Project Structure

```text
.
├── data/                   # Prompt CSV files
├── scripts/
│   └── run_pipeline.py     # Main CLI entrypoint
├── src/
│   ├── config.py           # Model IDs, output folders, metric pairs, defaults
│   ├── generation.py       # Text-to-image and image-to-image generation
│   ├── io_utils.py         # Prompt loading and CSV writing
│   └── metrics.py          # pHash, SSIM, HSV, ORB, template matching
├── main.py                 # Backward-compatible wrapper
└── requirements.txt
```

## Installation

Python 3.8+ is recommended. CUDA is strongly recommended for image generation.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your system from the official PyTorch selector if the default package is not appropriate for your CUDA version.

## Usage

Run commands from the repository root.

### Generate Base Images

```bash
python scripts/run_pipeline.py --do_base --prompts_csv data/1k.csv --max_count 100
```

### Generate Modified Images

```bash
python scripts/run_pipeline.py --do_modified --prompts_csv data/1k.csv --modified_prompts_csv data/modified_1k.csv --max_count 100
```

### Compute Metrics

```bash
python scripts/run_pipeline.py --do_metrics --csv_out outputs/metrics_results.csv
```

### Run the Full Pipeline

```bash
python scripts/run_pipeline.py --do_base --do_modified --do_metrics --max_count 100
```

The original entrypoint still works:

```bash
python main.py --do_metrics
```

## CLI Arguments

- `--device`: Device to use. Defaults to `cuda` when PyTorch detects CUDA, otherwise `cpu`.
- `--prompts_csv`: Base prompt CSV. Default: `data/1k.csv`.
- `--modified_prompts_csv`: Modified prompt CSV. Default: `data/modified_1k.csv`.
- `--max_count`: Maximum number of prompts or images to process.
- `--seed`: Random seed. Default: `1024`.
- `--steps`: Diffusion inference steps. Default: `30`.
- `--guidance`: Classifier-free guidance scale. Default: `7.5`.
- `--phash_thresh`: pHash distance threshold. Default: `30`.
- `--template_thresh`: Template matching threshold. Default: `0.80`.
- `--orb_nfeatures`: ORB feature count. Default: `1500`.
- `--orb_ratio_thresh`: ORB ratio test threshold. Default: `0.7`.
- `--orb_ransac_reproj`: RANSAC reprojection threshold. Default: `3.0`.
- `--do_base`: Generate base image sets.
- `--do_modified`: Generate modified image sets.
- `--do_metrics`: Compute metrics for configured folder pairs.
- `--csv_out`: Metrics CSV output path. Default: `outputs/metrics_results.csv`.

## Outputs

- `outputs/queries/`: Base model image folders.
- `outputs/modifiedqueries/`: Modified image folders.
- `outputs/metrics_results.csv`: Metric report for configured comparison pairs.

Generated outputs are intentionally not committed. Paths are relative to the repository root so the project can be moved between machines.

## Citation

If you use this code or the paper in your research, please cite:

```bibtex
@inproceedings{josyabhatla2026query,
  title={Query-Based Fingerprinting of Text-to-Image Large Language Models},
  author={Josyabhatla, Saketh Ram and Son, Junggab and Kim, Daeyoung},
  booktitle={12th IEEE International Conference on Big Data Security on Cloud (IEEE BigDataSecurity 2026)},
  year={2026},
  month={May}
}
```
