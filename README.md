# Query-Based Fingerprinting of Text-to-Image Large Language Models

This repository contains the official implementation of the paper:

**"Query-Based Fingerprinting of Text-to-Image Large Language Models"**  
*Saketh Ram Josyabhatla, Junggab Son, Daeyoung Kim*  
In the 12th IEEE International Conference on Big Data Security on Cloud (IEEE BigDataSecurity 2026), May 2026.

## Overview
This project provides a comprehensive framework to fingerprint and evaluate Text-to-Image (T2I) Large Language Models. It generates image sets using base and modified prompt queries across various open-source Diffusers-based T2I models and calculates similarity metrics to establish a reliable fingerprint.

The framework computes the following metrics between image sets:
- Perceptual Hash (pHash) Accuracy
- Structural Similarity Index Measure (SSIM)
- HSV Histogram Correlation
- ORB Feature Matching Inlier Ratio
- Template Matching

## Prerequisites

- Python 3.8+
- PyTorch (CUDA recommended for faster generation)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- OpenCV, Pillow, numpy, imagehash, scikit-image, tqdm

Install the required dependencies:
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy scikit-image imagehash tqdm
```

## Usage

The main script is `main.py`. It can perform three primary tasks: Base Generation, Modified Generation, and Metric Calculation.

### 1. Generate Base Images
Generates images using standard models (e.g., `stable-diffusion-2-1-base`, `waifu-diffusion`, etc.) from the prompts provided in a CSV file.

```bash
python main.py --do_base --prompts_csv data/1k.csv --max_count 100
```

### 2. Generate Modified Images
Generates images using fine-tuned models, img2img pipelines, or modified prompts to test the robustness of the fingerprints.

```bash
python main.py --do_modified --prompts_csv data/1k.csv --modified_prompts_csv data/modified_1k.csv --max_count 100
```

### 3. Calculate Metrics
Evaluates the similarity between different image sets and outputs a comprehensive CSV report.

```bash
python main.py --do_metrics --csv_out outputs/metrics_results.csv
```

### Full Pipeline execution
You can run all steps at once:
```bash
python main.py --do_base --do_modified --do_metrics --max_count 100
```

## Arguments

* `--device`: Device to use (`cuda` or `cpu`). Default is `cuda` if available.
* `--prompts_csv`: Path to the base prompts CSV file. Default: `data/1k.csv`
* `--modified_prompts_csv`: Path to the modified prompts CSV file. Default: `data/modified_1k.csv`
* `--max_count`: Maximum number of prompts to process.
* `--seed`, `--steps`, `--guidance`: Generation parameters.
* `--do_base`: Flag to execute base image generation.
* `--do_modified`: Flag to execute modified image generation.
* `--do_metrics`: Flag to calculate metrics.
* `--csv_out`: Output path for the metrics CSV file.

## Output Structure

* `outputs/queries/`: Contains generated base image datasets.
* `outputs/modifiedqueries/`: Contains generated modified image datasets.
* `outputs/metrics_results.csv`: A combined CSV containing similarity metric scores between compared sets.

## Citation
If you use this code or our paper in your research, please cite:
```bibtex
@inproceedings{josyabhatla2026query,
  title={Query-Based Fingerprinting of Text-to-Image Large Language Models},
  author={Josyabhatla, Saketh Ram and Son, Junggab and Kim, Daeyoung},
  booktitle={12th IEEE International Conference on Big Data Security on Cloud (IEEE BigDataSecurity 2026)},
  year={2026},
  month={May}
}
```