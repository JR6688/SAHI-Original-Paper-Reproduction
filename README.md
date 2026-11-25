# SAHI Original Paper Reproduction

## Project Overview

This project reproduces and evaluates the original Slicing Aided Hyper Inference (SAHI) methodology for small object detection using the VisDrone2019-DET dataset.

The project uses the FCOS object detection framework with SAHI slicing-based inference to investigate the effectiveness of detecting small objects in aerial images.

## Research Objectives

- Reproduce the original SAHI experimental methodology.
- Evaluate FCOS baseline detection performance.
- Evaluate FCOS with SAHI sliced inference.
- Compare detection performance using evaluation metrics.
- Analyse detection results and inference time.

## Methodology

- **Object detector:** FCOS
- **Small object detection method:** SAHI
- **Dataset:** VisDrone2019-DET
- **Evaluation:** Baseline versus SAHI inference

## Repository Structure

- `eval_tools/` – Evaluation and prediction scripts.
- `mmdet_configs/` – Model and dataset configurations.
- `mmdet_tools/` – Training utilities.
- `visdrone/` – VisDrone-related configuration files.
- `xview/` – xView-related configuration files.
- `author_results/` – Original-author comparison results.
- `runs/` – Experimental evaluation outputs.

## Original Work and Acknowledgement

This project is based on the original SAHI benchmark implementation by Fatih Cagatay Akyon and contributors.

Original repository: https://github.com/fcakyon/small-object-detection-benchmark

SAHI paper: Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection."

This repository contains reproduction experiments, evaluation scripts, and modifications conducted as part of an undergraduate Final Year Project.
