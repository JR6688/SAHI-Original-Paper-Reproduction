# UAVDT Experiment: Baseline vs SAHI

## 1. Overview

This experiment evaluates the impact of Slicing Aided Hyper Inference (SAHI) on object detection using the UAVDT dataset.

The experiment compares two detection approaches:

- **Baseline:** Object detection on the original full image.
- **SAHI:** Object detection using image slicing and prediction merging.

The purpose is to investigate whether SAHI improves detection performance on UAV aerial images.

## 2. Dataset

- Dataset: UAVDT
- Dataset split: Test set
- Number of test images: 53,676
- Detection classes: Car, Truck and Bus

## 3. Experimental Setup

| Setting | Configuration |
|---|---|
| Detection model | FCOS |
| Model weights | VisDrone-trained checkpoint |
| Baseline inference | Full-image inference |
| SAHI inference | Sliced-image inference |
| Slice size | 640 × 640 |
| Slice overlap ratio | 0.25 |
| Evaluation metrics | mAP, mAP50, mAP75 |

No UAVDT fine-tuning is claimed for this experiment.

## 4. Full Evaluation Results

| Metric | Baseline | SAHI |
|---|---:|---:|
| mAP | 0.102 | **0.203** |
| mAP50 | 0.201 | **0.304** |
| mAP75 | 0.092 | **0.245** |
| mAP-small | **0.081** | 0.067 |
| mAP-medium | 0.168 | **0.362** |
| mAP-large | 0.140 | **0.207** |

### Findings

SAHI achieved higher overall mAP, mAP50 and mAP75 than baseline in the full evaluation.

However, the small-object mAP decreased from 0.081 to 0.067. Therefore, the results indicate that SAHI improved overall detection performance but did not improve the small-object metric in this experiment.

## 5. Preliminary 10-Image Evaluation

| Metric | Baseline | SAHI |
|---|---:|---:|
| mAP | 0.574 | 0.400 |
| mAP50 | 0.981 | 0.738 |
| mAP75 | 0.602 | 0.322 |
| mAP-medium | 0.471 | 0.492 |
| mAP-large | 0.696 | 0.500 |

The preliminary 10-image evaluation produced different results from the full evaluation. It is reported separately and should not be used as a substitute for the full evaluation.

## 6. Repository Structure

```text
UAVDT-Experiment/
├── tools/
│   ├── convert_uavdt_test.py
│   ├── evaluate_uavdt_one_class.py
│   ├── make_uavdt_batches.py
│   ├── make_uavdt_test10.py
│   ├── merge_uavdt_results.py
│   ├── prepare_uavdt_class_eval.py
│   ├── run_uavdt_batches.py
│   ├── run_uavdt_eval.py
│   ├── run_uavdt_sahi_test10.py
│   └── run_uavdt_test10.py
└── results/
    ├── baseline/
    │   ├── full_evaluation.json
    │   ├── test10_evaluation.json
    │   └── test10_result.json
    └── SAHI/
        ├── full_evaluation.json
        ├── test10_evaluation.json
        └── test10_result.json
```

## 7. Results and Reproducibility

The `results/` directory contains evaluation summaries and preliminary prediction outputs.

The `tools/` directory contains Python scripts used for dataset preparation, inference and evaluation.

The original UAVDT dataset and large intermediate prediction files are not included in this repository due to their size.

## 8. Conclusion

The full UAVDT evaluation shows that SAHI improved overall detection performance compared with baseline. However, its small-object mAP was slightly lower.

These findings suggest that the effectiveness of SAHI may vary depending on object scale and dataset characteristics. Further investigation is required to understand its impact on tiny-object detection.
