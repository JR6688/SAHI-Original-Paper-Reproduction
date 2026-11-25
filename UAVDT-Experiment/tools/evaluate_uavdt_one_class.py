
import sys
from pathlib import Path
from sahi.scripts.coco_evaluation import evaluate

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python3 tools/evaluate_uavdt_one_class.py "
        "baseline|sahi car|truck|bus"
    )

mode = sys.argv[1]
category = sys.argv[2]

if mode not in ("baseline", "sahi"):
    raise SystemExit("Invalid mode")

if category not in ("car", "truck", "bus"):
    raise SystemExit("Invalid category")

base = Path("/workspace/runs") / (
    "UAVDT_baseline_batches"
    if mode == "baseline"
    else "UAVDT_SAHI_batches"
)

folder = base / "class_eval"

gt_file = folder / f"{category}_gt.json"
pred_file = folder / f"{category}_pred.json"

print("Mode:", mode)
print("Class:", category)
print("Ground truth:", gt_file)
print("Predictions:", pred_file)
print("Starting COCO evaluation...", flush=True)

evaluate(
    dataset_json_path=str(gt_file),
    result_json_path=str(pred_file),
    classwise=False,
    max_detections=500,
    return_dict=False,
)

print("Evaluation completed:", mode, category)
