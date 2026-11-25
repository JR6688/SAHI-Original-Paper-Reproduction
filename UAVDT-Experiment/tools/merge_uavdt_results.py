
import json
from pathlib import Path
from collections import Counter

from sahi.scripts.coco_evaluation import evaluate

ROOT = Path("/workspace")
GT_JSON = ROOT / "data/UAVDT/coco/test_3class.json"
RUNS = ROOT / "runs"

MODES = {
    "Baseline": RUNS / "UAVDT_baseline_batches",
    "SAHI": RUNS / "UAVDT_SAHI_batches",
}

with open(GT_JSON, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

expected_images = {img["id"] for img in ground_truth["images"]}
category_names = {
    cat["id"]: cat["name"]
    for cat in ground_truth["categories"]
}

for mode, batch_dir in MODES.items():
    print(f"\n{'=' * 55}")
    print(f"PROCESSING {mode.upper()}")
    print(f"{'=' * 55}")

    predictions = []
    missing = []

    for i in range(1, 109):
        result_file = batch_dir / f"batch_{i:03d}" / "result.json"

        if not result_file.exists():
            missing.append(i)
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            batch_predictions = json.load(f)

        predictions.extend(batch_predictions)

    if missing:
        raise RuntimeError(
            f"{mode}: Missing batches: {missing}"
        )

    invalid_images = {
        p["image_id"] for p in predictions
        if p["image_id"] not in expected_images
    }

    if invalid_images:
        raise RuntimeError(
            f"{mode}: Predictions contain unknown image IDs"
        )

    output_file = batch_dir / "merged_result.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print("Completed batches: 108")
    print("Total test images:", len(expected_images))
    print("Total predictions:", len(predictions))
    print("Predictions by class:")

    counts = Counter(
        category_names.get(p["category_id"], "other")
        for p in predictions
    )

    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    print("Merged file:", output_file)
    print(f"\nEvaluating {mode}...")

    evaluate(
        dataset_json_path=str(GT_JSON),
        result_json_path=str(output_file),
        classwise=True,
        max_detections=500,
        return_dict=False,
    )

    print(f"{mode} evaluation completed.")

print("\nBoth full-dataset evaluations completed.")
