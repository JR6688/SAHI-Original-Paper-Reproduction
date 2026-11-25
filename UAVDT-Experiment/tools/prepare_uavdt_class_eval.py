
import json
from pathlib import Path

ROOT = Path("/workspace")
GT_FILE = ROOT / "data/UAVDT/coco/test_3class.json"
BATCH_DIR = ROOT / "data/UAVDT/coco/batches"
RUNS = ROOT / "runs"

MODES = {
    "baseline": RUNS / "UAVDT_baseline_batches",
    "sahi": RUNS / "UAVDT_SAHI_batches",
}

with open(GT_FILE, "r", encoding="utf-8") as f:
    gt = json.load(f)

categories = gt["categories"]
images = gt["images"]

for category in categories:
    cid = category["id"]
    name = category["name"]

    class_gt = {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": images,
        "annotations": [
            a for a in gt["annotations"]
            if a["category_id"] == cid
        ],
        "categories": [category],
    }

    for mode, batch_dir in MODES.items():
        out_dir = batch_dir / "class_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        gt_out = out_dir / f"{name}_gt.json"
        pred_out = out_dir / f"{name}_pred.json"

        with open(gt_out, "w", encoding="utf-8") as f:
            json.dump(class_gt, f)

        total = 0

        with open(pred_out, "w", encoding="utf-8") as out:
            out.write("[")
            first = True

            for i in range(1, 109):
                batch_file = (
                    batch_dir / f"batch_{i:03d}" / "result.json"
                )

                if not batch_file.exists():
                    raise FileNotFoundError(batch_file)

                with open(batch_file, "r", encoding="utf-8") as f:
                    predictions = json.load(f)

                for p in predictions:
                    if p["category_id"] != cid:
                        continue

                    if not first:
                        out.write(",")

                    json.dump(p, out)
                    first = False
                    total += 1

            out.write("]")

        print(
            f"{mode} | {name} | "
            f"GT: {len(class_gt['annotations'])} | "
            f"Predictions: {total}"
        )

print("\nThree-class evaluation files prepared.")
