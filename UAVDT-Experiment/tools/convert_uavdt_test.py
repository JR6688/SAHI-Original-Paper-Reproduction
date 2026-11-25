
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

ANN_DIR = ROOT / "data" / "UAVDT" / "DatasetNinja" / "test" / "ann"
IMG_DIR = ROOT / "data" / "UAVDT" / "DatasetNinja" / "test" / "img"

VISDRONE_JSON = (
    Path(r"C:\Users\User\Project 1\FYP 1 Code")
    / "SAHI-Original-Paper"
    / "small-object-detection-benchmark"
    / "data" / "visdrone2019" / "coco" / "val.json"
)

OUTPUT = ROOT / "data" / "UAVDT" / "coco" / "test_3class.json"

TARGET_CLASSES = {"car", "truck", "bus"}


def main():
    if not ANN_DIR.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {ANN_DIR}")

    if not IMG_DIR.is_dir():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    if not VISDRONE_JSON.is_file():
        raise FileNotFoundError(
            f"VisDrone COCO JSON not found: {VISDRONE_JSON}"
        )

    with VISDRONE_JSON.open("r", encoding="utf-8-sig") as f:
        visdrone = json.load(f)

    # Keep the original VisDrone category IDs.
    categories = [
        c for c in visdrone["categories"]
        if c["name"].lower() in TARGET_CLASSES
    ]

    category_ids = {
        c["name"].lower(): c["id"] for c in categories
    }

    if set(category_ids) != TARGET_CLASSES:
        raise ValueError(
            f"Expected {TARGET_CLASSES}, found {set(category_ids)}"
        )

    coco = {
        "info": {
            "description": "UAVDT DatasetNinja test split, car/truck/bus",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    class_counts = Counter()
    image_id = 0
    annotation_id = 0
    missing_images = []

    annotation_files = sorted(ANN_DIR.glob("*.json"))

    for ann_path in annotation_files:
        image_name = ann_path.stem
        image_path = IMG_DIR / image_name

        if not image_path.is_file():
            missing_images.append(image_name)
            continue

        with ann_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        width = int(data["size"]["width"])
        height = int(data["size"]["height"])

        image_id += 1

        coco["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        for obj in data.get("objects", []) or []:
            class_name = obj.get("classTitle", "").lower()

            if class_name not in TARGET_CLASSES:
                continue

            points = obj.get("points", {}).get("exterior", [])

            if len(points) < 2:
                continue

            x1, y1 = points[0]
            x2, y2 = points[1]

            x_min = max(0.0, min(float(x1), float(x2)))
            y_min = max(0.0, min(float(y1), float(y2)))
            x_max = min(float(width), max(float(x1), float(x2)))
            y_max = min(float(height), max(float(y1), float(y2)))

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_width <= 0 or box_height <= 0:
                continue

            annotation_id += 1

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[class_name],
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "segmentation": []
            })

            class_counts[class_name] += 1

    if missing_images:
        print("ERROR: Annotation files have no matching images.")
        print("Missing count:", len(missing_images))
        print("Examples:", missing_images[:10])
        raise SystemExit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    print("\nUAVDT COCO conversion completed.")
    print("Images:", len(coco["images"]))
    print("Annotations:", len(coco["annotations"]))
    print("Category IDs:", category_ids)
    print("Class counts:", dict(class_counts))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
