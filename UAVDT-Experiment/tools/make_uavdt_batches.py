
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]

source = root / "data/UAVDT/coco/test_3class.json"
output_dir = root / "data/UAVDT/coco/batches"

BATCH_SIZE = 500

with open(source, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

output_dir.mkdir(parents=True, exist_ok=True)

for start in range(0, len(images), BATCH_SIZE):
    batch_images = images[start:start + BATCH_SIZE]
    image_ids = {img["id"] for img in batch_images}

    batch_annotations = [
        ann for ann in annotations
        if ann["image_id"] in image_ids
    ]

    batch = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": batch_images,
        "annotations": batch_annotations,
        "categories": data["categories"],
    }

    batch_number = start // BATCH_SIZE + 1
    output_file = output_dir / f"batch_{batch_number:03d}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(batch, f)

    print(
        f"Batch {batch_number:03d}: "
        f"{len(batch_images)} images, "
        f"{len(batch_annotations)} annotations"
    )

print("\nBatch preparation completed.")
print("Total images:", len(images))
print("Total batches:", (len(images) + BATCH_SIZE - 1) // BATCH_SIZE)
print("Output:", output_dir)
