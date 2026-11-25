
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]

source = root / "data/UAVDT/coco/test_3class.json"
output = root / "data/UAVDT/coco/test_10.json"

with open(source, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"][:10]
image_ids = {image["id"] for image in images}

annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] in image_ids
]

subset = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": images,
    "annotations": annotations,
    "categories": data["categories"],
}

with open(output, "w", encoding="utf-8") as f:
    json.dump(subset, f)

print("Test subset created:", output)
print("Images:", len(images))
print("Annotations:", len(annotations))
print("Categories:", data["categories"])
