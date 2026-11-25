
import sys
from pathlib import Path

from sahi.predict import predict
from sahi.scripts.coco_evaluation import evaluate

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else ""

if MODE not in ("baseline", "sahi"):
    raise SystemExit(
        "Usage: python3 run_uavdt_eval.py baseline"
        " OR python3 run_uavdt_eval.py sahi"
    )

MODEL_PATH = "/author/checkpoints/fcos_sf_visdrone.pth"
MODEL_CONFIG_PATH = (
    "/author/mmdet_configs/visdrone_fcos/"
    "fcos_crop_480_960_cls_60.py"
)

IMAGE_DIR = "/workspace/data/UAVDT/DatasetNinja/test/img"
DATASET_JSON = "/workspace/data/UAVDT/coco/test_3class.json"

if MODE == "baseline":
    NO_STANDARD = False
    NO_SLICED = True
    NAME = "UAVDT_baseline"
else:
    NO_STANDARD = True
    NO_SLICED = False
    NAME = "UAVDT_SAHI_640_25"

print("Mode:", MODE)
print("Checkpoint:", MODEL_PATH)
print("Dataset:", DATASET_JSON)

result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    model_category_mapping=None,
    model_category_remapping=None,
    source=IMAGE_DIR,
    no_standard_prediction=NO_STANDARD,
    no_sliced_prediction=NO_SLICED,
    image_size=None,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    novisual=True,
    dataset_json_path=DATASET_JSON,
    project="/workspace/runs",
    name=NAME,
    verbose=1,
    return_dict=True,
    force_postprocess_type=True,
)

result_json_path = str(
    Path(result["export_dir"]) / "result.json"
)

print("Predictions saved:", result_json_path)
print("Evaluation folder:", result["export_dir"])

evaluate(
    dataset_json_path=DATASET_JSON,
    result_json_path=result_json_path,
    classwise=True,
    max_detections=500,
    return_dict=False,
)

print("Evaluation completed:", MODE)
