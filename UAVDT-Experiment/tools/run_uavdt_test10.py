
from pathlib import Path

from sahi.predict import predict
from sahi.scripts.coco_evaluation import evaluate

MODEL_PATH = "/author/checkpoints/fcos_sf_visdrone.pth"
CONFIG_PATH = (
    "/author/mmdet_configs/visdrone_fcos/"
    "fcos_crop_480_960_cls_60.py"
)

IMAGE_DIR = "/workspace/data/UAVDT/DatasetNinja/test/img"
DATASET_JSON = "/workspace/data/UAVDT/coco/test_10.json"

print("Running 10-image baseline test...")
print("Mode: Full-image inference (no slicing)")
print("Checkpoint:", MODEL_PATH)

result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=CONFIG_PATH,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    source=IMAGE_DIR,
    no_standard_prediction=False,
    no_sliced_prediction=True,
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.5,
    novisual=True,
    dataset_json_path=DATASET_JSON,
    project="/workspace/runs",
    name="UAVDT_baseline_test10",
    verbose=1,
    return_dict=True,
)

result_json = str(Path(result["export_dir"]) / "result.json")

print("Predictions saved:", result_json)

evaluate(
    dataset_json_path=DATASET_JSON,
    result_json_path=result_json,
    classwise=True,
    max_detections=500,
    return_dict=False,
)

print("10-image baseline evaluation completed.")
