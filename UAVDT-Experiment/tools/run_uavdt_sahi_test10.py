
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

print("Running 10-image SAHI test...")
print("Mode: Sliced inference")
print("Slice size: 640 x 640")
print("Overlap: 25%")

result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=CONFIG_PATH,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    source=IMAGE_DIR,
    no_standard_prediction=True,
    no_sliced_prediction=False,
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
    name="UAVDT_SAHI_test10",
    verbose=1,
    return_dict=True,
    force_postprocess_type=True,
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

print("10-image SAHI evaluation completed.")
