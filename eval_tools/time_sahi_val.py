import json
import time
from pathlib import Path

import torch
from sahi.predict import predict

MODEL_PATH = "/workspace/checkpoints/fcos_sf_visdrone.pth"
MODEL_CONFIG_PATH = "/workspace/mmdet_configs/visdrone_fcos/fcos_full_cls_60.py"

IMAGES_PATH = "/workspace/data/visdrone2019/VisDrone2019-DET-val"
ANNOTATION_PATH = "/workspace/data/visdrone2019/coco/val.json"

OUTPUT_PATH = "/workspace/runs/fps_test"
OUTPUT_NAME = "FCOS_SAHI_VISDRONE_VAL"

with open(ANNOTATION_PATH, "r") as f:
    coco_data = json.load(f)

image_count = len(coco_data["images"])

print("Model: FCOS + SAHI")
print("Validation images:", image_count)
print("Slice size: 640 x 640")
print("Overlap: 25%")
print("GPU:", torch.cuda.get_device_name(0))

torch.cuda.synchronize()
start_time = time.perf_counter()

result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    source=IMAGES_PATH,
    no_standard_prediction=True,
    no_sliced_prediction=False,
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
    dataset_json_path=ANNOTATION_PATH,
    project=OUTPUT_PATH,
    name=OUTPUT_NAME,
    verbose=1,
    return_dict=True,
    force_postprocess_type=True,
)

torch.cuda.synchronize()
end_time = time.perf_counter()

total_time = end_time - start_time
average_time = total_time / image_count
fps = image_count / total_time

print("\n========== SAHI FPS RESULTS ==========")
print("Images:", image_count)
print("Total runtime:", round(total_time, 2), "seconds")
print("Average time per image:", round(average_time * 1000, 2), "ms")
print("FPS:", round(fps, 2))
print("======================================")