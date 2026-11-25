
import json
import sys
from pathlib import Path

from sahi.predict import predict

MODEL_PATH = "/author/checkpoints/fcos_sf_visdrone.pth"
CONFIG_PATH = (
    "/author/mmdet_configs/visdrone_fcos/"
    "fcos_crop_480_960_cls_60.py"
)

IMAGE_DIR = "/workspace/data/UAVDT/DatasetNinja/test/img"
BATCH_DIR = Path("/workspace/data/UAVDT/coco/batches")
RUN_DIR = Path("/workspace/runs")

if len(sys.argv) != 2 or sys.argv[1] not in ("baseline", "sahi"):
    raise SystemExit(
        "Usage: python3 tools/run_uavdt_batches.py baseline"
        " OR sahi"
    )

mode = sys.argv[1]

if mode == "baseline":
    output_root = RUN_DIR / "UAVDT_baseline_batches"
else:
    output_root = RUN_DIR / "UAVDT_SAHI_batches"

batch_files = sorted(BATCH_DIR.glob("batch_*.json"))

print("Mode:", mode, flush=True)
print("Number of batches:", len(batch_files), flush=True)

for index, batch_file in enumerate(batch_files, start=1):
    batch_name = batch_file.stem
    result_json = output_root / batch_name / "result.json"

    if result_json.exists() and result_json.stat().st_size > 2:
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                json.load(f)
            print(
                f"[{index}/{len(batch_files)}] "
                f"Skipping completed {batch_name}",
                flush=True,
            )
            continue
        except (json.JSONDecodeError, OSError):
            print("Invalid previous result; rerunning:", batch_name)

    print(
        f"\n[{index}/{len(batch_files)}] Processing {batch_name}",
        flush=True,
    )

    common_args = dict(
        model_type="mmdet",
        model_path=MODEL_PATH,
        model_config_path=CONFIG_PATH,
        model_confidence_threshold=0.01,
        model_device="cuda:0",
        source=IMAGE_DIR,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        novisual=True,
        dataset_json_path=str(batch_file),
        project=str(output_root),
        name=batch_name,
        verbose=1,
        return_dict=True,
    )

    if mode == "baseline":
        common_args.update(
            no_standard_prediction=False,
            no_sliced_prediction=True,
        )
    else:
        common_args.update(
            no_standard_prediction=True,
            no_sliced_prediction=False,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            force_postprocess_type=True,
        )

    result = predict(**common_args)

    saved_result = Path(result["export_dir"]) / "result.json"

    if not saved_result.exists():
        raise RuntimeError(
            f"Missing prediction output for {batch_name}"
        )

    print(
        f"Completed {batch_name}: {saved_result}",
        flush=True,
    )

print("\nAll batches completed for mode:", mode, flush=True)
