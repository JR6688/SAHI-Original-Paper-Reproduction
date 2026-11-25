
from pathlib import Path
from datetime import datetime
import sys
import shutil
import subprocess
import time
import os


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "demostration result"

DOCKER_IMAGE = "sahi-original"


def run_prediction(image_path, output_name):

    import torch
    from sahi.predict import predict

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\nSelected image: {image_path.name}")
    print(f"Device: {device}")

    start_time = time.perf_counter()

    result = predict(
        model_type="mmdet",
        model_path="/workspace/checkpoints/fcos_sf_visdrone.pth",
        model_config_path="/workspace/mmdet_configs/visdrone_fcos/fcos_crop_480_960_cls_60.py",
        model_confidence_threshold=0.01,
        model_device=device,
        source=str(image_path),
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
        novisual=False,
        dataset_json_path=None,
        project="/tmp/sahi_demo_output",
        name=output_name,
        visual_export_format="png",
        verbose=1,
        return_dict=True,
        force_postprocess_type=True,
    )

    elapsed = time.perf_counter() - start_time

    export_dir = Path(result["export_dir"])
    image_results = list(export_dir.rglob("*.png"))

    if not image_results:
        raise FileNotFoundError("Predicted image was not generated.")

    prediction_image = image_results[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save original image without replacing previous images
    original_image = OUTPUT_DIR / f"{output_name}_original{image_path.suffix}"
    shutil.copy2(image_path, original_image)

    # Save predicted image without replacing previous images
    predicted_image = OUTPUT_DIR / f"{output_name}_predicted.png"
    shutil.copy2(prediction_image, predicted_image)

    print("\nPrediction completed!")
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Original image saved: {original_image.name}")
    print(f"Predicted image saved: {predicted_image.name}")


def windows_launcher():

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    selected = filedialog.askopenfilename(
        title="Select an image for SAHI detection",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
        ],
    )

    root.destroy()

    if not selected:
        print("No image selected.")
        return

    selected = Path(selected)

    # Unique name for every prediction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_name = f"{selected.stem}_{timestamp}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Temporarily copy image into the project root for Docker
    temporary_image = PROJECT_DIR / f"demo_input{selected.suffix}"
    shutil.copy2(selected, temporary_image)

    docker_image_path = f"/workspace/{temporary_image.name}"

    command = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_DIR}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3", "-u",
        "/workspace/demo_single_image.py",
        "--inside-docker",
        docker_image_path,
        output_name,
    ]

    print("\nStarting SAHI prediction...\n")

    try:
        completed = subprocess.run(command, cwd=PROJECT_DIR)

        if completed.returncode != 0:
            print("\nPrediction failed. Check the error above.")
            return

    finally:
        if temporary_image.exists():
            temporary_image.unlink()

    predicted_image = OUTPUT_DIR / f"{output_name}_predicted.png"

    if predicted_image.exists():
        print("\nOpening predicted image...")
        os.startfile(predicted_image)


if __name__ == "__main__":

    if len(sys.argv) >= 4 and sys.argv[1] == "--inside-docker":
        run_prediction(Path(sys.argv[2]), sys.argv[3])
    else:
        windows_launcher()
