"""
predict.py – Run YOLOv11 Plant Disease Detection on leaf images

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --image path/to/folder/ --folder --save
    python predict.py --image leaf.jpg --conf 0.3 --save
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = [
    'Tomato_Early_Blight',
    'Tomato_Late_Blight',
    'Tomato_Healthy',
    'Potato_Early_Blight',
    'Potato_Late_Blight',
    'Potato_Healthy',
    'Pepper_Bacterial_Spot',
    'Pepper_Healthy'
]

HEALTHY_CLASSES = {'Tomato_Healthy', 'Potato_Healthy', 'Pepper_Healthy'}


def run_inference(image_path: str, model_path: str = "best.pt", conf: float = 0.25, save: bool = False):
    """
    Run inference on a single leaf image.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained .pt weights
        conf (float): Confidence threshold
        save (bool): Save annotated output image
    """
    print(f"🌿 Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"🔍 Analysing: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        verbose=False
    )

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"\n✅ Detected {len(boxes)} region(s):\n")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                status = "🟢 Healthy" if class_name in HEALTHY_CLASSES else "🔴 Diseased"
                print(f"  [{i+1}] {status} | {class_name} | confidence: {conf_score:.2f}")
        else:
            print("\n⚪ No detections above confidence threshold.")

        result.show()

    if save:
        print(f"\n💾 Saved output to: runs/detect/")

    return results


def run_on_folder(folder_path: str, model_path: str = "best.pt", conf: float = 0.25, save: bool = True):
    """
    Run inference on all images in a folder.
    """
    print(f"🌿 Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"📂 Running on folder: {folder_path}")
    results = model.predict(
        source=folder_path,
        conf=conf,
        save=save,
        verbose=True
    )

    diseased = 0
    healthy = 0
    for result in results:
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else ""
                if name in HEALTHY_CLASSES:
                    healthy += 1
                else:
                    diseased += 1

    print(f"\n📊 Summary:")
    print(f"  🟢 Healthy detections:  {healthy}")
    print(f"  🔴 Diseased detections: {diseased}")
    print(f"  📷 Total images processed: {len(results)}")

    if save:
        print("💾 Annotated images saved to: runs/detect/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Plant Disease Detection – Inference")
    parser.add_argument("--image", type=str, help="Path to leaf image or folder of images")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to model weights (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true", help="Save output images with bounding boxes")
    parser.add_argument("--folder", action="store_true", help="Run on all images in the given folder")

    args = parser.parse_args()

    if not args.image:
        parser.print_help()
        print("\nExample usage:")
        print("  python predict.py --image leaf.jpg")
        print("  python predict.py --image test/images/ --folder --save")
    elif args.folder:
        run_on_folder(args.image, args.model, args.conf, args.save)
    else:
        run_inference(args.image, args.model, args.conf, args.save)
