import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


APP_DIR = "/app"
SRC_DIR = os.path.join(APP_DIR, "src")
SEED_ROOT = os.path.join(APP_DIR, "seed")
WEIGHT_ROOT = os.path.join(APP_DIR, "weight")
ADV_EVAL_DIR = os.path.join(APP_DIR, "adv_eval")
ACC_RESULT_DIR = os.path.join(APP_DIR, "ACC_result")
WORK_ROOT = os.path.join(APP_DIR, "work")

DEFAULT_WEIGHT_STEM = os.path.join(
    WEIGHT_ROOT,
    "inception_v3_google-0cc3c7bd"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate clean ACC for seed dataset with inception_v3."
    )
    parser.add_argument(
        "mission_id",
        help="Mission id. The script will read /app/seed/<mission_id>.zip."
    )
    parser.add_argument(
        "weight",
        nargs="?",
        default="None",
        help=(
            "If None, use default /app/weight/inception_v3_google-0cc3c7bd(.pth/.pt). "
            "If not None, use /app/weight/<mission_id>.zip."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation. Default: 32."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0."
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep extracted temporary work directory."
    )
    return parser.parse_args()


def is_none_arg(value):
    return value is None or str(value).strip().lower() == "none"


def validate_safe_token(name, value):
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        raise ValueError(
            f"Invalid {name}: {value}. Allowed characters: A-Z a-z 0-9 _ . -"
        )


def ensure_dirs():
    os.makedirs(ADV_EVAL_DIR, exist_ok=True)
    os.makedirs(ACC_RESULT_DIR, exist_ok=True)
    os.makedirs(WORK_ROOT, exist_ok=True)


def unzip_to(zip_path, extract_dir):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_first_dir(root, dirname):
    for current_root, dirs, _ in os.walk(root):
        for d in dirs:
            if d == dirname:
                return os.path.join(current_root, d)
    return ""


def find_first_file(root, filename):
    for current_root, _, files in os.walk(root):
        for f in files:
            if f == filename:
                return os.path.join(current_root, f)
    return ""


def prepare_seed(mission_id, work_dir):
    seed_zip = os.path.join(SEED_ROOT, f"{mission_id}.zip")
    seed_extract_dir = os.path.join(work_dir, "seed")

    print(f"Seed zip: {seed_zip}")
    print(f"Extracting seed zip to: {seed_extract_dir}")

    unzip_to(seed_zip, seed_extract_dir)

    image_dir = find_first_dir(seed_extract_dir, "img")
    label_csv = find_first_file(seed_extract_dir, "images.csv")

    if not image_dir or not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory 'img' not found after extracting {seed_zip}"
        )

    if not label_csv or not os.path.isfile(label_csv):
        raise FileNotFoundError(
            f"Label file 'images.csv' not found after extracting {seed_zip}"
        )

    print(f"Image dir: {image_dir}")
    print(f"Label CSV: {label_csv}")

    return image_dir, label_csv


def normalize_state_dict(obj):
    """
    Compatible checkpoint formats:
    1. pure state_dict
    2. {"state_dict": ...}
    3. {"model": ...}
    4. DataParallel keys: module.xxx
    """
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]

    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]

    if not isinstance(obj, dict):
        raise ValueError("Unsupported checkpoint format.")

    return {
        k.replace("module.", ""): v
        for k, v in obj.items()
    }


def resolve_default_weight_path():
    candidates = [
        DEFAULT_WEIGHT_STEM,
        f"{DEFAULT_WEIGHT_STEM}.pth",
        f"{DEFAULT_WEIGHT_STEM}.pt",
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Default weight file not found. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def prepare_weight(mission_id, weight_arg, work_dir):
    if is_none_arg(weight_arg):
        weight_path = resolve_default_weight_path()
        print(f"Using default weight: {weight_path}")
        return weight_path

    weight_zip = os.path.join(WEIGHT_ROOT, f"{mission_id}.zip")
    weight_extract_dir = os.path.join(work_dir, "weight")

    print(f"Weight zip: {weight_zip}")
    print(f"Extracting weight zip to: {weight_extract_dir}")

    unzip_to(weight_zip, weight_extract_dir)

    candidates = []
    for current_root, _, files in os.walk(weight_extract_dir):
        for f in files:
            if f.lower().endswith((".pth", ".pt")):
                candidates.append(os.path.join(current_root, f))

    candidates.sort()

    if not candidates:
        raise FileNotFoundError(
            f"Weight zip exists but no .pth or .pt file found: {weight_zip}"
        )

    weight_path = candidates[0]
    print(f"Using uploaded weight: {weight_path}")

    return weight_path


class SeedImageDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform):
        self.image_dir = image_dir
        self.label_csv = label_csv
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not os.path.isfile(self.label_csv):
            raise FileNotFoundError(f"Label CSV not found: {self.label_csv}")

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if not self.images:
            raise RuntimeError(f"No images found in directory: {self.image_dir}")

        self.labels = {}
        with open(self.label_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            fieldnames = set(reader.fieldnames or [])
            required = {"ImageId", "TrueLabel"}
            missing = required - fieldnames

            if missing:
                raise ValueError(
                    f"images.csv must contain columns {required}, missing: {missing}"
                )

            for row in reader:
                image_id = str(row["ImageId"])
                # 项目原逻辑：TrueLabel 是 1-based，模型输出是 0-based，所以这里减 1
                true_label = int(row["TrueLabel"]) - 1
                self.labels[image_id] = true_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        image_id = os.path.splitext(filename)[0]

        if image_id not in self.labels:
            raise KeyError(
                f"ImageId '{image_id}' not found in label CSV: {self.label_csv}"
            )

        true_label = self.labels[image_id]

        return image, true_label, filename


def build_inception_v3(weight_path, device):
    model = tv_models.inception_v3(weights=None)

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = normalize_state_dict(checkpoint)

    result = model.load_state_dict(state_dict, strict=False)

    if getattr(result, "missing_keys", None):
        if len(result.missing_keys) > 0:
            print(f"Missing keys when loading model: {result.missing_keys}")

    if getattr(result, "unexpected_keys", None):
        if len(result.unexpected_keys) > 0:
            print(f"Unexpected keys when loading model: {result.unexpected_keys}")

    model = model.to(device)
    model.eval()

    return model


def write_outputs(mission_id, acc, rows):
    eval_path = os.path.join(ADV_EVAL_DIR, f"eval_{mission_id}.txt")
    result_path = os.path.join(ACC_RESULT_DIR, f"ACC_{mission_id}.txt")

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"ACC: {acc:.2f}\n")

    with open(result_path, "w", encoding="utf-8") as f:
        for filename, true_idx, pred_idx in rows:
            f.write(f"{filename} {true_idx} {pred_idx}\n")

    print(f"ACC eval saved to: {eval_path}")
    print(f"ACC detail saved to: {result_path}")


def main():
    args = parse_args()

    validate_safe_token("mission_id", args.mission_id)

    if not is_none_arg(args.weight):
        validate_safe_token("weight", args.weight)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    ensure_dirs()

    work_dir = os.path.join(WORK_ROOT, f"ACC_{args.mission_id}_{os.getpid()}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        image_dir, label_csv = prepare_seed(args.mission_id, work_dir)
        weight_path = prepare_weight(args.mission_id, args.weight, work_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Using device: {device}")

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

        dataset = SeedImageDataset(
            image_dir=image_dir,
            label_csv=label_csv,
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        model = build_inception_v3(weight_path, device)

        total = 0
        correct = 0
        rows = []

        with torch.no_grad():
            for images, true_labels, filenames in dataloader:
                images = images.to(device)
                true_labels = true_labels.to(device)

                logits = model(images)
                pred_labels = torch.argmax(logits, dim=1)

                correct += torch.sum(pred_labels == true_labels).item()
                total += int(true_labels.shape[0])

                true_cpu = true_labels.detach().cpu().tolist()
                pred_cpu = pred_labels.detach().cpu().tolist()

                for filename, true_idx, pred_idx in zip(filenames, true_cpu, pred_cpu):
                    #适配imagenet索引
                    rows.append((str(filename), int(true_idx)+1, int(pred_idx)+1))

        if total <= 0:
            raise RuntimeError("Dataset is empty, cannot calculate ACC.")

        acc = 100.0 * correct / total

        write_outputs(args.mission_id, acc, rows)

        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"ACC: {acc:.2f}")

    finally:
        if args.keep_work:
            print(f"Work directory kept: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
