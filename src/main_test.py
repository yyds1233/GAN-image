import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import models
import custom_data as cd
from advGAN import AdvGAN_Attack


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Mission-specific hyperparameter JSON path."
    )
    return parser.parse_args()


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y")

    return bool(value)


def load_hyperparameters(config_file):
    with open(config_file, "r", encoding="utf-8") as hp_file:
        cfg = json.load(hp_file)

    required_keys = [
        "mission_id",
        "target_dataset",
        "image_dir",
        "label_csv",
        "checkpoint_dir",
        "loss_dir",
        "adv_image_dir",
        "eval_txt",
        "AdvGAN_epochs",
        "AdvGAN_learning_rate",
        "maximum_perturbation_allowed",
        "alpha",
        "beta",
        "gamma",
        "kappa",
        "c",
        "D_number_of_steps_per_batch",
        "G_number_of_steps_per_batch",
        "is_relativistic"
    ]

    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing keys in config {config_file}: {missing}")

    return cfg


def create_dirs(cfg):
    dirs = [
        cfg["checkpoint_dir"],
        cfg["loss_dir"],
        cfg["adv_image_dir"],
        os.path.dirname(cfg["eval_txt"])
    ]

    if cfg.get("save_npy", False):
        dirs.append(cfg.get("npy_dir", ""))

    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def normalize_state_dict(obj):
    """
    兼容常见权重格式：
    1. 纯 state_dict
    2. {"state_dict": ...}
    3. {"model": ...}
    4. DataParallel 的 module.xxx
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


def load_weight_if_needed(model, weight_path, device):
    if not weight_path:
        return model

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Target weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = normalize_state_dict(checkpoint)

    result = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded target weight: {weight_path}")

    if len(result.missing_keys) > 0:
        print(f"Missing keys when loading target model: {result.missing_keys}")

    if len(result.unexpected_keys) > 0:
        print(f"Unexpected keys when loading target model: {result.unexpected_keys}")

    return model


def build_target_model(model_name, weight_path, device):
    """
    当前项目 HighResolution 原本默认是 Inception v3。
    这里额外预留了 resnet50/vgg16/vgg19，但是否能正常加载自定义权重，
    取决于权重文件和模型结构是否一致。
    """
    model_name = model_name.lower()

    if model_name in ("inception", "inception_v3"):
        if weight_path:
            model = tv_models.inception_v3(weights=None)
        else:
            model = tv_models.inception_v3(weights="DEFAULT")
        input_size = 299
        canonical_name = "inception_v3"

    elif model_name in ("resnet", "resnet50"):
        if weight_path:
            model = tv_models.resnet50(weights=None)
        else:
            model = tv_models.resnet50(weights="DEFAULT")
        input_size = 224
        canonical_name = "resnet50"

    elif model_name == "vgg16":
        if weight_path:
            model = tv_models.vgg16(weights=None)
        else:
            model = tv_models.vgg16(weights="DEFAULT")
        input_size = 224
        canonical_name = "vgg16"

    elif model_name == "vgg19":
        if weight_path:
            model = tv_models.vgg19(weights=None)
        else:
            model = tv_models.vgg19(weights="DEFAULT")
        input_size = 224
        canonical_name = "vgg19"

    else:
        raise NotImplementedError(
            f"Unsupported target_model_name: {model_name}. "
            f"Supported: inception_v3, resnet50, vgg16, vgg19."
        )

    model = load_weight_if_needed(model, weight_path, device)
    model = model.to(device)
    model.eval()

    return model, input_size, canonical_name


def init_params(cfg, device):
    target = cfg["target_dataset"]

    if target != "HighResolution":
        raise NotImplementedError(
            "This engineering version currently supports HighResolution missions. "
            "MNIST/CIFAR10 can be added later if needed."
        )

    batch_size = int(cfg.get("batch_size", 30))
    l_inf_bound = float(cfg["maximum_perturbation_allowed"])
    n_labels = int(cfg.get("n_labels", 1000))
    n_channels = 3

    target_model_name = cfg.get("target_model_name", "inception_v3")
    target_weight_path = cfg.get("target_weight_path", "")

    target_model, input_size, canonical_model_name = build_target_model(
        target_model_name,
        target_weight_path,
        device
    )

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    dataset = cd.HighResolutionDataset(
        main_dir=cfg["image_dir"],
        label_csv=cfg["label_csv"],
        transform=transform
    )

    shuffle = str_to_bool(cfg.get("shuffle", False))
    num_workers = int(cfg.get("num_workers", 0))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return (
        dataloader,
        target_model,
        batch_size,
        l_inf_bound,
        n_labels,
        n_channels,
        len(dataset),
        canonical_model_name
    )


def maybe_save_npy(cfg, filename, array):
    if not cfg.get("save_npy", False):
        return

    npy_dir = cfg.get("npy_dir")
    if not npy_dir:
        return

    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, filename), array)


def test_clean_performance(cfg, dataloader, target_model, dataset_size, device, mode="train"):
    target = cfg["target_dataset"]

    n_correct = 0
    true_labels = []
    pred_labels = []

    target_model.eval()

    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            img, true_label = data
            img = img.to(device)
            true_label = true_label.to(device)

            pred_label = torch.argmax(target_model(img), 1)
            n_correct += torch.sum(pred_label == true_label, 0).item()

            true_labels.append(true_label.detach().cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    maybe_save_npy(cfg, f"{mode}_clean_true_labels.npy", true_labels)
    maybe_save_npy(cfg, f"{mode}_clean_pred_labels.npy", pred_labels)

    print(target)
    print(f"Clean Correctly Classified in full dataset: {n_correct}")
    print(
        "Clean Accuracy in {} full dataset: {}%\n".format(
            target,
            100 * n_correct / dataset_size
        )
    )

    return true_labels, pred_labels, n_correct


def test_attack_performance(
    cfg,
    dataloader,
    mode,
    adv_GAN,
    target_model,
    l_inf_bound,
    dataset_size,
    device
):
    target = cfg["target_dataset"]
    adv_image_dir = cfg["adv_image_dir"]

    os.makedirs(adv_image_dir, exist_ok=True)

    n_correct = 0
    true_labels = []
    pred_labels = []
    img_np = []
    adv_img_np = []

    inv_norm = cd.NormalizeInverse(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    adv_GAN.eval()
    target_model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            img, true_label = data
            img = img.to(device)
            true_label = true_label.to(device)

            perturbation = adv_GAN(img)

            if perturbation.shape[-2:] != img.shape[-2:]:
                perturbation = F.interpolate(
                    perturbation,
                    size=img.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            adv_img = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + img
            adv_img = torch.clamp(adv_img, 0, 1)

            pred_label = torch.argmax(target_model(adv_img), 1)
            n_correct += torch.sum(pred_label == true_label, 0).item()

            true_labels.append(true_label.detach().cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())

            if cfg.get("save_npy", False):
                img_np.append(img.detach().permute(0, 2, 3, 1).cpu().numpy())
                adv_img_np.append(adv_img.detach().permute(0, 2, 3, 1).cpu().numpy())

            true_label_cpu = true_label.detach().cpu().numpy()
            pred_label_cpu = pred_label.detach().cpu().numpy()

            print(f"Saving images for batch {i + 1} out of {len(dataloader)}")

            for j in range(adv_img.shape[0]):
                true_idx = int(true_label_cpu[j])
                pred_idx = int(pred_label_cpu[j])

                if target == "HighResolution":
                    saved_adv = inv_norm(adv_img[j].detach().clone())
                    saved_adv = torch.clamp(saved_adv, 0, 1)
                else:
                    saved_adv = adv_img[j].detach()

                save_path = os.path.join(
                    adv_image_dir,
                    f"example_{i}_{j}_true_{true_idx}_pred_{pred_idx}.png"
                )

                save_image(saved_adv, save_path)

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    maybe_save_npy(cfg, f"{mode}_true_labels.npy", true_labels)
    maybe_save_npy(cfg, f"{mode}_pred_labels.npy", pred_labels)

    if cfg.get("save_npy", False) and len(img_np) > 0:
        img_np = np.concatenate(img_np, axis=0)
        adv_img_np = np.concatenate(adv_img_np, axis=0)
        maybe_save_npy(cfg, f"{mode}_img_np.npy", img_np)
        maybe_save_npy(cfg, f"{mode}_adv_img_np.npy", adv_img_np)

    print(target)
    print(f"Correctly Classified in full dataset under attack: {n_correct}")
    print(
        "Accuracy under attacks in {} full dataset: {}%\n".format(
            target,
            100 * n_correct / dataset_size
        )
    )

    return true_labels, pred_labels, n_correct


def write_eval_txt(
    cfg,
    dataset_size,
    clean_correct,
    adv_correct,
    losses,
    canonical_model_name
):
    eval_txt = cfg["eval_txt"]
    os.makedirs(os.path.dirname(eval_txt), exist_ok=True)

    clean_acc = 100 * clean_correct / dataset_size
    adv_acc = 100 * adv_correct / dataset_size
    acc_drop = clean_acc - adv_acc

    with open(eval_txt, "w", encoding="utf-8") as f:
        f.write(f"mission_id: {cfg['mission_id']}\n")
        f.write(f"target_dataset: {cfg['target_dataset']}\n")
        f.write(f"target_model_name: {canonical_model_name}\n")
        f.write(f"dataset_size: {dataset_size}\n")
        f.write(f"image_dir: {cfg['image_dir']}\n")
        f.write(f"label_csv: {cfg['label_csv']}\n")
        f.write(f"target_weight_path: {cfg.get('target_weight_path', '')}\n")
        f.write("\n")

        f.write("attack_parameters:\n")
        f.write(f"AdvGAN_epochs: {cfg['AdvGAN_epochs']}\n")
        f.write(f"AdvGAN_learning_rate: {cfg['AdvGAN_learning_rate']}\n")
        f.write(f"maximum_perturbation_allowed: {cfg['maximum_perturbation_allowed']}\n")
        f.write(f"alpha: {cfg['alpha']}\n")
        f.write(f"beta: {cfg['beta']}\n")
        f.write(f"gamma: {cfg['gamma']}\n")
        f.write(f"kappa: {cfg['kappa']}\n")
        f.write(f"c: {cfg['c']}\n")
        f.write(f"D_number_of_steps_per_batch: {cfg['D_number_of_steps_per_batch']}\n")
        f.write(f"G_number_of_steps_per_batch: {cfg['G_number_of_steps_per_batch']}\n")
        f.write(f"is_relativistic: {cfg['is_relativistic']}\n")
        f.write("\n")

        f.write("accuracy:\n")
        f.write(f"clean_correct: {clean_correct}\n")
        f.write(f"adv_correct: {adv_correct}\n")
        f.write(f"clean_accuracy: {clean_acc:.4f}\n")
        f.write(f"adv_accuracy: {adv_acc:.4f}\n")
        f.write(f"accuracy_drop: {acc_drop:.4f}\n")
        f.write("\n")

        f.write("losses_by_epoch:\n")

        epoch_count = len(losses.get("loss_D", []))
        for idx in range(epoch_count):
            f.write(
                f"epoch={idx + 1}, "
                f"loss_D={losses['loss_D'][idx]}, "
                f"loss_G={losses['loss_G'][idx]}, "
                f"loss_adv={losses['loss_adv'][idx]}, "
                f"loss_G_gan={losses['loss_G_gan'][idx]}, "
                f"loss_hinge={losses['loss_hinge'][idx]}\n"
            )


def main():
    args = parse_args()

    print("\nLOADING CONFIGURATIONS...")
    cfg = load_hyperparameters(args.config)

    print("\nCREATING NECESSARY DIRECTORIES...")
    create_dirs(cfg)

    print("\nCHECKING FOR CUDA...")
    use_cuda = str_to_bool(cfg.get("use_cuda", True))
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("Using device: ", device)

    print("\nPREPARING DATASET AND TARGET MODEL...")
    (
        dataloader,
        target_model,
        batch_size,
        l_inf_bound,
        n_labels,
        n_channels,
        dataset_size,
        canonical_model_name
    ) = init_params(cfg, device)

    print(f"Mission ID: {cfg['mission_id']}")
    print(f"Target model: {canonical_model_name}")
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}")

    print("\nTESTING CLEAN PERFORMANCE ON FULL DATASET...")
    _, _, clean_correct = test_clean_performance(
        cfg=cfg,
        dataloader=dataloader,
        target_model=target_model,
        dataset_size=dataset_size,
        device=device,
        mode="train"
    )

    print("\nTRAINING ADVGAN...")
    advGAN = AdvGAN_Attack(
        device=device,
        model=target_model,
        n_labels=n_labels,
        n_channels=n_channels,
        target=cfg["target_dataset"],
        lr=float(cfg["AdvGAN_learning_rate"]),
        l_inf_bound=l_inf_bound,
        alpha=float(cfg["alpha"]),
        beta=float(cfg["beta"]),
        gamma=float(cfg["gamma"]),
        kappa=float(cfg["kappa"]),
        c=float(cfg["c"]),
        n_steps_D=int(cfg["D_number_of_steps_per_batch"]),
        n_steps_G=int(cfg["G_number_of_steps_per_batch"]),
        is_relativistic=str_to_bool(cfg["is_relativistic"]),
        checkpoint_dir=cfg["checkpoint_dir"],
        loss_dir=cfg["loss_dir"]
    )

    losses = advGAN.train(
        train_dataloader=dataloader,
        epochs=int(cfg["AdvGAN_epochs"])
    )

    print("\nLOADING TRAINED ADVGAN...")
    final_epoch = int(cfg["AdvGAN_epochs"])
    adv_GAN_path = os.path.join(
        cfg["checkpoint_dir"],
        f"G_epoch_{final_epoch}.pth"
    )

    if not os.path.isfile(adv_GAN_path):
        raise FileNotFoundError(f"Trained AdvGAN generator not found: {adv_GAN_path}")

    adv_GAN = models.Generator(
        n_channels,
        n_channels,
        cfg["target_dataset"]
    ).to(device)

    adv_GAN.load_state_dict(torch.load(adv_GAN_path, map_location=device))
    adv_GAN.eval()

    print("\nTESTING PERFORMANCE OF ADVGAN ON FULL DATASET...")
    _, _, adv_correct = test_attack_performance(
        cfg=cfg,
        dataloader=dataloader,
        mode="train",
        adv_GAN=adv_GAN,
        target_model=target_model,
        l_inf_bound=l_inf_bound,
        dataset_size=dataset_size,
        device=device
    )

    print("\nWRITING EVALUATION TXT...")
    write_eval_txt(
        cfg=cfg,
        dataset_size=dataset_size,
        clean_correct=clean_correct,
        adv_correct=adv_correct,
        losses=losses,
        canonical_model_name=canonical_model_name
    )

    print("\nSUMMARY:")
    print("Full Dataset Clean Accuracy: {:.2f}%".format(100 * clean_correct / dataset_size))
    print("Full Dataset Adv Accuracy: {:.2f}%".format(100 * adv_correct / dataset_size))
    print("Accuracy Drop: {:.2f}%".format(100 * (clean_correct - adv_correct) / dataset_size))
    print(f"Adversarial images saved to: {cfg['adv_image_dir']}")
    print(f"Evaluation txt saved to: {cfg['eval_txt']}")


if __name__ == "__main__":
    main()