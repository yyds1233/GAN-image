import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class HighResolutionDataset(Dataset):
    def __init__(self, main_dir, label_csv, transform=None):
        self.main_dir = main_dir
        self.label_csv = label_csv
        self.transform = transform

        if not os.path.isdir(self.main_dir):
            raise FileNotFoundError(f"Image directory not found: {self.main_dir}")

        if not os.path.isfile(self.label_csv):
            raise FileNotFoundError(f"Label CSV not found: {self.label_csv}")

        self.all_imgs = sorted([
            file for file in os.listdir(self.main_dir)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.all_imgs) == 0:
            raise RuntimeError(f"No images found in directory: {self.main_dir}")

        self.df = pd.read_csv(self.label_csv, header=0)

        required_columns = {"ImageId", "TrueLabel"}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"Label CSV must contain columns {required_columns}, "
                f"missing: {missing_columns}"
            )

        self.df["ImageId"] = self.df["ImageId"].astype(str)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_file = self.all_imgs[idx]
        img_loc = os.path.join(self.main_dir, img_file)

        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            tensor_image = transforms.ToTensor()(image)

        img_name = os.path.splitext(img_file)[0]
        img_df = self.df.loc[self.df["ImageId"] == img_name]

        if img_df.empty:
            raise KeyError(
                f"ImageId '{img_name}' not found in label CSV: {self.label_csv}"
            )

        # 原项目注释说明：该数据集 TrueLabel 比 ImageNet 索引大 1
        true_label = int(img_df["TrueLabel"].iloc[0]) - 1

        return tensor_image, true_label


def split_dataset(dataset, test_size=.1, shuffle=True):
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    return indices[split:], indices[:split]


class NormalizeInverse(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        invert_norm = transforms.Normalize(
            mean=[-m_elem / s_elem for m_elem, s_elem in zip(self.mean, self.std)],
            std=[1 / elem for elem in self.std]
        )
        return invert_norm(tensor)