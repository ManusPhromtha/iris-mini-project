import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from data_prep import iris_df

with tempfile.TemporaryDirectory() as tmpdir:
    TRAIN_IMAGE_FOLDER = os.path.join(tmpdir, "train/")
    VALID_IMAGE_FOLDER = os.path.join(tmpdir, "valid/")
    TEST_IMAGE_FOLDER = os.path.join(tmpdir, "test/")


class LabelImageGroupDataset(Dataset):
    def __init__(self, X: list[tuple[str, pd.DataFrame]], img_dir, transform=None, target_transform=None):
        self.labels_str = [row[0] for row in X]
        self.labels_int = [i for i in range(len(X))]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_int)

    def __getitem__(self, idx):
        label_path = os.path.join(self.img_dir, self.labels_str[idx])
        images = [Image.open(os.path.join(label_path, filename)) for filename in os.listdir(label_path)]

        labels = torch.tensor([self.labels_int[idx] for _ in range(len(images))])
        if self.transform:
            images = torch.stack([self.transform(image) for image in images], dim=0)
        if self.target_transform:
            labels = torch.stack([self.target_transform(label) for label in labels], dim=0)
        return images, labels


def getDataset(label_count: int = -1) -> tuple[LabelImageGroupDataset, LabelImageGroupDataset, LabelImageGroupDataset]:
    X = []
    for idx, group in iris_df.groupby(by="Label")[["ImagePath"]]:
        X.append((idx, group))
    X = X[:label_count]

    X_train, X_test, _, _ = train_test_split(X, np.arange(len(X)), test_size=0.40, random_state=42)
    X_valid, X_test, _, _ = train_test_split(X_test, np.arange(len(X_test)), test_size=0.50, random_state=42)

    def prepare_dir(folder_path, image_group):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        os.makedirs(folder_path)

        for label, group in image_group:
            label_path = os.path.join(folder_path, label)
            os.makedirs(label_path)

            for image_path in group["ImagePath"]:
                filename = image_path.split("/")[-1]
                os.symlink(os.path.abspath(image_path), os.path.join(label_path, filename))

    prepare_dir(TRAIN_IMAGE_FOLDER, X_train)
    prepare_dir(VALID_IMAGE_FOLDER, X_valid)
    prepare_dir(TEST_IMAGE_FOLDER, X_test)

    transform = transforms.Compose([transforms.Resize((640, 480)), transforms.Grayscale(1), transforms.ToTensor()])

    TrainDataset = LabelImageGroupDataset(X_train, TRAIN_IMAGE_FOLDER, transform=transform)
    ValidDataset = LabelImageGroupDataset(X_valid, VALID_IMAGE_FOLDER, transform=transform)
    TestDataset = LabelImageGroupDataset(X_test, TEST_IMAGE_FOLDER, transform=transform)

    return TrainDataset, ValidDataset, TestDataset
