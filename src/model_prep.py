import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_prep import iris_df
from filepath import TEMP_DIR

TRAIN_IMAGE_FOLDER = os.path.join(TEMP_DIR, "train/")
VALID_IMAGE_FOLDER = os.path.join(TEMP_DIR, "valid/")
TEST_IMAGE_FOLDER = os.path.join(TEMP_DIR, "test/")


class LabelImageGroupDataset(Dataset):
    def __init__(self, X: list[tuple[str, pd.DataFrame]], img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.label_groups = []
        filenames = []
        for row in X:
            self.label_groups.append(row[0])
            for filename in os.listdir(os.path.join(self.img_dir, row[0])):
                filenames.append(filename.split(".")[0])

        self.le = LabelEncoder()
        self.le.fit(filenames)

    def __len__(self):
        return len(self.label_groups)

    def __getitem__(self, idx):
        label_path = os.path.join(self.img_dir, self.label_groups[idx])

        images = []
        labels = []

        for filename in os.listdir(label_path):
            images.append(Image.open(os.path.join(label_path, filename)))
            labels.append(self.le.transform([filename.split(".")[0]])[0])
        labels = torch.tensor(labels)

        groups = torch.tensor([idx for i in range(len(images))])

        if self.transform:
            images = torch.stack([self.transform(image) for image in images], dim=0)

        return images, groups, labels

    def get_label_str(self, labels: list[int]) -> list[str]:
        return self.le.inverse_transform(labels)


def split_data(label_count: int = -1):
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

    with open(os.path.join(TEMP_DIR, "train.pkl"), "wb") as file:
        pickle.dump(X_train, file)
    with open(os.path.join(TEMP_DIR, "valid.pkl"), "wb") as file:
        pickle.dump(X_valid, file)
    with open(os.path.join(TEMP_DIR, "test.pkl"), "wb") as file:
        pickle.dump(X_test, file)

    return X_train, X_valid, X_test


def get_data():
    with open(os.path.join(TEMP_DIR, "train.pkl"), "rb") as file:
        X_train = pickle.load(file)
    with open(os.path.join(TEMP_DIR, "valid.pkl"), "rb") as file:
        X_valid = pickle.load(file)
    with open(os.path.join(TEMP_DIR, "test.pkl"), "rb") as file:
        X_test = pickle.load(file)

    return X_train, X_valid, X_test


def create_dataset(
    X_train, X_valid, X_test
) -> tuple[LabelImageGroupDataset, LabelImageGroupDataset, LabelImageGroupDataset]:
    transform = transforms.Compose(
        [
            transforms.Resize((640, 480)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    TrainDataset = LabelImageGroupDataset(X_train, TRAIN_IMAGE_FOLDER, transform=transform)
    ValidDataset = LabelImageGroupDataset(X_valid, VALID_IMAGE_FOLDER, transform=transform)
    TestDataset = LabelImageGroupDataset(X_test, TEST_IMAGE_FOLDER, transform=transform)

    return TrainDataset, ValidDataset, TestDataset


def create_dataset_loader(
    TrainDataset, ValidDataset, TestDataset, batch_size=4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    TrainDatasetLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    ValidDatasetLoader = DataLoader(ValidDataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    TestDatasetLoader = DataLoader(TestDataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return TrainDatasetLoader, ValidDatasetLoader, TestDatasetLoader


def get_dataset_loader(batch_size=4):
    X_train, X_valid, X_test = get_data()
    TrainDataset, ValidDataset, TestDataset = create_dataset(X_train, X_valid, X_test)

    return create_dataset_loader(TrainDataset, ValidDataset, TestDataset, batch_size=batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Model Data Preparation Script",
        description="Prepare data for model training and evaluating",
    )
    parser.add_argument("-n", "--label-count", default=-1, help="The amount of label to use")
    args = parser.parse_args()
    label_count = int(args.label_count)

    print(f"Spliting data with f{label_count} label")
    X_train, X_valid, X_test = split_data(label_count=label_count)
    print(len(X_train), len(X_valid), len(X_test))
