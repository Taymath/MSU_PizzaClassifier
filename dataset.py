import os
import glob
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


def get_data_loaders(data_config):
    classes_csv = pd.read_csv(data_config.classes_csv)
    num_classes = len(classes_csv)
    print(f"num_classes from classes.csv = {num_classes}")

    train_csv = pd.read_csv(data_config.train_csv)

    train_loader, val_loader, _ = create_dataloaders(
        train_csv=train_csv,
        train_img_dir=data_config.train_img_dir,
        test_csv=None,
        test_img_dir=None,
        batch_size=data_config.batch_size,
        image_size=data_config.image_size,
        val_size=data_config.val_size,
        num_workers=data_config.num_workers,
        stratify=data_config.stratify,
        random_state=data_config.random_state
    )
    return train_loader, val_loader


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train'):
        self.csv = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.has_labels = 'label' in self.csv.columns

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        image_name = row['image_name']
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode in ['train', 'val'] and self.has_labels:
            label = row['label']
            return image, label
        return image


def create_dataloaders(
    train_csv,
    train_img_dir,
    test_csv=None,
    test_img_dir=None,
    batch_size=24,
    image_size=256,
    val_size=0.25,
    num_workers=0,
    stratify=False,
    random_state=42
):
    if stratify and 'label' in train_csv.columns:
        train_df, val_df = train_test_split(
            train_csv,
            test_size=val_size,
            random_state=random_state,
            stratify=train_csv['label']
        )
    else:
        train_df, val_df = train_test_split(
            train_csv,
            test_size=val_size,
            random_state=random_state
        )

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=25),
        T.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0)),
        T.ColorJitter(contrast=0.25),
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    train_dataset = ImageDataset(train_df, train_img_dir, transform=train_transform, mode='train')
    val_dataset = ImageDataset(val_df, train_img_dir, transform=test_transform, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = None
    if test_csv is not None and test_img_dir is not None:
        test_dataset = ImageDataset(test_csv, test_img_dir, transform=test_transform, mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, val_loader, test_loader


class TestDataset(Dataset):

    def __init__(self, test_img_dir, submission_csv, transform=None):
        super().__init__()
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(test_img_dir, "*.jpg"))
        self.submission_df = pd.read_csv(submission_csv)

        self.submission_df.set_index("id", inplace=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]

        filename = os.path.basename(path)
        id_str, _ = os.path.splitext(filename)
        id_str = id_str.replace("test_", "")
        img_id = int(id_str)

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        true_label = self.submission_df.loc[img_id, "label"]
        return image, img_id, true_label


def create_test_dataloader_with_labels(
    test_img_dir,
    submission_csv,
    batch_size=24,
    image_size=256,
    num_workers=0
):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    dataset = TestDataset(test_img_dir, submission_csv, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return loader
