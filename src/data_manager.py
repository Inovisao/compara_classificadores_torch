#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains a class which is responsible for managing the data for the model.
""" 

from hyperparameters import DATA_HYPERPARAMETERS, DATA_AUGMENTATION

import numpy as np
import os
import pathlib
from PIL import Image
from sklearn.model_selection import train_test_split
import tifffile as tiff
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms


def get_transforms(image_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"], 
                   data_augmentation=DATA_HYPERPARAMETERS["USE_DATA_AUGMENTATION"]):
    # Create a transforms pipeline. It may only resize the images,
    # resize and apply data augmentation, and, in both cases, it may or may not also normalize the data.
    if data_augmentation:
        transforms_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),    
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(p=DATA_AUGMENTATION["HORIZONTAL_FLIP"]),
            transforms.RandomVerticalFlip(p=DATA_AUGMENTATION["VERTICAL_FLIP"]),
            transforms.RandomRotation(degrees=DATA_AUGMENTATION["ROTATION"]),
            transforms.RandomPerspective(),
            #transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transforms_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if normalize else None,
        ])

    return transforms_pipeline


class CustomDataset(Dataset):
    def __init__(self, data_dir, filenames, labels, transform=None):
        self.data_dir = data_dir
        self.filenames = filenames
        self.labels = labels
        self.labels_map = os.listdir(data_dir)
        self.transform = transform

    def __getitem__(self, idx):

        assert self.labels_map == DATA_HYPERPARAMETERS["CLASSES"], "Problem with the class list..."

        # Get the label corresponding to the index.
        label = self.labels[idx]

        # Get the filename corresponding to the index.
        filename = self.filenames[idx]

        # Create the filepath.
        filepath = os.path.join(self.data_dir, label, filename)

        # Get the extension for one image.
        ext = pathlib.Path(filepath).suffix

        # If needed, the following lines can be changed to load images with other extensions with other packages.
        if ext == ".tiff" or ext == ".tif":
            image = transforms.functional.to_tensor(tiff.imread(filepath).astype(np.int32) / 255.)
        else:
            image = transforms.functional.to_tensor(Image.open(filepath))

        # Get the index of the label from the map of labels.
        label_num = torch.tensor(self.labels_map.index(label))

        # Apply transformations, if any has been specified.
        if self.transform:
            image = self.transform(image)
        
        # Return one image, its label and its filename.
        return image, label_num, filename

    def __len__(self):
        return len(self.filenames)


def read_images(data_dir, subset):
    """
    Args:
        data_dir (str): the root directory, which must contain at least the train and test subdirectories.
        subset (str): either train or test.

    Returns:
        dataset: a CustomDataset object, which can be passed into a dataloader.
    """
    # Create the path for the subset (/train or /test).
    subset_directory = os.path.join(data_dir, subset)

    # Empty lists to which the filenames and the labels will be appended.
    filenames = []
    labels = []

    for root, directories, files in os.walk(subset_directory):
        # Ignore the subset root directory.
        if not (root == subset_directory):
            # Get the names of the files and their corresponding labels.
            for file in files:
                filenames.append(file)
                img_label = root.replace(subset_directory, "")
                img_label = img_label.replace("/", "")
                labels.append(img_label)
        
    # Apply data_augmentation only for training and validation sets.
    if subset != "test":
        dataset = CustomDataset(data_dir=subset_directory,
                                filenames=filenames,
                                labels=labels,
                                transform=get_transforms())
    else:
        dataset = CustomDataset(data_dir=subset_directory,
                                filenames=filenames,
                                labels=labels,
                                transform=get_transforms(data_augmentation=False))
    
    return dataset


def print_data_informations(train_data, val_data, test_data, train_dataloader):
    for X, y, _ in train_dataloader:
        print(f"Images batch size: {X.shape[0]}")
        print(f"Number of channels: {X.shape[1]}")
        print(f"Height: {X.shape[2]}")
        print(f"Width: {X.shape[3]}")
        print(f"Labels batch size: {y.shape[0]}")
        print(f"Label data type: {y.dtype}")
        break
    
    total_images = len(train_data) + len(val_data) + len(test_data)
    print(f"Total number of images: {total_images}")
    print(f"Number of training images: {len(train_data)} ({100 * len(train_data) / total_images:>2f}%)")
    print(f"Number of validation images: {len(val_data)} ({100 * len(val_data) / total_images:>2f}%)")
    print(f"Number of test images: {len(test_data)} ({100 * len(test_data) / total_images:>2f}%)")
    labels_map = DATA_HYPERPARAMETERS["CLASSES"]
    print(f"\nClasses: {labels_map}")

    
def get_data(data_dir=DATA_HYPERPARAMETERS["ROOT_DATA_DIR"], 
             val_split=DATA_HYPERPARAMETERS["VAL_SPLIT"], 
             batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"]):
    """ This function is used to get the data to the model.

    Args:
        data_dir (str): the root data directory, which must contain at least the train and test subdirectories.
        val_split (float): percentage of the train dataset to be used for validation.
        batch_size (int): number of images used in each feedforward step.

    Returns:
        dict: a dict with three dataloaders, one for training, one for validation and another one for test.
    """

    train_dataset = read_images(data_dir, "train")
    test_dataset = read_images(data_dir, "test")

    # Get indexes for training and validation.
    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=val_split)

    # Apply the indexes and get the images for both training and validation sets.
    val_dataset = Subset(train_dataset, val_idx)
    train_dataset = Subset(train_dataset, train_idx)
    
    # Create the loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print_data_informations(train_dataset, val_dataset, test_dataset, train_dataloader)

    return train_dataloader, val_dataloader, test_dataloader
