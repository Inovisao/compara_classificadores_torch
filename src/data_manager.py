#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains a class which is responsible for managing the data for the model.
""" 

from hyperparameters import DATA_HYPERPARAMETERS, DATA_AUGMENTATION, SIAMESE_DATA_HYPERPARAMETERS

import numpy as np
import os
import cv2
import glob
import random
import pathlib
from PIL import Image
from sklearn.model_selection import train_test_split
import tifffile as tiff
import torch
from torch.nn import Identity
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms


def get_transforms(image_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"], 
                   data_augmentation=DATA_HYPERPARAMETERS["USE_DATA_AUGMENTATION"],
                   for_gradcam=False):
    # Create a transforms pipeline. It may only resize the images,
    # resize and apply data augmentation, and, in both cases, it may or may not also normalize the data.
    if data_augmentation:
        transforms_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(),
            transforms.RandomGrayscale(),
            transforms.RandomInvert(),
            transforms.RandomSolarize(threshold=0.75),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize() if DATA_AUGMENTATION["RAND_EQUALIZE"] else Identity(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(p=DATA_AUGMENTATION["HORIZONTAL_FLIP"]),
            transforms.RandomVerticalFlip(p=DATA_AUGMENTATION["VERTICAL_FLIP"]),
            transforms.RandomRotation(degrees=DATA_AUGMENTATION["ROTATION"]),
            transforms.RandomPerspective(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if DATA_HYPERPARAMETERS["NORMALIZE"] else Identity(),
        ])
    elif (for_gradcam):
        transforms_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if DATA_HYPERPARAMETERS["NORMALIZE"] else Identity(),
        ])
    else:
        transforms_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if DATA_HYPERPARAMETERS["NORMALIZE"] else Identity(),
        ])

    return transforms_pipeline


def preprocess(file_path):
    transform = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(img, (SIAMESE_DATA_HYPERPARAMETERS["IMAGE_SIZE"], SIAMESE_DATA_HYPERPARAMETERS["IMAGE_SIZE"]))
    resized_img = resized_img.astype(np.float32) / 255.0
    transformed_img = transform(resized_img)
    return transformed_img


class SiameseDatasetCls(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels_map = DATA_HYPERPARAMETERS["CLASSES"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, target_label = self.data[idx]
        input_data = preprocess(file_path)
        target_idx = self.labels_map.index(target_label)
        return input_data, target_idx



class SiameseDatasetRec(Dataset):
    def __init__(self, anchor_ids, validation_ids, labels):
        self.anchor_ids = anchor_ids
        self.validation_ids = validation_ids
        self.labels = labels

    def __len__(self):
        return len(self.anchor_ids)

    def __getitem__(self, index):
        anchor_id = self.anchor_ids[index]
        validation_id = self.validation_ids[index]
        label = self.labels[index]
        return anchor_id, validation_id, label

    

class CustomDataset(Dataset):
    def __init__(self, data_dir, filenames, labels, transform=None):
        self.data_dir = data_dir
        self.filenames = filenames
        self.labels = labels
        self.labels_map = DATA_HYPERPARAMETERS["CLASSES"]
        self.transform = transform

    def __getitem__(self, idx):

        #assert self.labels_map == DATA_HYPERPARAMETERS["CLASSES"], "Problem with the class list..."

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
            image = transforms.functional.to_tensor(tiff.imread(filepath).astype(np.int32) / DATA_HYPERPARAMETERS["DATA_SCALE_FACTOR"])
        else:
            image = transforms.functional.to_tensor(Image.open(filepath)) / DATA_HYPERPARAMETERS["DATA_SCALE_FACTOR"]

        # Get the index of the label from the map of labels.
        label_num = torch.tensor(self.labels_map.index(label))

        # Apply transformations, if any has been specified.
        if self.transform:
            image = self.transform(image)
        
        # Return one image, its label and its filename.
        return image, label_num, filename

    def __len__(self):
        return len(self.filenames)


def read_images(data_dir, subset, is_siamese):
    """
    Args:
        data_dir (str): the root directory, which must contain at least the train and test subdirectories.
        subset (str): either train or test.
        is_siamese (bool): Whether to use SiameseDatasetCls instead of CustomDataset. Defaults to False.

    Returns:
        dataset: a CustomDataset or SiameseDatasetCls object, which can be passed into a dataloader.
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
        if is_siamese:
            dataset = SiameseDatasetCls(data=list(zip(filenames, labels)))
        else:
            dataset = CustomDataset(data_dir=subset_directory,
                                    filenames=filenames,
                                    labels=labels,
                                    transform=get_transforms())
    else:
        if is_siamese:
            dataset = SiameseDatasetCls(data=list(zip(filenames, labels)))
        else:
            dataset = CustomDataset(data_dir=subset_directory, # For testing
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
             batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"],
             is_siamese=False):
    """ This function is used to get the data to the model.

    Args:
        data_dir (str): the root data directory, which must contain at least the train and test subdirectories.
        val_split (float): percentage of the train dataset to be used for validation.
        batch_size (int): number of images used in each feedforward step.
        is_siamese (bool, optional): Whether to use SiameseDatasetCls instead of CustomDataset. Defaults to False.

    Returns:
        dict: a dict with three dataloaders, one for training, one for validation, and another one for test.
    """

    train_dataset = read_images(data_dir, "train", is_siamese)
    test_dataset = read_images(data_dir, "test", is_siamese)
    

    # Get indexes for training and validation.
    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=val_split)

    # Apply the indexes and get the images for both training and validation sets.
    val_dataset = Subset(train_dataset, val_idx)
    train_dataset = Subset(train_dataset, train_idx)
    
    # Create the loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    
    print_data_informations(train_dataset, val_dataset, test_dataset, train_dataloader)

    return train_dataloader, val_dataloader, test_dataloader


def get_siamese_data(data_dir=SIAMESE_DATA_HYPERPARAMETERS["ROOT_DATA_DIR"], 
                     val_split=SIAMESE_DATA_HYPERPARAMETERS["VAL_SPLIT"], 
                     batch_size_rec=SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_REC"],
                     batch_size_cls=SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_CLS"]):
    """This function is used to prepare data for a Siamese network model.

    Args:
        data_dir (str): The root data directory containing at least the 'train' and 'test' subdirectories.
                                  Defaults to the value specified in SIAMESE_DATA_HYPERPARAMETERS.
        val_split (float): The percentage of the training dataset to be used for validation.
                                      Defaults to the value specified in SIAMESE_DATA_HYPERPARAMETERS.
        batch_size_rec (int): The batch size for the Siamese network's training dataset.
                                         Defaults to the value specified in SIAMESE_DATA_HYPERPARAMETERS.
        batch_size_cls (int): The batch size for the classifier network's dataloaders.
                                         Defaults to the value specified in SIAMESE_DATA_HYPERPARAMETERS.

    Returns:
        tuple: A tuple containing four dataloaders - one for the recognition network's training dataset,
               one for the classifier network's training dataset, one for validation, and one for testing.
    """

    # Step 1: Get dataloaders using get_data function
    train_dataloader_cls, val_dataloader, test_dataloader = get_data(data_dir=data_dir, val_split=val_split, batch_size=batch_size_cls, is_siamese=True)

    # Step 2: Form pairs for train_dataloader_rec
    train_paths = [data[0] for data in train_dataloader_cls.dataset.data]
    labels_map = DATA_HYPERPARAMETERS["CLASSES"]
    class_sample_count = {class_name: 0 for class_name in labels_map}
    anchor_paths = []
    validation_paths = []
    labels = []

    for anchor_path in train_paths:
        anchor_class = os.path.basename(os.path.dirname(anchor_path))
        if class_sample_count[anchor_class] < SIAMESE_DATA_HYPERPARAMETERS["CLASS_SAMPLE_SIZE"]:
            positive_samples = [data for data in train_dataloader_cls.dataset.data if data[1] == anchor_class]
            negative_samples = [data for data in train_dataloader_cls.dataset.data if data[1] != anchor_class]

            # Balance positive and negative samples
            min_samples = min(len(positive_samples), len(negative_samples))

            for _ in range(min_samples):
                anchor = anchor_path
                positive_sample = random.choice(positive_samples)[0]
                negative_sample = random.choice(negative_samples)[0]

                anchor_id = train_paths.index(anchor)
                positive_id = train_paths.index(positive_sample)
                negative_id = train_paths.index(negative_sample)

                anchor_paths.append(anchor_id)
                validation_paths.append(positive_id)
                labels.append(1)  # Same class
                anchor_paths.append(anchor_id)
                validation_paths.append(negative_id)
                labels.append(0)  # Different class

            class_sample_count[anchor_class] += 1

    train_data = SiameseDatasetRec(anchor_paths, validation_paths, labels)
    train_dataloader_rec = DataLoader(train_data, batch_size=batch_size_rec, shuffle=True, num_workers=14)

    # Step 3: Print information
    total_images = len(train_dataloader_cls.dataset) + len(val_dataloader.dataset) + len(test_dataloader.dataset)
    print(f"Total number of images: {total_images}")
    print(f"Number of training images: {len(train_dataloader_cls.dataset)} ({100 * len(train_dataloader_cls.dataset) / total_images:.2f}%)")
    print(f"Number of validation images: {len(val_dataloader.dataset)} ({100 * len(val_dataloader.dataset) / total_images:.2f}%)")
    print(f"Number of test images: {len(test_dataloader.dataset)} ({100 * len(test_dataloader.dataset) / total_images:.2f}%)")

    print(f"\nClasses: {labels_map}")

    return train_dataloader_rec, train_dataloader_cls, val_dataloader, test_dataloader
