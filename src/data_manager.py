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
            transforms.Resize((image_size, image_size), antialias=None),
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
            transforms.Resize((image_size, image_size), antialias=None),
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


class SiameseDataset(Dataset):
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    
    print_data_informations(train_dataset, val_dataset, test_dataset, train_dataloader)

    return train_dataloader, val_dataloader, test_dataloader


def get_siamese_data(data_dir=SIAMESE_DATA_HYPERPARAMETERS["ROOT_DATA_DIR"], 
                     val_split=SIAMESE_DATA_HYPERPARAMETERS["VAL_SPLIT"], 
                     batch_size_rec=SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_REC"],
                     batch_size_cls=SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_CLS"]):
    """
    Prepare data for a Siamese network model.

    Args:
        data_dir (str): The root data directory containing at least the 'train' and 'test' subdirectories.
        val_split (float): The percentage of the training dataset to be used for validation.
        batch_size_rec (int): The batch size for the Siamese network's training dataset.
        batch_size_cls (int): The batch size for the classifier network's dataloaders.

    Returns:
        tuple: A tuple containing four dataloaders - one for the recognition network's training dataset,
               one for the classifier network's training dataset, one for validation, and one for testing.
    """
    # Get data loaders for the classifier network
    train_dataloader_cls, val_dataloader, test_dataloader = get_data(data_dir, val_split, batch_size_cls)

    # Get class labels map
    labels_map = SIAMESE_DATA_HYPERPARAMETERS["CLASSES"]
    
    # Sample pairs for recognition dataloader
    anchor_ids = []
    validation_ids = []
    labels = []
    class_sample_size = SIAMESE_DATA_HYPERPARAMETERS["CLASS_SAMPLE_SIZE"]

    # Pair generation logic
    for class_label in labels_map:
        print(f'Processing pairs for class {class_label}')
        positives = []
        negatives = []

        for batch_index, batch in enumerate(train_dataloader_cls):
            _, batch_labels, _ = batch
            for id_inside_batch, label in enumerate(batch_labels):
                if label == labels_map.index(class_label):
                    positives.append([batch_index, id_inside_batch])
                else:
                    negatives.append([batch_index, id_inside_batch])      
        positives = torch.tensor(positives)
        negatives = torch.tensor(negatives)
        
        positive_pairs = []
        negative_pairs = []
        for anchor_id in positives:
            for positive_validation_id in positives:
                positive_pair = [anchor_id, positive_validation_id, 1]
                positive_pairs.append(positive_pair)
            for negative_validation_id in negatives:
                negative_pair = [anchor_id, negative_validation_id, 0]
                negative_pairs.append(negative_pair)
        
        positive_size = len(positive_pairs)
        negative_size = len(negative_pairs)
        min_size = min(positive_size, negative_size)
        if class_sample_size > 0:
            min_size = min(min_size, class_sample_size)
            sampled_indices = torch.randperm(positive_size)[:min_size]
            positive_pairs = [positive_pairs[i] for i in sampled_indices]
            sampled_indices = torch.randperm(negative_size)[:min_size]
            negative_pairs = [negative_pairs[i] for i in sampled_indices]
        else:
            if positive_size > negative_size:
                sampled_indices = torch.randperm(positive_size)[:min_size]
                positive_pairs = [positive_pairs[i] for i in sampled_indices]
            else:
                sampled_indices = torch.randperm(negative_size)[:min_size]
                negative_pairs = [negative_pairs[i] for i in sampled_indices]

        for idx in range(len(positive_pairs)):
            # Fill from positive_pairs
            positive_pair = positive_pairs[idx]
            anchor_ids.append(positive_pair[0])
            validation_ids.append(positive_pair[1])
            labels.append(positive_pair[2])
            
            # Fill from negative_pairs
            negative_pair = negative_pairs[idx]
            anchor_ids.append(negative_pair[0])
            validation_ids.append(negative_pair[1])
            labels.append(negative_pair[2])
    print('Pair creation for all classes finished')

    # Create Siamese dataset for recognition (only file paths)
    train_dataset_rec = SiameseDataset(anchor_ids, validation_ids, labels)
    train_dataloader_rec = DataLoader(train_dataset_rec, batch_size=batch_size_rec, shuffle=True, num_workers=14)

    # Validate dataset population
    assert len(train_dataset_rec) > 0, "Recognition dataset is empty"

    return train_dataloader_rec, train_dataloader_cls, val_dataloader, test_dataloader