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


class SiameseDataset(Dataset):
    def __init__(self, anchor_paths, validation_paths):
        self.anchor_paths, self.validation_paths, self.labels = self.generate_balanced_pairs(anchor_paths, validation_paths)

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, index):
        anchor_img = preprocess(self.anchor_paths[index])
        validation_img = preprocess(self.validation_paths[index])
        label = self.labels[index]
        return anchor_img, validation_img, label

    def generate_balanced_pairs(self, anchor_paths, validation_paths):
        pairs_by_class = {}
        for anchor, validation in zip(anchor_paths, validation_paths):
            anchor_class = os.path.basename(os.path.dirname(anchor))
            validation_class = os.path.basename(os.path.dirname(validation))
            
            if anchor_class not in pairs_by_class:
                pairs_by_class[anchor_class] = {"positive": [], "negative": []}
            
            if anchor_class == validation_class:
                pairs_by_class[anchor_class]["positive"].append((anchor, validation, 1))
            else:
                pairs_by_class[anchor_class]["negative"].append((anchor, validation, 0))
        
        anchor_output = []
        validation_output = []
        label_output = []
        
        for class_data in pairs_by_class.values():
            positive_samples = class_data["positive"]
            negative_samples = class_data["negative"]
            
            min_samples = min(len(positive_samples), len(negative_samples))
            
            for anchor, validation, label in positive_samples[:min_samples]:
                anchor_output.append(anchor)
                validation_output.append(validation)
                label_output.append(label)
            
            for anchor, validation, label in negative_samples[:min_samples]:
                anchor_output.append(anchor)
                validation_output.append(validation)
                label_output.append(label)
        
        return anchor_output, validation_output, label_output
    

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
                     batch_size=SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE"]):

    TRAIN_PATH = os.path.join(data_dir,'train')
    TEST_PATH = os.path.join(data_dir,'test')
    if SIAMESE_DATA_HYPERPARAMETERS["CLASS_SAMPLE_SIZE"] > -1:
        train_paths = []
        class_sample_count = {}
        for root, dirs, files in os.walk(TRAIN_PATH):
            for file_path in glob.glob(os.path.join(root, '*.jpg')):
                class_name = os.path.basename(os.path.dirname(file_path))
                if class_name not in class_sample_count:
                    class_sample_count[class_name] = 0
                if class_sample_count[class_name] < SIAMESE_DATA_HYPERPARAMETERS["CLASS_SAMPLE_SIZE"]:
                    train_paths.append(file_path)
                    class_sample_count[class_name] += 1
                    if len(train_paths) >= SIAMESE_DATA_HYPERPARAMETERS["NUM_CLASSES"] * SIAMESE_DATA_HYPERPARAMETERS["CLASS_SAMPLE_SIZE"]:
                        break
    else:
        train_paths = [file_path for root, dirs, files in os.walk(TRAIN_PATH) for file_path in glob.glob(os.path.join(root, '*.jpg'))]
    all_pairs = np.array(np.meshgrid(train_paths, train_paths)).T.reshape(-1, 2)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    anchor_paths = all_pairs[:, 0]
    validation_paths = all_pairs[:, 1]
    dataset = SiameseDataset(anchor_paths, validation_paths)
    train_size = int((1-val_split) * len(dataset))
    val_size = (len(dataset) - train_size)
    train_data, val_data= torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    one_shot_data = [[preprocess(files_in_subfolder[0]), os.path.basename(sub_folder)] for sub_folder in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, sub_folder)) and (files_in_subfolder := glob.glob(os.path.join(TRAIN_PATH, sub_folder, '*.jpg')))]
    test_data = [[preprocess(file), os.path.basename(sub_folder)] for sub_folder in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, sub_folder)) for file in glob.glob(os.path.join(TEST_PATH, sub_folder, '*.jpg'))]

    total_images = len(train_data) + len(val_data) + len(test_data)
    print(f"Total number of images: {total_images}")
    print(f"Number of training images: {len(train_data)} ({100 * len(train_data) / total_images:>2f}%)")
    print(f"Number of validation images: {len(val_data)} ({100 * len(val_data) / total_images:>2f}%)")
    print(f"Number of test images: {len(test_data)} ({100 * len(test_data) / total_images:>2f}%)")
    
    labels_map = DATA_HYPERPARAMETERS["CLASSES"]
    print(f"\nClasses: {labels_map}")

    return train_data_loader, val_data_loader, test_data, one_shot_data
