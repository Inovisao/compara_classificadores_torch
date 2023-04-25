#/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

ROOT_DATA_DIR = "../data"
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, "test")
CLASSES = sorted(os.listdir(TRAIN_DATA_DIR))
NUM_CLASSES = len(CLASSES)

# Hyperparameters pertaining to the data to be passed into the model.
DATA_HYPERPARAMETERS = {
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 8,
    "VAL_SPLIT": 0.2,
    "USE_DATA_AUGMENTATION": True,
    "DATA_SCALE_FACTOR": 1, # This divides the data when it is read; useful for scaling (e.g., to [0, 1]) 
    "NORMALIZE": True,
    
    "IN_CHANNELS": 3,
    "ROOT_DATA_DIR": ROOT_DATA_DIR,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,
    "CLASSES": CLASSES,
    "NUM_CLASSES": NUM_CLASSES,
}

# No learning rate here. The lr must be set in roda.sh.
MODEL_HYPERPARAMETERS = {
    "NUM_EPOCHS": 1000,
    "PATIENCE": 30,
    "TOLERANCE": 0.1,
    "USE_TRANSFER_LEARNING": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "CREATE_GRADCAM": True, # Avoid using, I'll do something better later

}

# Parameters for data augmentation. If necessary, add more here.
DATA_AUGMENTATION = {
    "HORIZONTAL_FLIP": 0.5,
    "VERTICAL_FLIP": 0.5,
    "ROTATION": 90,
    "RAND_EQUALIZE": False,
}

