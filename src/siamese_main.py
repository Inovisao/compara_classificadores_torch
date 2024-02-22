#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This is the main file of the program.

"""
from arch_optim import architectures, optimizers, gradcam_layer_getters, get_architecture, get_optimizer
import data_manager
import helper_functions
from hyperparameters import SIAMESE_DATA_HYPERPARAMETERS, SIAMESE_MODEL_HYPERPARAMETERS, DATA_AUGMENTATION
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import sigmoid
from torchvision import transforms

#clear gpu cache memory
torch.cuda.empty_cache()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin, device):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, euclidean_distance, target):
        loss_contrastive = torch.mean((target) * torch.log(torch.pow((1+(1/euclidean_distance)),euclidean_distance)) +
                                      (1 - target) * (1-torch.log(torch.pow((1+(1/euclidean_distance)),euclidean_distance))))

        return loss_contrastive

class L2Dist(nn.Module):
    def __init__(self):
        super(L2Dist, self).__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.nn.functional.pairwise_distance(input_embedding, validation_embedding)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model, num_attributes, num_classes):
        super(SiameseNetwork, self).__init__()
        self.recognition_head = nn.Sequential( 
                                    nn.Linear(num_attributes, num_attributes),
                                    nn.ReLU(),
                                    nn.Linear(num_attributes, int(num_attributes/2)))
        self.classification_head = nn.Sequential( 
                                    nn.Linear(num_attributes, num_attributes),
                                    nn.ReLU(),
                                    nn.Linear(num_attributes, num_classes))
        self.embedding = embedding_model
        self.siamese_layer = L2Dist()

    def forward(self, input_image, validation_image):
        input_embedding = self.embedding(input_image)
        validation_embedding = self.embedding(validation_image)
        input_embedding_head = self.recognition_head(input_embedding)
        validation_embedding_head = self.recognition_head(validation_embedding)
        recognition_output = self.siamese_layer(input_embedding_head, validation_embedding_head)
        classification_output= self.classification_head(input_embedding)
        return recognition_output, classification_output

def main():
    # Use cuda if it is available. If not, use the CPU.
    device = SIAMESE_MODEL_HYPERPARAMETERS["DEVICE"]
    print(f"Using {device}.")

    if SIAMESE_MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"]:
        print("Using transfer learning!")
    else:
        print("Not using transfer learning!")

    # Get CLI arguments.
    args = helper_functions.get_args()

    # Assert that the optimizer exists in the list above.
    assert args["optimizer"].casefold() in optimizers, \
        "Optimizer not recognized. Maybe it hasn't been implemented yet."

    # Assert that the architecture exists in the list above.
    assert args["architecture"].casefold() in architectures, \
        "Architecture not recognized. Maybe it hasn't been implemented yet."

    # assert args["architecture"].casefold() in gradcam_layer_getters, \
    #     "No function to get the target layer for the GradCAM found."

    # Get the model.
    embedding_layer = get_architecture(args["architecture"],
                             in_channels=SIAMESE_DATA_HYPERPARAMETERS["IN_CHANNELS"],
                             out_classes=SIAMESE_MODEL_HYPERPARAMETERS["NUM_ATTRIBUTES"],
                             pretrained=SIAMESE_MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"])
    model = SiameseNetwork(embedding_model=embedding_layer,
                           num_attributes=SIAMESE_MODEL_HYPERPARAMETERS["NUM_ATTRIBUTES"],
                           num_classes=SIAMESE_DATA_HYPERPARAMETERS["NUM_CLASSES"])

    # Send the model to the correct device.
    model = model.to(device)
    print("===================================")
    print("==> MODEL")
    print(model)
    print("===================================")
    print("==> MODEL HYPERPARAMETERS")
    print(SIAMESE_MODEL_HYPERPARAMETERS)
    print("===================================")
    print("==> DATA HYPERPARAMETERS")
    print(SIAMESE_DATA_HYPERPARAMETERS)
    print("===================================")
    print("==> DATA AUGMENTATION")
    print(DATA_AUGMENTATION)
    print("===================================")

    # Get the optimizer.
    optimizer_rec = get_optimizer(optimizer=args["optimizer"], model=model, learning_rate=(args["learning_rate"]/SIAMESE_MODEL_HYPERPARAMETERS["LR_SCALE_FACTOR"]))
    optimizer_cls = get_optimizer(optimizer=args["optimizer"], model=model, learning_rate=args["learning_rate"])

    try:
        assert optimizer_rec.__name__
    except AttributeError as _:
        optimizer_rec.__name__ = args["optimizer"]
    try:
        assert optimizer_cls.__name__
    except AttributeError as _:
        optimizer_cls.__name__ = args["optimizer"]

    # Get the loss function.
    loss_function_recognition = ContrastiveLoss(margin=SIAMESE_MODEL_HYPERPARAMETERS["MARGIN"],device=SIAMESE_MODEL_HYPERPARAMETERS["DEVICE"])
    loss_function_classification = nn.CrossEntropyLoss()

    # Get the dataloaders.
    train_dataloader_rec, train_dataloader_cls, val_dataloader, test_dataloader = data_manager.get_siamese_data()

    # Give the model a name.
    model_name = "siamese_" + str(args["run"]) + "_" + str(args["architecture"]) + \
        "_" + str(args["optimizer"]) + "_" + str(args["learning_rate"])

    # Create a path to save the model.
    path_to_save = "../model_checkpoints/" + model_name + ".pth"

    history = helper_functions.fit_siamese(train_dataloader_rec=train_dataloader_rec,
                                   train_dataloader_cls=train_dataloader_cls,
                                   val_dataloader=val_dataloader,
                                   model=model,
                                   optimizer_rec=optimizer_rec,
                                   optimizer_cls=optimizer_cls,
                                   loss_fn_rec=loss_function_recognition,
                                   loss_fn_cls=loss_function_classification,
                                   epochs=SIAMESE_MODEL_HYPERPARAMETERS["NUM_EPOCHS"],
                                   patience=SIAMESE_MODEL_HYPERPARAMETERS["PATIENCE"],
                                   tolerance=SIAMESE_MODEL_HYPERPARAMETERS["TOLERANCE"],
                                   path_to_save=path_to_save)

    # Define the paths to save the history files.
    path_to_history_csv = "../results/history/" + model_name + "_HISTORY.csv"
    path_to_history_png = "../results/history/" + model_name + "_HISTORY.png"

    # Save the history as csv.

    # Load the best weights for testing.
    model.load_state_dict(torch.load(path_to_save))

    path_to_matrix_csv = "../results/matrix/" + model_name + "_MATRIX.csv"
    path_to_matrix_png = "../results/matrix/" + model_name + "_MATRIX.png"

    # Test, save the results and get precision, recall and fscore.
    precision, recall, fscore = helper_functions.test_siamese(test_dataloader=test_dataloader,
                                                      model=model,
                                                      path_to_save_matrix_csv=path_to_matrix_csv,
                                                      path_to_save_matrix_png=path_to_matrix_png,
                                                      labels_map=SIAMESE_DATA_HYPERPARAMETERS["CLASSES"])


    # Create a string with run, learning rate, architecture,
    # optimizer, precision, recall and fscore, to append to the csv file:
    results = str(args["run"]) + "," + str(args["learning_rate"]) + "," + "siamese_" + str(args["architecture"]) + \
        "," + str(args["optimizer"]) + "," + str(precision) + "," + str(recall) + "," + str(fscore) + "\n"

    history.to_csv(path_to_history_csv, sep =',')
    # history = pd.read_csv(history, delimiter=',')
    # Plot the history and save as png.
    helper_functions.plot_history_siamese(history, path_to_history_png)
    # Open file, write and close.
    f = open("../results_dl/results.csv", "a")
    f.write(results)
    f.close()

# Call the main function.
if __name__ == "__main__":
    main()
