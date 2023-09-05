#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This is the main file of the program.
    
"""
from arch_optim import architectures, optimizers, gradcam_layer_getters, get_architecture, get_optimizer
import data_manager
import helper_functions
from hyperparameters import SIAMESE_DATA_HYPERPARAMETERS, SIAMESE_MODEL_HYPERPARAMETERS, DATA_AUGMENTATION

import os
import torch
from torch import nn
from torchvision import transforms

########################################################################################################################
########################################################################################################################
class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)
    
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding = embedding_model
        self.siamese_layer = L1Dist()
        self.classifier = nn.Linear(4096, 1)
        self.model_output = nn.Sigmoid()

    def forward(self, input_image, validation_image):
        input_embedding = self.embedding(input_image)
        validation_embedding = self.embedding(validation_image)
        distances = self.siamese_layer(input_embedding, validation_embedding)
        classifier_output = self.classifier(distances)
        output = self.model_output(classifier_output)
        return output

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

    assert args["architecture"].casefold() in gradcam_layer_getters, \
        "No function to get the target layer for the GradCAM found."

    # Get the model.
    embedding = get_architecture(args["architecture"], 
                             in_channels=SIAMESE_DATA_HYPERPARAMETERS["IN_CHANNELS"], 
                             out_classes=4096, 
                             pretrained=SIAMESE_MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"])
    
    model = SiameseNetwork(embedding)
    model = model.to(device)    
    
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
    optimizer = get_optimizer(optimizer=args["optimizer"], model=model, learning_rate=args["learning_rate"])
    
    try:
        assert optimizer.__name__
    except AttributeError as _:
        optimizer.__name__ = args["optimizer"]
        
    # Get the loss function.
    loss_function = nn.CrossEntropyLoss()
    
    # Get the dataloaders.
    train_dataloader, val_dataloader, test_dataloader = data_manager.get_data()
    
    # Give the model a name.
    model_name = str(args["run"]) + "_" + str(args["architecture"]) + \
        "_" + str(args["optimizer"]) + "_" + str(args["learning_rate"])

    # Create a path to save the model.
    path_to_save = "../model_checkpoints/" + model_name + ".pth"
    
    history = helper_functions.fit(train_dataloader=train_dataloader,
                                   val_dataloader=val_dataloader,
                                   model=model,
                                   optimizer=optimizer,
                                   loss_fn=loss_function,
                                   epochs=SIAMESE_MODEL_HYPERPARAMETERS["NUM_EPOCHS"],
                                   patience=SIAMESE_MODEL_HYPERPARAMETERS["PATIENCE"],
                                   tolerance=SIAMESE_MODEL_HYPERPARAMETERS["TOLERANCE"],
                                   path_to_save=path_to_save)
    
    # Define the paths to save the history files.
    path_to_history_csv = "../results/history/" + model_name + "_HISTORY.csv"
    path_to_history_png = "../results/history/" + model_name + "_HISTORY.png"

    # Save the history as csv.
    history.to_csv(path_to_history_csv)
    # Plot the history and save as png.
    helper_functions.plot_history(history, path_to_history_png)
    
    # Load the best weights for testing.
    model.load_state_dict(torch.load(path_to_save))

    # Define the paths to save the confusion matrix files.
    path_to_matrix_csv = "../results/matrix/" + model_name + "_MATRIX.csv"    
    path_to_matrix_png = "../results/matrix/" + model_name + "_MATRIX.png"

    # Test, save the results and get precision, recall and fscore.
    precision, recall, fscore = helper_functions.test(dataloader=test_dataloader,
                                                      model=model, 
                                                      path_to_save_matrix_csv=path_to_matrix_csv, 
                                                      path_to_save_matrix_png=path_to_matrix_png,
                                                      labels_map=SIAMESE_DATA_HYPERPARAMETERS["CLASSES"])

    
    # Create a string with run, learning rate, architecture,
    # optimizer, precision, recall and fscore, to append to the csv file:
    results = str(args["run"]) + "," + str(args["learning_rate"]) + "," + str(args["architecture"]) + \
        "," + str(args["optimizer"]) + "," + str(precision) + "," + str(recall) + "," + str(fscore) + "\n"

    # Open file, write and close.
    f = open("../results_dl/results.csv", "a")
    f.write(results)
    f.close()

    # # Create the GradCAM files.
    # if SIAMESE_MODEL_HYPERPARAMETERS["CREATE_GRADCAM"] and SIAMESE_DATA_HYPERPARAMETERS["IN_CHANNELS"] == 3 and args["architecture"] != "ielt":
        
    #     test_dataloader.dataset.transform = transforms.Compose([
    #         transforms.Resize((SIAMESE_DATA_HYPERPARAMETERS["IMAGE_SIZE"], SIAMESE_DATA_HYPERPARAMETERS["IMAGE_SIZE"]))
    #     ])

    #     # Get the target layer for the GradCAM:
    #     gradcam_target_layer = gradcam_layer_getters[args["architecture"]](model=model)

    #     # Create the name of the folder to save the images.
    #     gradcam_filepath = "../results/GradCAM_" + str(args["run"]) + "," + str(args["learning_rate"]) + "," + \
    #         str(args["architecture"]) + "," + str(args["optimizer"])

    #     # Create the directory for the GradCAM files.
    #     os.mkdir(path=gradcam_filepath)

    #     print("Generating GradCAM files for the test images...")

    #     helper_functions.create_gradcam(dataloader=test_dataloader,
    #                                     model=model,
    #                                     target_layer=gradcam_target_layer,
    #                                     subset="test",
    #                                     labels_map=SIAMESE_DATA_HYPERPARAMETERS["CLASSES"],
    #                                     path_to_save=gradcam_filepath)
    # elif SIAMESE_DATA_HYPERPARAMETERS["IN_CHANNELS"] != 3:
    #     print("GradCAM available only for images with 3 channels.")
    # elif args["architecture"] == "ielt":
    #     print("GradCAM not available for the IELT architecture.")
    # else:
    #     print("Skipping GradCAM...")

    # # Confirm that execution has been properly completed.
    # print("\nFinished execution.")


# Call the main function.
if __name__ == "__main__":
    main()
