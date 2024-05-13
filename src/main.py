#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This is the main file of the program.
    
"""
from arch_optim import architectures, optimizers, gradcam_layer_getters, get_architecture, get_optimizer
import data_manager
import explainers
import helper_functions
from hyperparameters import DATA_HYPERPARAMETERS, MODEL_HYPERPARAMETERS, DATA_AUGMENTATION

import os
import torch
from torch import nn
from torchvision import transforms

########################################################################################################################
########################################################################################################################


def main():
    # Use cuda if it is available. If not, use the CPU.
    device = MODEL_HYPERPARAMETERS["DEVICE"]
    print(f"Using {device}.")

    if MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"]:
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
    model = get_architecture(args["architecture"], 
                             in_channels=DATA_HYPERPARAMETERS["IN_CHANNELS"], 
                             out_classes=DATA_HYPERPARAMETERS["NUM_CLASSES"], 
                             pretrained=MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"])
    
    
    # Send the model to the correct device.
    model = model.to(device)
    print("===================================")
    print("==> MODEL")
    print(model)
    print("===================================")
    print("==> MODEL HYPERPARAMETERS")
    print(MODEL_HYPERPARAMETERS)
    print("===================================")
    print("==> DATA HYPERPARAMETERS")
    print(DATA_HYPERPARAMETERS)
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
                                   epochs=MODEL_HYPERPARAMETERS["NUM_EPOCHS"],
                                   patience=MODEL_HYPERPARAMETERS["PATIENCE"],
                                   tolerance=MODEL_HYPERPARAMETERS["TOLERANCE"],
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
                                                      labels_map=DATA_HYPERPARAMETERS["CLASSES"])

    
    # Create a string with run, learning rate, architecture,
    # optimizer, precision, recall and fscore, to append to the csv file:
    results = str(args["run"]) + "," + str(args["learning_rate"]) + "," + str(args["architecture"]) + \
        "," + str(args["optimizer"]) + "," + str(precision) + "," + str(recall) + "," + str(fscore) + "\n"

    # Open file, write and close.
    f = open("../results_dl/results.csv", "a")
    f.write(results)
    f.close()

    
    print("Explaining results...")

    # Generate explanations
    if ("gradcam" in MODEL_HYPERPARAMETERS["EXPLAINERS"]):
        gradcam_layer = gradcam_layer_getters[args["architecture"]](model)
        if gradcam_layer is None:
            print("GradCAM is not available for this model.")
        else:
            print("Generating GradCAM explanations...")
            if not os.path.exists("../results/gradcam"):
                os.makedirs("../results/gradcam")
        
            if not os.path.exists(f"../results/gradcam/{model_name}"):
                os.makedirs(f"../results/gradcam/{model_name}")

            explainers.generate_gradcam(model, model_name, gradcam_layer, test_dataloader, DATA_HYPERPARAMETERS["CLASSES"], device)



    print("\nFinished execution.")

# Call the main function.
if __name__ == "__main__":
    main()
