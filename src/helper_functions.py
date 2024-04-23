#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from hyperparameters import MODEL_HYPERPARAMETERS, DATA_HYPERPARAMETERS, SIAMESE_MODEL_HYPERPARAMETERS, SIAMESE_DATA_HYPERPARAMETERS
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn
from sklearn import metrics
import torch
from torchmetrics import Recall, Precision, F1Score
from torch.nn.functional import softmax
from torchvision import transforms
from data_manager import get_transforms

device = MODEL_HYPERPARAMETERS["DEVICE"]


def get_args():
    """
    This function gets the arguments of the program, that is, the architecture, the optimizer and the learning rate.

    Returns: a dictionary with the values of the arguments, in which the keys are the names defined for each argument in the second argument of each of the functions below.
    """

    # Instantiate the argument parser.
    arg_parser = argparse.ArgumentParser()

    # Parse the architecture.
    arg_parser.add_argument("-a", "--architecture", required=True, default=None, type=str)
    
    # Parse the optimizer.
    arg_parser.add_argument("-o", "--optimizer", required=True, default=None, type=str)

    # Parse the number of the run.
    arg_parser.add_argument("-r", "--run", required=True, default=None, type=int)
    
    # Parse the learning rate.
    arg_parser.add_argument("-l", "--learning_rate", required=True, default=None, type=float)

    # Parse the arguments and return them as a dictionary.
    return vars(arg_parser.parse_args())


def train(dataloader, model, loss_fn, optimizer):
    """
    This function is used to train a model. It is not called by itself, but inside the 'fit' function below.

    Args:
        dataloader: the training dataloader.
        model: the model to be trained.
        loss_fn: the loss function to be used.
        optimizer: the optimizer to be used.

    Returns: the training loss and the accuracy.

    """
    # Calculate the total number of images (which will be necessary below, to calculate the accuracy).
    size = len(dataloader.dataset)

    # Get the number of batches.
    num_batches = len(dataloader)

    # Puts the model in training mode.
    model.train()

    # Initialize the loss and the accuracy with value 0.
    train_loss, train_accuracy = 0, 0

    # Initialize the number of correct predictions as 0.
    num_correct = 0

    # Iterate over the batches with a counter (enumeration).
    for batch, (X, y, _) in enumerate(dataloader):
        # Send the images and the labels to the device.
        X, y = X.to(device, dtype=torch.float), y.to(device)
        
        if optimizer.__name__ == "sam":
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            train_loss += loss.item()
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        else:
            # Make predictions using the current weights.
            pred = model(X)
            
            # Calculate the loss with the predictions and the true values.
            loss = loss_fn(pred, y)

            # Accumulate the loss to calculate the mean training loss.
            train_loss += loss_fn(pred, y).item()

            # Accumulate the number of correct predictions to calculate the accuracy during training.
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Calculate the gradient for each trainable parameter.
            loss.backward()

            # Use the optimizer to optimize the model.
            optimizer.step()

            # Set the parameter gradients to zero, so that it doesn't get accumulated for the backward step.
            optimizer.zero_grad()

        if batch == 0:
            batch_size = len(X)

        # Show training information every n batches.
        if batch % 10 == 0:
            loss_2_show = loss.item()
            print("Batch:", batch)
            current = (batch * batch_size) + len(X)
            print(f"Loss: {loss_2_show:>7f} [{current:>5d} / {size:>5d}]")

    # Calculate the mean loss during training.
    train_loss /= num_batches

    # Calculate the accuracy during training.
    train_accuracy = num_correct / size

    # Return training loss and accuracy.
    return train_loss, train_accuracy


def validation(dataloader, model, loss_fn):
    """
    This function is used to validate the training. Like the train function, it is not called by itself, but by the fit function.

    Args:
        dataloader: the validation dataloader.
        model: the model whose training must be validated.
        loss_fn: the loss function to be used.

    Returns: the mean validation loss and the accuracy.

    """
    # Calculate the total number of images.
    size = len(dataloader.dataset)

    # Calculate the number of batches.
    num_batches = len(dataloader)

    # Put the model in evaluation mode.
    model.eval()

    # Initialize the validation loss and the number of correct predictions with value 0.
    val_loss, num_correct = 0, 0

    # Proceed without gradient calculation (this reduces the charge on the memory).
    with torch.no_grad():
        # Iterate over the dataloader.
        for X, y, _ in dataloader:
            # Send the images and the labels to the correct device.
            X, y = X.to(device, dtype=torch.float), y.to(device)

            # Make predictions with the model.
            pred = model(X)

            # Calculate the loss.
            loss = loss_fn(pred, y).item()

            # Accumulate the loss to calculate the mean validation loss.
            val_loss += loss

            # Accumulate the number of correct predictions to calculate the accuracy.
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate the mean loss.
    val_loss /= num_batches

    # Calculate the accuracy.
    val_accuracy = num_correct / size

    # Print validation statistics.
    print(f"\nValidation statistics:")
    print(f"Total number of images: {size}.")
    print(f"Total number of correct predictions: {int(num_correct)}.")
    print(f"Mean loss: {val_loss:>8f}.")
    print(f"Accuracy: {(100*val_accuracy):>0.2f}%.")

    # Return the mean validation loss and the validation accuracy.
    return val_loss, val_accuracy


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, epochs, patience, tolerance, path_to_save):
    """
    This function fits the model to the training data for a number of epochs.

    Args:
        train_dataloader: the training dataloader.
        val_dataloader: the validation dataloader.
        model: the model to be trained.
        optimizer: the optimizer with which the weights will get adjusted.
        loss_fn: the loss function to be used.
        epochs: the number of epochs.
        patience: the maximum number of epochs without improvement accepted. Training will stop when this number is exceeded.
        tolerance: the biggest change that doesn't count as improvement.
        path_to_save: the path to which the model's best weights are to be saved.

    Returns:

    """
    # Initialize a variable to watch the number of epochs without improvement for the patience limit.
    total_without_improvement = 0

    # Initialize an empty list to hold the training and validation values during training.
    loss_history = []
    acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Show the path to the saved model.
    print(f"The best weights will be saved in: {path_to_save}.")

    optimizer.zero_grad()

    # Run the training program for the specified number of epochs.
    for epoch in range(epochs):
        # Show the number of the epoch.
        print("\n\n==================================================")
        print(f"\nExecuting epoch number: {epoch + 1}")

        # Train the model and get the loss and the accuracy.
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)

        # Validate the training and get the loss and the accuracy.
        val_loss, val_acc = validation(dataloader=val_dataloader,
                                       model=model,
                                       loss_fn=loss_fn)

        # Append the losses and the accuracies to the lists initilized above.
        loss_history.append(train_loss)
        acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Get the first loss calculated in the first epoch.
        if epoch == 0:
            lowest_loss = val_loss
            print("Saving model in the first epoch...")
            torch.save(model.state_dict(), path_to_save)
        # For the other epochs, verifies whether of not the current validation loss is better than the lowest loss
        # observed until the epoch (also considering the tolerance).
        elif val_loss < lowest_loss - tolerance:
            # Save the new lowest loss.
            lowest_loss = val_loss

            # Save the model weights.
            print("Best validation loss found. Saving model...")
            torch.save(model.state_dict(), path_to_save)

            # Reset the patience counter.
            total_without_improvement = 0
        else:
            total_without_improvement += 1
            print("Epochs without improvement:", total_without_improvement)

        # Show the best validation loss observed until the current epoch.
        print(f"Best validation loss until now: {lowest_loss:>0.5f}")

        # Verify the patience. Break if patience is over.
        if total_without_improvement > patience:
            print(f"Patience ended with {epoch + 1} epochs.")
            break

    print("Training finished.")

    # Return a pandas dataframe with training and validation accuracies and losses.
    return pd.DataFrame(data={
        "loss": loss_history,
        "acc": acc_history,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history
    })


def train_siamese(dataloader_rec, dataloader_cls, model, loss_fn_rec, loss_fn_cls, optimizer_rec, optimizer_cls):
    size_rec = len(dataloader_rec.dataset)
    size_cls = len(dataloader_cls.dataset)
    num_batches_rec = len(dataloader_rec)
    num_batches_cls = len(dataloader_cls)

    model.train()
    train_loss_rec, train_accuracy_rec, train_loss_cls, train_accuracy_cls = 0, 0, 0, 0
    num_correct_rec, num_correct_cls = 0, 0

    for batch in dataloader_rec:
        anchor = batch[0].to(device)
        validation = batch[1].to(device)
        labels = batch[2].float().to(device)

        rec_output, _ = model(anchor, validation)
        loss_rec = loss_fn_rec(rec_output, labels)

        optimizer_rec.zero_grad()
        loss_rec.backward()
        optimizer_rec.step()

        train_loss_rec += loss_rec.item()
        num_correct_rec += (rec_output <= SIAMESE_MODEL_HYPERPARAMETERS["THRESHOLD"]).sum().item()

    for batch in dataloader_cls:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        _, cls_output = model(images, images)
        loss_cls = loss_fn_cls(cls_output, labels)

        optimizer_cls.zero_grad()
        loss_cls.backward()
        optimizer_cls.step()

        train_loss_cls += loss_cls.item()
        num_correct_cls += (cls_output.argmax(1) == labels).type(torch.float).sum().item()


    print('Training statistics:')

    print('Recognition:')
    print('Total number of images: ', size_rec)
    print('Total number of correct predictions: ', num_correct_rec)
    train_accuracy_rec = num_correct_rec / size_rec
    print('Accuracy: ', train_accuracy_rec)
    train_loss_rec /= num_batches_rec
    print('Mean loss: ', train_loss_rec)

    print('Classification:')
    print('Total number of images: ', size_cls)
    print('Total number of correct predictions: ', num_correct_cls)
    train_accuracy_cls = num_correct_cls / size_cls
    print('Accuracy: ', train_accuracy_cls)
    train_loss_cls /= num_batches_cls
    print('Mean loss: ', train_loss_cls)

    return train_loss_rec, train_loss_cls, train_accuracy_rec, train_accuracy_cls

def validation_siamese(val_dataloader, model, loss_fn):
    """
    This function is used to validate the siamese training. Like the train function, it is not called by itself, but by the fit function.

    Args:
        val_dataloader: the validation dataloader.
        model: the model whose training must be validated.
        loss_fn: the loss function to be used.

    Returns: the mean validation loss and the accuracy.

    """
    # Calculate the total number of images.
    size = len(val_dataloader.dataset)
    num_batches = len(val_dataloader)

    # Put the model in evaluation mode.
    model.eval()

    # Initialize the validation loss and the number of correct predictions with value 0.
    val_loss, num_correct = 0, 0

    # Proceed without gradient calculation (this reduces the charge on the memory).
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            _, cls_output = model(images, images)
            loss = loss_fn(cls_output, labels)

            val_loss += loss.item()
            num_correct += (cls_output.argmax(1) == labels).type(torch.float).sum().item()
            
    # Calculate the mean loss.
    val_loss /= num_batches

    # Calculate the accuracy.
    val_accuracy = num_correct / size

    # Print validation statistics.
    print(f"\nValidation statistics:")
    print(f"Total number of images: {size}.")
    print(f"Total number of correct predictions: {int(num_correct)}.")
    print(f"Mean loss: {val_loss:>8f}.")
    print(f"Accuracy: {(100*val_accuracy):>0.2f}%.")

    # Return the mean validation loss and the validation accuracy.
    return val_loss, val_accuracy


def fit_siamese(train_dataloader_rec, train_dataloader_cls, val_dataloader, model, optimizer_rec, optimizer_cls, loss_fn_rec, loss_fn_cls, epochs, patience, tolerance, path_to_save):
    """
    This function fits the model to the training data for a number of epochs.

    Args:
        train_dataloader_rec: the training dataloader for recognition.
        train_dataloader_cls: the training dataloader for classification.
        val_dataloader: the validation dataloader.
        one_shot_data: the samples for each class to be used in one-shot learning
        model: the model to be trained.
        optimizer_rec: the optimizer with which the weights will get adjusted for recognition.
        optimizer_cls: the optimizer withe which the weights will get ajusted for classification.
        loss_fn: the loss function to be used.
        epochs: the number of epochs.
        patience: the maximum number of epochs without improvement accepted. Training will stop when this number is exceeded.
        tolerance: the biggest change that doesn't count as improvement.
        path_to_save: the path to which the model's best weights are to be saved.

    Returns: he metrics of accuracy and loss for training and validation

    """
    # Initialize a variable to watch the number of epochs without improvement for the patience limit.
    total_without_improvement = 0

    # Initialize an empty list to hold the training and validation values during training.
    loss_history_rec = []
    acc_history_rec = []
    loss_history_cls = []
    acc_history_cls = []
    val_loss_history = []
    val_acc_history = []

    # Show the path to the saved model.
    print(f"The best weights will be saved in: {path_to_save}.")

    optimizer_rec.zero_grad()
    optimizer_cls.zero_grad()

    # Run the training program for the specified number of epochs.
    for epoch in range(epochs):
        # Show the number of the epoch.
        print("\n\n==================================================")
        print(f"\nExecuting epoch number: {epoch + 1}")

        # Train the model and get the loss and the accuracy.
        train_loss_rec, train_loss_cls, train_acc_rec, train_acc_cls = train_siamese(dataloader_rec=train_dataloader_rec, 
                                                                                     dataloader_cls=train_dataloader_cls, 
                                                                                     model=model, 
                                                                                     loss_fn_rec=loss_fn_rec, 
                                                                                     loss_fn_cls=loss_fn_cls, 
                                                                                     optimizer_rec=optimizer_rec,
                                                                                     optimizer_cls=optimizer_cls)

        # Validate the training and get the loss and the accuracy.
        val_loss, val_acc = validation_siamese(val_dataloader=val_dataloader,
                                               model=model,
                                               loss_fn=loss_fn_cls)

        # Append the losses and the accuracies to the lists initilized above.
        loss_history_rec.append(train_loss_rec)
        acc_history_rec.append(train_acc_rec)
        loss_history_cls.append(train_loss_cls)
        acc_history_cls.append(train_acc_cls)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Get the first loss calculated in the first epoch.
        if epoch == 0:
            lowest_loss = val_loss
            print("Saving model in the first epoch...")
            torch.save(model.state_dict(), path_to_save)
        # For the other epochs, verifies whether of not the current validation loss is better than the lowest loss
        # observed until the epoch (also considering the tolerance).
        elif val_loss < lowest_loss - tolerance:
            # Save the new lowest loss.
            lowest_loss = val_loss

            # Save the model weights.
            print("Best validation loss found. Saving model...")
            torch.save(model.state_dict(), path_to_save)

            # Reset the patience counter.
            total_without_improvement = 0
        else:
            total_without_improvement += 1
            print("Epochs without improvement:", total_without_improvement)

        # Show the best validation loss observed until the current epoch.
        print(f"Best validation loss until now: {lowest_loss:>0.5f}")

        # Verify the patience. Break if patience is over.
        if total_without_improvement > patience:
            print(f"Patience ended with {epoch + 1} epochs.")
            break

    print("Training finished.")

    # Return a pandas dataframe with training and validation accuracies and losses.
    return pd.DataFrame(data={
        "train_loss_rec": loss_history_rec,
        "train_acc_rec": acc_history_rec,
        "train_loss_cls": loss_history_cls,
        "train_acc_cls": acc_history_cls,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history
    })

def save_confused_image(path_to_save_matrix_csv, image, image_file_name, label, prediction):
    """
    This function saves the images that were wrongly classified.

    Args:
        path_to_save_matrix_csv: the path to the csv file containing the confusion matrix.
        image: the image to be saved.
        image_file_name: the image file name to be saved.
        label: the true label of the image.
        prediction: the predicted label of the image.

    Returns: nothing, but saves the image file with the name of the true label and the predicted label.

    """

    path_to_confused_images = os.path.join(path_to_save_matrix_csv, "confused_images")
    if not os.path.exists(path_to_confused_images): os.makedirs(path_to_confused_images)
    path_to_label = os.path.join(path_to_confused_images, label)
    if not os.path.exists(path_to_label): os.makedirs(path_to_label)
    path_to_prediction = os.path.join(path_to_label, prediction)
    if not os.path.exists(path_to_prediction): os.makedirs(path_to_prediction)
    path_to_save_image = os.path.join(path_to_prediction, image_file_name)
    image.save(path_to_save_image)




def test(dataloader, model, path_to_save_matrix_csv, path_to_save_matrix_png, labels_map):
    """
    This function tests a model.
    Args:
        dataloader: the test dataloader.
        model: the model to be tested.
        path_to_save_matrix_csv: the path to save the confusion matrix as a .csv file.
        path_to_save_matrix_png: the path to save the confusion matrix as a .png image.
        labels_map: a list with the labels. It will be used to create a list with the wrong classification.

    Returns: precision, recall and fscore calculated for the model in regard to the predictions on the test dataset.

    """
    # Put the model in evaluation mode.
    model.eval()

    # Get the total number of images.
    num_images = len(dataloader.dataset)

    # Initialize empty lists for predictions and labels.
    predictions, labels = [], []

    # Initialize the number of correct predictions with value 0.
    test_correct = 0

    # Proceed without calculating the gradients.
    with torch.no_grad():
        # Iterate over the data.
        for img, label, filename in dataloader:
            # Send images and labels to the correct device.
            img, label = img.to(device, dtype=torch.float), label.to(device)

            # Make predictions with the model.
            prediction = model(img)
            prediction_prob_values = softmax(prediction)

            # Get the index of the prediction with the highest probability.
            prediction = prediction.argmax(1)

            # Append predictions and labels to the lists initialized earlier.
            # Also, send both predictions and labels to the cpu.
            predictions.extend(prediction.cpu())
            labels.extend(label.cpu())

            # Accumulate the number of correct predictions.
            test_correct += (prediction == label).type(torch.float).sum().item()
            
            for i in range(len(img)):
                if (labels_map[prediction[i]] != labels_map[label[i]]):
                    print(f"File {filename[i]} is {labels_map[label[i]]}. Predicted as: {labels_map[prediction[i]]}.\nProbabilities: {prediction_prob_values[i]}\n")
                    save_confused_image(path_to_save_matrix_csv, img[i], filename[i], labels_map[label[i]], labels_map[prediction[i]])


    # Calculate the accuracy.
    acc = test_correct / num_images

    # Create the confusion matrix.
    matrix = metrics.confusion_matrix(labels, predictions)

    # Get the classes for the matrix.
    classes = DATA_HYPERPARAMETERS["CLASSES"]

    # Convert the matrix into a pandas dataframe.
    df_matrix = pd.DataFrame(matrix)

    # Save the matrix as a csv file.
    df_matrix.to_csv(path_to_save_matrix_csv)

    # Create a graphical matrix.
    plt.figure()
    sn.heatmap(df_matrix, annot=True, yticklabels=classes, xticklabels=classes)
    plt.title("Confusion matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    # Save the figure.
    plt.savefig(path_to_save_matrix_png, bbox_inches="tight")
    
    # Get some metrics.
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(labels, predictions, average="macro")

    # Write some results.
    print(f"Total number of predictions: {len(dataloader.dataset)}.")
    print(f"Number of correct predictions: {test_correct}.")
    print(f"Test accuracy: {(100 * acc):>0.2f}%.\n")
    print('\nPerformance metrics in the test set:')
    print(metrics.classification_report(labels, predictions))

    # Return the metrics.
    return precision, recall, fscore


def test_siamese(test_dataloader, model, path_to_save_matrix_csv, path_to_save_matrix_png, labels_map):
    """
    This function tests a siamese model.
    Args:
        test_dataloader: the test dataloader.
        model: the model to be tested.
        path_to_save_matrix_csv: the path to save the confusion matrix as a .csv file.
        path_to_save_matrix_png: the path to save the confusion matrix as a .png image.
        labels_map: a list with the labels. It will be used to create a list with the wrong classification.

    Returns: precision, recall and fscore calculated for the model in regard to the predictions on the test dataset.

    """
    print("Starting test process")

    predictions = []
    true_labels = []

    # Put the model in evaluation mode.
    model.eval()

    # Proceed without gradient calculation (this reduces the charge on the memory).
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            _, cls_output = model(images, images)

            # Store predictions and true labels
            predictions.extend(cls_output.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute precision, recall, and F1 score
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(true_labels, predictions, average="macro")

    # Compute confusion matrix
    matrix = metrics.confusion_matrix(true_labels, predictions)
    df_matrix = pd.DataFrame(matrix, columns=labels_map, index=labels_map)
    df_matrix.to_csv(path_to_save_matrix_csv)

    # Plot confusion matrix
    plt.figure()
    sn.heatmap(df_matrix, annot=True, yticklabels=True, xticklabels=True)
    plt.title("Confusion matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.savefig(path_to_save_matrix_png, bbox_inches="tight")

    # Print final results
    print("Final results:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", fscore)

    return precision, recall, fscore


def plot_history(history, path_to_save):
    """
    This function plots the history. It creates two subplots, one for the training and validation losses and another
    one for the training and validation accuracies.

    Args:
        history: a pandas dataframe containing the history in four columns: loss, val_loss, acc, and val_acc.
        path_to_save: the path to save the history plot.

    Returns: nothing, but it saves the plot as an image file whose format must be defined through the name given
    with the path to save it.

    """
    # Get the number of epochs.
    epochs = list(range(len(history["val_loss"])))

    # First subplot, with losses.
    fig, ax = plt.subplots(1, 2)
    # Plot the training loss.
    
    # Get scale-appropriate ylim
    all_losses = list(history["loss"]) + list(history["val_loss"])
    quantiles = np.quantile(all_losses, [0.25, 0.75])
    iqr = (quantiles[1] - quantiles[0])
    norm_sup = quantiles[1] + (1.5 * iqr)
    norm_inf = quantiles[0] - (1.5 * iqr)
    ax[0].set_ylim(norm_inf, norm_sup)
    
    # Plot
    ax[0].plot(epochs, history["loss"], label="Training loss")
    ax[0].plot(epochs, history["val_loss"], label="Validation loss")
    # Give the subplot a title.
    ax[0].set_title("Losses", fontsize=12)
    # Specify axes' names.
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    
    # Put a legend into the subplot.
    ax[0].legend()

    # Plot training accuracy.
    # Give the subplot a title.
    ax[1].set_title("Accuracies", fontsize=12)
    # Specify axes' names.
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    
    # Plot
    ax[1].plot(epochs, history["acc"], label="Training accuracy")
    ax[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    
    # Put a legend into the subplot.
    ax[1].legend()

    # Set the big title
    suptitle = plt.suptitle("Training vs. Validation", fontsize=14)
    
    # Adjust the layout
    fig.tight_layout()
    # Save the plot.
    plt.savefig(path_to_save, bbox_extra_artists=(suptitle, ), bbox_inches="tight", dpi=300)



def plot_history_siamese(history, path_to_save):
    """
    This function plots the history. It creates two subplots, one for the training and validation losses and another
    one for the training and validation accuracies.

    Args:
        history: a pandas dataframe containing the history in four columns: loss, val_loss, acc, and val_acc.
        path_to_save: the path to save the history plot.

    Returns: nothing, but it saves the plot as an image file whose format must be defined through the name given
    with the path to save it.

    """
    # Get the number of epochs.
    epochs = list(range(len(history["val_loss"])))

    # Create subplots with two columns.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot, with losses.
    # Get scale-appropriate ylim
    all_losses = list(history["train_loss_rec"]) + list(history["train_loss_cls"]) + list(history["val_loss"])
    quantiles = np.quantile(all_losses, [0.25, 0.75])
    iqr = (quantiles[1] - quantiles[0])
    norm_sup = quantiles[1] + (1.5 * iqr)
    norm_inf = quantiles[0] - (1.5 * iqr)
    ax[0].set_ylim(norm_inf, norm_sup)
    
    # Plot
    ax[0].plot(epochs, history["train_loss_rec"], label="Training Recognition loss")
    ax[0].plot(epochs, history["train_loss_cls"], label="Training Classification loss")
    ax[0].plot(epochs, history["val_loss"], label="Validation loss")
    # Give the subplot a title.
    ax[0].set_title("Losses", fontsize=12)
    # Specify axes' names.
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    # Put a legend into the subplot.
    ax[0].legend()

    # Second subplot, with accuracies.
    # Plot
    ax[1].plot(epochs, history["train_acc_rec"], label="Training Recognition accuracy")
    ax[1].plot(epochs, history["train_acc_cls"], label="Training Classification accuracy")
    ax[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    # Give the subplot a title.
    ax[1].set_title("Accuracies", fontsize=12)
    # Specify axes' names.
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    # Put a legend into the subplot.
    ax[1].legend()

    # Set the big title
    suptitle = plt.suptitle("Training vs. Validation", fontsize=14)

    # Adjust the layout
    fig.tight_layout()
    # Save the plot.
    plt.savefig(path_to_save, bbox_extra_artists=(suptitle,), bbox_inches="tight", dpi=300)
