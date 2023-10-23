#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from hyperparameters import MODEL_HYPERPARAMETERS, DATA_HYPERPARAMETERS, SIAMESE_MODEL_HYPERPARAMETERS
import matplotlib.pyplot as plt
import numpy as np
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
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


def train_siamese(dataloader, model, loss_fn, optimizer):
    """
    This function is used to train a siamese model. It is not called by itself, but inside the 'fit_siamese' function below.

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
    for batch in dataloader:
        anchor = batch[0].to(device)
        validation = batch[1].to(device)
        labels = batch[2].float().to(device)
        outputs = model(anchor, validation)
        
        loss = loss_fn(outputs.squeeze(), labels)
        train_loss += loss_fn(outputs.squeeze(), labels).item()
        num_correct += ((outputs.squeeze() >= SIAMESE_MODEL_HYPERPARAMETERS["ACC_THRESHOLD"]) == labels).type(torch.float).sum().item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the mean loss during training.
    train_loss /= num_batches

    # Calculate the accuracy during training.
    train_accuracy = num_correct / size

    # Return training loss and accuracy.
    return train_loss, train_accuracy


def validation_siamese(dataloader, model, loss_fn):
    """
    This function is used to validate the siamese training. Like the train function, it is not called by itself, but by the fit function.

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
        for val_batch in dataloader:
            anchor = val_batch[0].to(device)
            validation = val_batch[1].to(device)
            labels = val_batch[2].float().to(device)
            outputs = model(anchor, validation)
            
            loss = loss_fn(outputs.squeeze(), labels).item()
            val_loss += loss
            num_correct += ((outputs.squeeze() >= SIAMESE_MODEL_HYPERPARAMETERS["ACC_THRESHOLD"]) == labels).type(torch.float).sum().item()

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


def fit_siamese(train_dataloader, val_dataloader, model, optimizer, loss_fn, epochs, patience, tolerance, path_to_save):
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
    print('entrou')
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
        train_loss, train_acc = train_siamese(train_dataloader, model, loss_fn, optimizer)

        # Validate the training and get the loss and the accuracy.
        val_loss, val_acc = validation_siamese(dataloader=val_dataloader,
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


def test_siamese(test_data, one_shot_data, model, path_to_save_matrix_csv, path_to_save_matrix_png, labels_map):
    
    precision_metric = Precision(task="multiclass", num_classes=len(labels_map))
    recall_metric = Recall(task="multiclass", num_classes=len(labels_map))
    fscore_metric = F1Score(task="multiclass", num_classes=len(labels_map))

    predictions, labels = [], []

    with torch.no_grad():
        for item in test_data:
            image = item[0].unsqueeze(0).to(device)
            label = next(class_id for class_id, class_name in enumerate(labels_map) if class_name == item[1])
            highest_score = float('-inf')
            image_class = -1
            print(len(one_shot_data))
            for anchor in one_shot_data:
                anchor_image = anchor[0].unsqueeze(0).to(device)
                pred_score = model(image, anchor_image)
                if pred_score > highest_score:
                    highest_score = pred_score
                    image_class = next(class_id for class_id, class_name in enumerate(labels_map) if class_name == anchor[1])
            
            print(f'Expected class: {labels_map[label]} Predicted class: {labels_map[image_class] if image_class >= 0 else "No Class Identified"} Score: {highest_score.item()}')
            
            predictions.append(image_class)
            labels.append(label)

            precision_metric.update(torch.tensor([image_class]), torch.tensor([label]))
            recall_metric.update(torch.tensor([image_class]), torch.tensor([label]))
            fscore_metric.update(torch.tensor([image_class]), torch.tensor([label]))

        avg_precision = precision_metric.compute()
        avg_precision = avg_precision.item()
        avg_recall = recall_metric.compute()
        avg_recall = avg_recall.item()
        avg_fscore = fscore_metric.compute()
        avg_fscore = avg_fscore.item()

        matrix = metrics.confusion_matrix(labels, predictions)
        df_matrix = pd.DataFrame(matrix, columns=labels_map, index=labels_map)
        df_matrix.to_csv(path_to_save_matrix_csv)
        plt.figure()
        sn.heatmap(df_matrix, annot=True, yticklabels=True, xticklabels=True)
        plt.title("Confusion matrix", fontsize=14)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.savefig(path_to_save_matrix_png, bbox_inches="tight")

    print("Final results:")
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1 Score:", avg_fscore)

    return avg_precision, avg_recall, avg_fscore


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


def create_gradcam(dataloader, model, target_layer, subset, labels_map, path_to_save):
    """
    This function creates the GradCAM images for images in the dataloader given as argument.
    Args:
        dataloader: the dataloader with the images for which the GradCAM files are to be created.
        model: the model to be explained.
        target_layer: the layer with which the GradCAM visualizations are to be created. According to the
        GradCAM documenation, this is usually the last convolutional layer of the model. As each model has a different
        architecture up to last layer (at least in PyTorch), I did not think of a way of making this non-parametric,
        nor do I think I should. Therefore, this must be passed as an argument.
        subset: the name of the subset (test). This is used to name the files without confusion.
        labels_map: a list with the labels. The indexes are used to convert predictions into class names.
        path_to_save: the directory to save the GradCAM files.

    Returns: nothing. It saves the files in the specified directory as .png image files.

    """
    # Iterate through the batches.
    for X, y, original_file in dataloader:

        # Send the images, the labels and the model to the GPU.
        X, y = X.to(device, torch.float), y.to(device)
        model = model.to(device)

        # Get predictions for the model.
        preds = model(X)

        # Iterate through the images to generate the GradCAM files.
        for i in range(len(X)):
            # Get one image and adjust the pixel values.
            one_image = X[i, :, :, :].cpu().numpy() * 255.

            # Get one image as an Image object.
            one_image = Image(one_image, batched=False, channel_last=False)

            # Compose a transform to turn the image into a tensor.
            transform = get_transforms(image_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"],
                                       data_augmentation=False,
                                       for_gradcam=True)

            # Instantiate a GradCAM object.
            explainer = GradCAM(model=model,
                                target_layer=target_layer,
                                preprocess_function=lambda ims: torch.stack([transform(im.to_pil()) for im in ims]))

            # Explain one image.
            explanations = explainer.explain(one_image, y[i].item())

            # Plot gradcam.
            plot = explanations.plot(class_names=labels_map)

            # Set supertitle.
            plot[0].suptitle(f"Label: {labels_map[y[i].item()]}. Predicted as: {labels_map[torch.argmax(preds[i])]}.",
                             y=0.85,
                             fontsize=14)

            # Create the filename.
            filename = str(subset) + "_" + str(original_file[i]) + "_is_" + str(labels_map[y[i]]) + "_predicted_as_" + \
                str(labels_map[torch.argmax(preds[i])]) + ".png"

            # Save the plot.
            plt.savefig(os.path.join(path_to_save, filename))

            # Close the image.
            plt.close()

        torch.cuda.empty_cache()
