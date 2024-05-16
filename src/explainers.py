from captum import attr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

interpolation = cv2.INTER_LINEAR

def plot_attributions(original_img, positive, label, prediction, class_list, save_path, negative=None, plot_original=True, attr_alpha=0.5):
    num_cols = len(class_list)
    num_rows = 2 if negative is not None else 1

    #fig, axs = plt.subplots(num_rows, num_cols)
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))
    axs = fig.subplots(num_rows, num_cols)

    for c in range(len(class_list)):
        # Plot positive attributions
        if plot_original: axs[0, c].imshow(original_img)
        axs[0, c].imshow(positive[c], alpha=attr_alpha, cmap='jet')
        axs[0, c].set_title(class_list[c], fontdict={'fontsize': 8})

        if (c == 0) and (negative is not None):
            axs[0, c].set_ylabel("Positive", fontsize=10)
            axs[0, c].set_xlabel("")
            axs[0, c].set_xticks([])
            axs[0, c].set_yticks([])
            for p in ['top', 'bottom', 'left', 'right']:
                axs[0, c].spines[p].set_visible(False)
        else:
            axs[0, c].axis('off')


        if negative is not None:
            # Plot negative attributions
            if plot_original: axs[1, c].imshow(original_img)
            axs[1, c].imshow(np.abs(negative[c]), alpha=attr_alpha, cmap='jet')

            if c == 0:
                axs[1, c].set_ylabel("Negative", fontsize=10)
                axs[1, c].set_xlabel("")
                axs[1, c].set_xticks([])
                axs[1, c].set_yticks([])
                for p in ['top', 'bottom', 'left', 'right']:
                    axs[1, c].spines[p].set_visible(False)
            else:
                axs[1, c].axis('off')

    
    fig.suptitle(f"Label = {class_list[label]}, Prediction = {class_list[prediction]}\n", fontsize=10)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def generate_gradcam(model, model_name, layer, test_dataloader, class_list, device):
    explainer = attr.LayerGradCam(model, layer)

    # Get image size
    img_size = test_dataloader.dataset[0][0].shape[-1]

    # Iterate over the batches
    for imgs, labels, filenames in test_dataloader:
        # Send to the device and get predictions
        imgs, labels = imgs.to(device), labels.to(device)
        predictions = model(imgs)
        pred_indices = torch.argmax(predictions, dim=1)

        
        # Plot all the attributions in a grid for each image
        for i, img in enumerate(imgs):
            original_img = img.cpu().numpy().transpose(1, 2, 0)
            true_idx = labels[i]
            pred_idx = pred_indices[i]

            # Get the attributions for each class in a list
            attributions = list()
            for c in range(len(class_list)):
                attributions.append(explainer.attribute(img.unsqueeze(0), target=c))
            
            # Get the attributions for different signs
            pos_attr = [attr * (attr >= 0) for attr in attributions]
            neg_attr = [attr * (attr < 0) for attr in attributions]

            # Resize and normalize ? (zero if max == min)
            pos_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=interpolation) for attr in pos_attr]
            #pos_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in pos_attr]

            neg_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=interpolation) for attr in neg_attr]
            #neg_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in neg_attr]

            # Plot the attributions
            save_path = f"../results/gradcam/{model_name}/is_{class_list[true_idx]}_pred_as_{class_list[pred_idx]}_{filenames[i]}.png"
            plot_attributions(original_img=original_img, positive=pos_attr, negative=neg_attr, label=labels[i], prediction=pred_indices[i], class_list=class_list, save_path=save_path)
            

def generate_occlusion(model, 
                       model_name, 
                       test_dataloader, 
                       class_list, 
                       device):
    explainer = attr.Occlusion(model)

    image_size = test_dataloader.dataset[0][0].shape[-1]

    # Iterate over the batches
    for imgs, labels, filenames in test_dataloader:
        # Send to the device and get predictions
        imgs, labels = imgs.to(device), labels.to(device)
        predictions = model(imgs)
        pred_indices = torch.argmax(predictions, dim=1)

        attributions_all_classes = list()
        for c in range(len(class_list)):
            attributions_all_classes.append(explainer.attribute(imgs, target=c, sliding_window_shapes=(3, 16, 16), strides=16, baselines=0))


        # Plot all the attributions in a grid for each image
        for i, img in enumerate(imgs):
            original_img = img.cpu().numpy().transpose(1, 2, 0)
            true_idx = labels[i]
            pred_idx = pred_indices[i]

            # Get the attributions for each class in a list
            attributions = list()
            for c in range(len(class_list)):
                attributions.append(attributions_all_classes[c][i])

            # Get the attributions for different signs
            pos_attr = [attr * (attr >= 0) for attr in attributions]
            neg_attr = [attr * (attr < 0) for attr in attributions]

            pos_attr = [p.mean(dim=0) for p in pos_attr]
            neg_attr = [n.mean(dim=0) for n in neg_attr]
            
            # Resize and normalize ? (zero if max == min)
            pos_attr = [attr.cpu().detach().numpy() for attr in pos_attr]
            #pos_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in pos_attr]
            
            neg_attr = [attr.cpu().detach().numpy() for attr in neg_attr]
            #neg_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in neg_attr]

            # Plot the attributions
            save_path = f"../results/occlusion/{model_name}/is_{class_list[true_idx]}_pred_as_{class_list[pred_idx]}_{filenames[i]}.png"
            plot_attributions(original_img=original_img, positive=pos_attr, negative=neg_attr, label=labels[i], prediction=pred_indices[i], class_list=class_list, save_path=save_path)


def generate_guided_backprop(model, model_name, test_dataloader, class_list, device):
    explainer = attr.GuidedBackprop(model)

    img_size = test_dataloader.dataset[0][0].shape[-1]

    # Iterate over the batches
    for imgs, labels, filenames in test_dataloader:
        # Send to the device and get predictions
        imgs, labels = imgs.to(device), labels.to(device)
        predictions = model(imgs)
        pred_indices = torch.argmax(predictions, dim=1)
    
        # Plot all the attributions in a grid for each image
        for i, img in enumerate(imgs):
            original_img = img.cpu().numpy().transpose(1, 2, 0)
            true_idx = labels[i]
            pred_idx = pred_indices[i]

            # Get the attributions for each class in a list
            attributions = list()
            for c in range(len(class_list)):
                img.requires_grad = True
                attributions.append(explainer.attribute(img.unsqueeze(0), target=c))

            # Get the attributions for different signs
            pos_attr = [attr * (attr >= 0) for attr in attributions]
            neg_attr = [attr * (attr < 0) for attr in attributions]

            # Resize and normalize ? (zero if max == min)

            pos_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=cv2.INTER_NEAREST) for attr in pos_attr]
            pos_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in pos_attr]

            neg_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=cv2.INTER_NEAREST) for attr in neg_attr]
            neg_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in neg_attr]

            # Plot the attributions
            save_path = f"../results/guided_backprop/{model_name}/is_{class_list[true_idx]}_pred_as_{class_list[pred_idx]}_{filenames[i]}.png"
            plot_attributions(original_img=original_img, positive=pos_attr, negative=neg_attr, label=labels[i], prediction=pred_indices[i], class_list=class_list, save_path=save_path, plot_original=False, attr_alpha=1.0)


def generate_guided_gradcam(model, model_name, layer, test_dataloader, class_list, device):
    pass


def generate_shap(model, model_name, train_dataloader, test_dataloader, class_list, device):
    # Get background dataset
    background_dataset = train_dataloader.dataset
    if len(background_dataset) > train_dataloader.batch_size:
        background_dataset = torch.utils.data.Subset(background_dataset, torch.randperm(len(background_dataset))[:train_dataloader.batch_size])
    else: # Get half (?) the dataset if it is too small
        half = len(background_dataset) // 2
        background_dataset = torch.utils.data.Subset(background_dataset, torch.randperm(len(background_dataset)[:half]))
    background_dataset = torch.stack([i[0].to(device) for i in background_dataset])

    # Initialize the explainer
    model.eval()
    explainer = shap.DeepExplainer(model, background_dataset)

    # Iterate over the batches
    for imgs, labels, filenames in test_dataloader:
        # Send to the device and get predictions
        imgs, labels = imgs.to(device), labels.to(device)

        # Get predictions
        predictions = model(imgs)
        pred_indices = torch.argmax(predictions, dim=1)

        # Get shap values
        shap_values = explainer.shap_values(imgs)
        shap_values = np.array(shap_values) # (outs, batch, channels, 256, 256)
        shap_values = shap_values.transpose(0, 1, 3, 4, 2)

        # Create individual plots for each image
        # Iterate over the images
        for i, img in enumerate(imgs):
            original_img = img.unsqueeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            true_idx = labels[i]
            pred_idx = pred_indices[i]

            # Shap for this image
            one_shap = shap_values[:, i, :, :, :]
            one_shap = [np.expand_dims(s, axis=0) for s in one_shap[:]]
            shap.plots.image(one_shap, original_img, labels=[class_list], true_labels=[class_list[true_idx]], show=False)
            plt.suptitle(f"Label = {class_list[true_idx]}, Prediction = {class_list[pred_idx]}\n", fontsize=10)
            plt.savefig(f"../results/shap/{model_name}/is_{class_list[true_idx]}_pred_as_{class_list[pred_idx]}_{filenames[i]}.png", bbox_inches='tight', dpi=300)
            plt.close()

