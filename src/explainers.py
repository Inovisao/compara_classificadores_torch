from captum import attr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attributions(original_img, positive, label, prediction, class_list, save_path, negative=None):
    num_cols = len(class_list)
    num_rows = 2 if negative is not None else 1

    fig, axs = plt.subplots(num_rows, num_cols)

    for c in range(len(class_list)):
        # Plot positive attributions
        axs[0, c].imshow(original_img)
        axs[0, c].imshow(positive[c], alpha=0.4, cmap='jet')
        axs[0, c].set_title(class_list[c], fontdict={'fontsize': 8})
        #axs[0, c].axis('off')
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
            axs[1, c].imshow(original_img)
            axs[1, c].imshow(np.abs(negative[c]), alpha=0.4, cmap='jet')

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
            pos_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=cv2.INTER_NEAREST) for attr in pos_attr]
            #pos_attr = [(attr - attr.min()) / (attr.max() - attr.min()) if attr.max() != attr.min() else np.zeros_like(attr) for attr in pos_attr]

            neg_attr = [cv2.resize(attr.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (img_size, img_size), interpolation=cv2.INTER_NEAREST) for attr in neg_attr]
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


def generate_guided_backprop():
    pass


def generate_guided_gradcam():
    pass


def generate_shap():
    pass



