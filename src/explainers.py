from captum import attr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


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

            # Plot the original image
            #plt.subplot(len(class_list)+1, 1, 1)
            fig, axs = plt.subplots(len(class_list)+1, 1)
            axs[0].imshow(original_img)
            axs[0].set_title("Original Image", fontdict={'fontsize': 10})
            axs[0].axis('off')


            # Get the attributions for each class and plot them
            for c in range(len(class_list)):
                attributions = explainer.attribute(img.unsqueeze(0), target=c)
                attributions = attributions.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze()

                # Get the attributions for different signs
                pos_attr = attributions * (attributions >= 0)
                neg_attr = attributions * (attributions < 0)

                # Resize and normalize (zero if max == min)
                pos_attr = cv2.resize(pos_attr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                if pos_attr.max() != pos_attr.min():
                    pos_attr = (pos_attr - pos_attr.min()) / (pos_attr.max() - pos_attr.min())
                else:
                    pos_attr = np.zeros_like(pos_attr)

                neg_attr = cv2.resize(neg_attr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                if neg_attr.max() != neg_attr.min():
                    neg_attr = (neg_attr - neg_attr.min()) / (neg_attr.max() - neg_attr.min())
                else:
                    neg_attr = np.zeros_like(neg_attr)


                # Create rgb image
                img_for_plot = np.zeros((img_size, img_size, 3))
                img_for_plot[:, :, 1] = pos_attr
                img_for_plot[:, :, 0] = np.abs(neg_attr)


                # Plot the image
                axs[c+1].imshow(img_for_plot)
                axs[c+1].imshow(original_img, alpha=0.4)
                axs[c+1].set_title(class_list[c], fontdict={'fontsize': 10})
                axs[c+1].axis('off')


            fig.tight_layout()
            fig.suptitle(f"GradCAM: label = {class_list[labels[i]]}, prediction = {class_list[pred_indices[i]]}\n", fontsize=10)
            plt.savefig(f"../results/gradcam/{model_name}/is_{class_list[labels[i]]}_pred_as_{class_list[pred_indices[i]]}_{filenames[i]}.png", bbox_inches='tight', dpi=300)
            plt.close()


def generate_occlusion():
    pass


def generate_shap():
    pass



