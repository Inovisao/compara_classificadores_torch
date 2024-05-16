import argparse
import captum
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import models
from torchvision.transforms.functional import normalize


LAYERS = ["layer1", "layer2", "layer3", "layer4"]
model = models.resnet18() # É pra funcionar com qualquer variante da ResNet
interpolation = cv2.INTER_LINEAR

def explain_layer(model, target_layer, input_image, classes):
    explainer = captum.attr.LayerGradCam(model, target_layer)
    image_size = input_image.shape[-1]

    positive_attributions = list()
    negative_attributions = list()
    for c in range(len(classes)):
        attributions = explainer.attribute(input_image, target=c)
        positive = attributions * (attributions >= 0)
        negative = attributions * (attributions < 0)

        positive = cv2.resize(positive.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (image_size, image_size), interpolation=interpolation)
        negative = cv2.resize(negative.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(), (image_size, image_size), interpolation=interpolation)

        positive_attributions.append(positive)
        negative_attributions.append(negative)
    
    return positive_attributions, negative_attributions


def plot_all_layers(original_image, attributions_per_layer, label, prediction, classes, save_path):
    num_cols = len(classes)
    num_rows = 2 * len(LAYERS)

    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))

    # Add subplots
    axs = fig.subplots(num_rows, num_cols)

    for l, layer in enumerate(LAYERS):
        for c in range(len(classes)):
            positive = attributions_per_layer[layer]["positive"][c]
            negative = attributions_per_layer[layer]["negative"][c]

            # Plot positive attributions
            axs[2*l, c].imshow(original_image)
            # Min-max norm
            if positive.max() != positive.min():
                positive = (positive - positive.min()) / (positive.max() - positive.min())
            axs[2*l, c].imshow(positive, alpha=0.4, cmap='jet')
            axs[2*l, c].set_title(classes[c], fontdict={'fontsize': 8})
            if (c == 0) and (negative is not None):
                axs[2*l, c].set_ylabel(f"{layer}\nPositive", fontsize=8)
                axs[2*l, c].set_xlabel("")
                axs[2*l, c].set_xticks([])
                axs[2*l, c].set_yticks([])
                for p in ['top', 'bottom', 'left', 'right']:
                    axs[2*l, c].spines[p].set_visible(False)
            else:
                axs[2*l, c].axis('off')


            if negative is not None:
                # Plot negative attributions
                axs[2*l+1, c].imshow(original_image)
                # Min-max norm
                negative = np.abs(negative)
                if negative.max() != negative.min():
                    negative = (negative - negative.min()) / (negative.max() - negative.min())
                axs[2*l+1, c].imshow(negative, alpha=0.4, cmap='jet')

                if c == 0:
                    axs[2*l+1, c].set_ylabel(f"{layer}\nNegative", fontsize=8)
                    axs[2*l+1, c].set_xlabel("")
                    axs[2*l+1, c].set_xticks([])
                    axs[2*l+1, c].set_yticks([])
                    for p in ['top', 'bottom', 'left', 'right']:
                        axs[2*l+1, c].spines[p].set_visible(False)
                else:
                    axs[2*l+1, c].axis('off')
    
    fig.suptitle(f"Label = {label}, Prediction = {prediction}\n", fontsize=10)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()   
            


def get_args():
    # Instantiate argparse
    parser = argparse.ArgumentParser(description="Arguments for explanation.")
    parser.add_argument("-f", "--fold", type=int, help="Fold number.")
    parser.add_argument("-w", "--weights", type=str, help="Path to the weights.")
    parser.add_argument("-td", "--test_dir", type=str, help="Path to the test directory.")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_dir = args.test_dir
    classes = sorted(os.listdir(test_dir))
    num_classes = len(classes)

#    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(args.weights), strict=True)
    model = model.to("cuda")

    # Get all images in the test directory.
    all_images = list()
    for c in classes:
        some_images = os.listdir(os.path.join(test_dir, c))
        all_images.extend([os.path.join(c, i) for i in some_images])
    
    for i in all_images:
        image = cv2.imread(os.path.join(test_dir, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        original_image = image.copy()
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.
        image = image.to("cuda")

        label, image_name = i.split("/")

        # Get the prediction.
        model.eval()
        prediction = model(image)
        prediction_idx = torch.argmax(prediction).item()
        prediction_class = classes[prediction_idx]

        # Get the attributions.
        attributions_per_layer = dict()
        for l in LAYERS:
            layer = getattr(model, l)
            attributions_per_layer[l] = {"positive": None, "negative": None}
            attributions_per_layer[l]["positive"], attributions_per_layer[l]["negative"] = explain_layer(model, layer, image, classes)
        

        situation = "correct" if label == prediction_class else "incorrect"        
        save_path = f"./results_gradcam_multilayer_resnet18/fold_{args.fold}/{situation}_is_{label}_predicted_as_{prediction_class}_{image_name}.png"
        

        plot_all_layers(original_image, attributions_per_layer, label, prediction_class, classes, save_path)

        print("Done image: ", i)



    













