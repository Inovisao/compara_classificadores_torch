import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import torch
from torch import nn

class FeatureImportancePlot():
    def __init__(self, 
                 images: torch.tensor, 
                 attributions: torch.tensor,
                 filenames: list[str],
                 plot_type: str,
                 save_dir_path: str) -> None:

        # Arruma as imagens.
        images = images.squeeze()
        if images.dim() == 4:
            images = images.permute(0, 2, 3, 1)
        
        images = images.cpu().detach().numpy()

        # Arruma as atribuições.
        attributions = attributions.squeeze()
        if attributions.dim() == 4:
            attributions = attributions.permute(0, 2, 3, 1)
        
        attributions = attributions.cpu().detach().numpy()
        
        # Define os atributos.
        self.images = images
        self.attributions = attributions
        self.filenames = filenames
        self.plot_type = plot_type
        self.save_dir_path = save_dir_path

        plot_types = {
            "overlay": self.plot_overlay,
            "hadamard": self.plot_hadamard
        }

        self.plot = plot_types[plot_type]

    
    def plot_one(self, idx: int) -> None:
        # Seleciona a imagem e a atribuição. Normaliza e redimensiona a atribuição.
        img = self.images[idx]
        attr = self.attributions[idx]
        attr = (attr - attr.min()) / (attr.max() - attr.min())
        attr = cv2.resize(attr, (img.shape[1], img.shape[0]))

        save_path = os.path.join(self.save_dir_path, self.filenames[idx] + ".png")

        self.plot(img, attr, save_path)


    def plot_overlay(self, 
                     img: np.ndarray, 
                     attr: np.ndarray, 
                     save_path: str) -> None:
        plt.imshow(img)
        plt.imshow(attr, cmap="jet", alpha=0.3)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


    def plot_hadamard(self, 
                      img: np.ndarray, 
                      attr: np.ndarray, 
                      save_path: str) -> None:
        # Calcula o produto de Hadamard.
        attr = attr[:, :, np.newaxis] * img

        # Plota
        plt.imshow(attr)
        plt.axis("off")
        plt.show()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    

    def plot_all(self):
        pool = Pool()
        pool.map(self.plot_one, range(self.images.shape[0]))

        
def append_label_pred_to_filename(filename: str, 
                                  label: int|str, 
                                  prediction: int|str, 
                                  classes: list[str]=None) -> str:
    if classes is not None:
        label = classes[label]
        prediction = classes[prediction]

    label_pred_filename = str(f"is_{label}_pred_as_{prediction}_") + filename.split("/")[-1]

    return label_pred_filename
            
        
