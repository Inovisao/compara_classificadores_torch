from captum.attr import LayerGradCam
from collections.abc import Callable
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from ._utils import *

class GradCAM():
    def __init__(self, 
                 model: nn.Module, 
                 target_layer: nn.Module,
                 dataloader: DataLoader,
                 save_dir_path: str,
                 target_class: str|int="prediction",
                 classes: list[str]=None,
                 device: str=None,
                 attr_post_processing_function: Callable[[torch.tensor], torch.tensor]=None):

        self.model = model
        self.target_layer = target_layer
        self.dataloader = dataloader
        self.save_dir_path = save_dir_path
        self.target_class = target_class
        self.classes = classes
        self.attr_post_processing_function = attr_post_processing_function

        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(self.device)

    def explain(self) -> None:
        explainer = LayerGradCam(self.model, self.target_layer)
        for images, labels, filenames in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Pega previsões do modelo.
            predictions = self.model(images)
            predictions_idx = torch.argmax(predictions, dim=1)

            # Define a classe a ser explicada.
            targets = self.target_class

            if self.target_class == "prediction":
                targets = predictions_idx.tolist()

            if self.target_class == "ground_truth":
                targets = labels.tolist()
            

            # Faz as atribuições e chama a função de pós-processamento, se ela existir.
            attributions = explainer.attribute(images, 
                                               target=targets, 
                                               attr_dim_summation=True if self.attr_post_processing_function is None else False)

            if self.attr_post_processing_function is not None:
                attributions = self.attr_post_processing_function(attributions)

            filenames = [append_label_pred_to_filename(filename, label, prediction, self.classes) for filename, label, prediction in zip(filenames, labels.tolist(), predictions_idx.tolist())]

            # Salva as imagens.
            plotter = FeatureImportancePlot(images=images, 
                                            attributions=attributions, 
                                            filenames=filenames, 
                                            plot_type="hadamard", 
                                            save_dir_path=self.save_dir_path)

            plotter.plot_all()


            






