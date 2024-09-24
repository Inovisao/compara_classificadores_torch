import cv2
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from scipy.special import softmax
from sklearn import metrics
from sklearn import preprocessing
import torch

        
def append_label_pred_to_filename(filename: str, 
                                  label: int|str, 
                                  prediction: int|str, 
                                  classes: list[str]=None) -> str:
    if classes is not None:
        label = classes[label]
        prediction = classes[prediction]

    label_pred_filename = str(f"is_{label}_pred_as_{prediction}_") + filename.split("/")[-1]

    return label_pred_filename
            

def min_max_normalization(x: torch.tensor) -> torch.tensor:
    x_min = x.min()
    x_max = x.max()

    x_normalized = (x - x_min) / (x_max - x_min)

    return x_normalized


def save_results(results: dict, save_dir_path: str) -> None:
    df = pd.DataFrame(results)
    if os.path.exists(save_dir_path):
        # Lê o arquivo CSV e adiciona os novos resultados.
        old_df = pd.read_csv(save_dir_path)
        df = pd.concat([old_df, df], axis=0)
    
    df.to_csv(save_dir_path, index=False)


class FeatureImportancePlot():
    def __init__(self, 
                 images: torch.tensor, 
                 attributions: torch.tensor,
                 filenames: list[str],
                 plot_type: str,
                 save_dir_path: str) -> None:

        # Arruma as imagens.
        images = images.squeeze()
        if images.dim() == 3:
            images = images.unsqueeze(0)

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
        attr = min_max_normalization(attr)
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


class EvaluationCurves():
    def __init__(self, 
                 labels: np.ndarray[int], 
                 predictions: np.ndarray[float], 
                 from_logits: bool=False,
                 save_path: str=None) -> None:

        self.labels = labels
        self.predictions = predictions
        self.from_logits = from_logits
        self.save_path = save_path

        if self.from_logits:
            self.predictions = softmax(self.predictions, axis=1)
        
        self.num_classes = self.predictions.shape[1]

    def precision_recall(self, return_results: bool=False) -> dict:
        per_class_results = {
            "precision": dict(),
            "recall": dict(),
            "average_precision": dict()
        }

        micro_results = {
            "precision": None,
            "recall": None,
            "average_precision": None
        }

        all_ap_score = metrics.average_precision_score(self.labels, self.predictions, average=None)

        for i in range(self.num_classes):
            per_class_results["precision"][i], per_class_results["recall"][i], _ = metrics.precision_recall_curve(self.labels, self.predictions[:, i], pos_label=i)

            per_class_results["average_precision"][i] = all_ap_score[i]

        # Binariza as labels e calcula a micro-média da curva de precisão-revocação.
        binary_labels = preprocessing.label_binarize(self.labels, classes=range(self.num_classes))
        micro_results["precision"], micro_results["recall"], _ = metrics.precision_recall_curve(binary_labels.ravel(), self.predictions.ravel()) 

        # Calcula a micro-média da precisão média.
        micro_results["average_precision"] = metrics.average_precision_score(self.labels, self.predictions, average="micro")

        if self.save_path is not None:
            # Plota a micro-média da curva de precisão-revocação.
            metrics.PrecisionRecallDisplay(micro_results["precision"], 
                                           micro_results["recall"], 
                                           average_precision=micro_results["average_precision"]).plot()
            plt.title("Micro-average Precision-Recall curve")
            plt.savefig(self.save_path + "_micro_precision_recall.png", bbox_inches="tight")
            plt.clf()

            _, ax = plt.subplots()

            # Plota as curvas de precisão-revocação por classe.
            for i in range(self.num_classes):
                display = metrics.PrecisionRecallDisplay(per_class_results["precision"][i], 
                                                         per_class_results["recall"][i], 
                                                         average_precision=per_class_results["average_precision"][i])
                display.plot(ax=ax, name=f"Class {i}")

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.title(f"Precision-Recall curve")
            plt.savefig(self.save_path + f"_macro_precision_recall.png", bbox_inches="tight")
            plt.clf()
        
        if return_results:
            return {"per_class": per_class_results, "micro": micro_results}


def save_json(data: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(data, f)


def recursive_ndarray_tolist_conversion(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        elif isinstance(value, dict):
            data[key] = recursive_ndarray_tolist_conversion(value)
    
    return data