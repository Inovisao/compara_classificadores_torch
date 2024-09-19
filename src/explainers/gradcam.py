from captum.attr import LayerGradCam
from collections.abc import Callable
from sklearn import metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from ._utils import *

class GradCAM():
    def __init__(self, 
                 model: nn.Module, 
                 target_layer: nn.Module,
                 dataloader: DataLoader,
                 save_dir_path: str,
                 args: dict,
                 target_class: str|int="prediction",
                 classes: list[str]=None,
                 device: str=None,
                 attr_post_processing_function: Callable[[torch.tensor], torch.tensor]=None):

        self.model = model
        self.target_layer = target_layer
        self.dataloader = dataloader
        self.save_dir_path = save_dir_path
        self.args = args
        self.target_class = target_class
        self.classes = classes
        self.attr_post_processing_function = attr_post_processing_function

        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(self.device)

        self.explainer = LayerGradCam(self.model, self.target_layer)


    def explain(self) -> None:
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
            attributions = self.explainer.attribute(images, 
                                                    target=targets, 
                                                    attr_dim_summation=True if self.attr_post_processing_function is None else False)
     
            if self.attr_post_processing_function is not None:
                attributions = self.attr_post_processing_function(attributions)

            filenames = [append_label_pred_to_filename(filename, label, prediction, self.classes) for filename, label, prediction in zip(filenames, labels.tolist(), predictions_idx.tolist())]

            # Salva as imagens.
            plotter = FeatureImportancePlot(images=images, 
                                            attributions=attributions, 
                                            filenames=filenames, 
                                            plot_type="overlay", 
                                            save_dir_path=self.save_dir_path)

            plotter.plot_all()
    

    def evaluate(self) -> None:
        results = {
            "fold": [],
            "learning_rate": [],
            "architecture": [],
            "optimizer": [],
            "predictions": [],
            "prediction_logits": [],
            "prediction_probabilities": [],
            "labels": [],
            "filenames": [],
            "gradcam_predictions": [],
            "gradcam_prediction_logits": [],
            "gradcam_prediction_probabilities": [],
        }

        for images, labels, filenames in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Guarda informações conhecidas a priori.
            results["filenames"].extend(filenames)
            results["labels"].extend(labels.tolist())

            with torch.no_grad():
                # Pega previsões do modelo sobre as imagens originais.
                predictions = self.model(images)
                predictions_probabilities = torch.nn.functional.softmax(predictions, dim=1)
                predictions_idx = torch.argmax(predictions, dim=1).tolist()

            # Guarda informações das previsões referentes às imagens originais.
            results["predictions"].extend(predictions_idx)
            results["prediction_logits"].extend([predictions[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])
            results["prediction_probabilities"].extend([predictions_probabilities[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])


            # Calcula as métricas. Aqui fazemos necessariamente com as classes previstas, pois queremos avaliar a qualidade do mapa de importância em relação à previsão.
            attributions = self.explainer.attribute(images, 
                                                    target=predictions_idx,
                                                    attr_dim_summation=True if self.attr_post_processing_function is None else False)

            if self.attr_post_processing_function is not None:
                attributions = self.attr_post_processing_function(attributions)
            
            # Normaliza as atribuições (min-max), aplica resize e multiplica pelas imagens originais.
            batch_min_max = torch.vmap(min_max_normalization)
            attributions = batch_min_max(attributions)
            attributions = resize(img=attributions, size=(images.shape[2], images.shape[3]))

            images = images * attributions

            predictions = self.model(images)
            predictions_probabilities = torch.nn.functional.softmax(predictions, dim=1)
            predictions_idx = torch.argmax(predictions, dim=1).tolist()

            # Guarda informações das previsões referentes às imagens com GradCAM.
            results["gradcam_predictions"].extend(predictions_idx)
            results["gradcam_prediction_logits"].extend([predictions[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])
            results["gradcam_prediction_probabilities"].extend([predictions_probabilities[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])

        results["fold"] = [self.args["run"]] * len(results["predictions"])
        results["learning_rate"] = [self.args["learning_rate"]] * len(results["predictions"])
        results["architecture"] = [self.args["architecture"]] * len(results["predictions"])
        results["optimizer"] = [self.args["optimizer"]] * len(results["predictions"])

        # Salva os resultados.
        save_results(results, "../results/gradcam_results.csv")

        # Calcula novas métricas com base nos resultados após a aplicação do GradCAM.
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(results["labels"], results["predictions"], average="macro", zero_division=0.0)

        gradcam_precision, gradcam_recall, gradcam_fscore, _ = metrics.precision_recall_fscore_support(results["labels"], results["gradcam_predictions"], average="macro", zero_division=0.0)

        precision_diff = np.abs(precision - gradcam_precision)
        recall_diff = np.abs(recall - gradcam_recall)
        fscore_diff = np.abs(fscore - gradcam_fscore)

        results = {
            "fold": self.args["run"],
            "learning_rate": self.args["learning_rate"],
            "architecture": self.args["architecture"],
            "optimizer": self.args["optimizer"],
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "gradcam_precision": gradcam_precision,
            "gradcam_recall": gradcam_recall,
            "gradcam_fscore": gradcam_fscore,
            "precision_diff": precision_diff,
            "recall_diff": recall_diff,
            "fscore_diff": fscore_diff
        }
        results = {k: [v] for k, v in results.items()}

        save_results(results, save_dir_path="../results_dl/gradcam_metrics.csv")











