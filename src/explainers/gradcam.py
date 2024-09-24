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
                 full_results_path: str=None,
                 fold: int=1,
                 architecture: str=None,
                 optimizer: str=None,
                 learning_rate: str|float=None,
                 target_class: str|int="prediction",
                 plot_type: str="overlay",
                 classes: list[str]=None,
                 device: str=None,
                 attr_post_processing_function: Callable[[torch.tensor], torch.tensor]=None):
        # Inicializa os atributos que não precisam de processamento.
        self.model = model
        self.target_layer = target_layer
        self.dataloader = dataloader
        self.save_dir_path = save_dir_path
        self.full_results_path = full_results_path
        self.fold = fold
        self.architecture = architecture
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.target_class = target_class
        self.plot_type = plot_type
        self.classes = classes
        self.attr_post_processing_function = attr_post_processing_function

        # Inicializa atributos que precisam de processamento ou validação
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # Deve ter um jeito mais inteligente de fazer isto...
        if self.architecture is None or self.optimizer is None or self.learning_rate is None:
            if self.architecture or self.optimizer or self.learning_rate:
                raise ValueError("If you pass any of the model's hyperparameters, you must pass all of them.")

        self.model_name = None
        if self.architecture is not None:
            self.model_name = f"{self.fold}_{self.architecture}_{self.optimizer}_{self.learning_rate}"
        
        # Cria o diretório para salvar as imagens.
        os.makedirs(os.path.join(self.save_dir_path, f"gradcam_{self.model_name}"), exist_ok=True)
        
        # Ajusta o modelo e o explainer.
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
                                            plot_type=self.plot_type, 
                                            save_dir_path=os.path.join(self.save_dir_path, f"gradcam_{self.model_name}"))

            plotter.plot_all()
    

    def evaluate(self) -> None:
        results = {
            "fold": [],
            "learning_rate": [],
            "architecture": [],
            "optimizer": [],
            "all_outputs": [],
            "predictions": [],
            "prediction_logits": [],
            "prediction_probabilities": [],
            "labels": [],
            "filenames": [],
            "gradcam_predictions": [],
            "gradcam_all_outputs": [],
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
            results["all_outputs"].extend(predictions.tolist())
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
            results["gradcam_all_outputs"].extend(predictions.tolist())
            results["gradcam_predictions"].extend(predictions_idx)
            results["gradcam_prediction_logits"].extend([predictions[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])
            results["gradcam_prediction_probabilities"].extend([predictions_probabilities[i, predictions_idx[i]].item() for i in range(len(predictions_idx))])

        results["fold"] = [self.fold] * len(results["predictions"])
        results["learning_rate"] = [self.learning_rate] * len(results["predictions"])
        results["architecture"] = [self.architecture] * len(results["predictions"])
        results["optimizer"] = [self.optimizer] * len(results["predictions"])

        # Salva os resultados.
        save_results(results, os.path.join(self.save_dir_path, "gradcam_results.csv"))

        # Calcula novas métricas com base nos resultados após a aplicação do GradCAM.
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(results["labels"], results["predictions"], average="macro", zero_division=0.0)

        gradcam_precision, gradcam_recall, gradcam_fscore, _ = metrics.precision_recall_fscore_support(results["labels"], results["gradcam_predictions"], average="macro", zero_division=0.0)

        precision_diff = np.abs(precision - gradcam_precision)
        recall_diff = np.abs(recall - gradcam_recall)
        fscore_diff = np.abs(fscore - gradcam_fscore)

        eval_results = {
            "fold": self.fold,
            "learning_rate": self.learning_rate,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
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
        eval_results = {k: [v] for k, v in eval_results.items()}

        save_results(eval_results, save_dir_path=self.full_results_path)

        pr_results = EvaluationCurves(labels=np.array(results["labels"]), 
                                      predictions=np.array(results["all_outputs"]), 
                                      from_logits=True,
                                      save_path=os.path.join(self.save_dir_path, self.model_name)).precision_recall(return_results=True)

        pr_gc_results = EvaluationCurves(labels=np.array(results["labels"]),
                                         predictions=np.array(results["gradcam_all_outputs"]),
                                         from_logits=True,
                                         save_path=os.path.join(self.save_dir_path, self.model_name + "_gradcam")).precision_recall(return_results=True)
        
        pr_results = recursive_ndarray_tolist_conversion(pr_results)
        pr_gc_results = recursive_ndarray_tolist_conversion(pr_gc_results)

        save_json(pr_results, os.path.join(self.save_dir_path, self.model_name + "_pr_results.json"))
        save_json(pr_gc_results, os.path.join(self.save_dir_path, self.model_name + "_gradcam_pr_results.json"))


