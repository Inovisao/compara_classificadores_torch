from argparse import ArgumentParser
import sys
import torch
from torch.utils.data import DataLoader

from architectures import get_architecture
from explainers import GradCAM, layer_selection
from data_manager import get_data
from hyperparameters import *


def parse_args():
    """
    This function gets the arguments of the program.

    Returns: a dictionary with the values of the arguments, in which the keys are the names defined for each argument in the second argument of each of the functions below.
    """

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-a", "--architecture", required=True, default=None, type=str)
    arg_parser.add_argument("-o", "--optimizer", required=True, default=None, type=str)
    arg_parser.add_argument("-r", "--run", required=True, default=1, type=int)
    arg_parser.add_argument("-l", "--learning_rate", required=True, default=None, type=float)
    arg_parser.add_argument("-e", "--explainer", required=False, default=None, type=str)

    # Parse the arguments and return them as a dictionary.
    return vars(arg_parser.parse_args())


args = parse_args()

model_name = str(args["run"]) + "_" + str(args["architecture"]) + "_" + str(args["optimizer"]) + "_" + str(args["learning_rate"])

# Beware!!! The training and validation datasets are NOT the same
# that were used to train the model.
# The separation is done randomly.
# The test dataset remains the same!
train_dataloader, val_dataloader, test_dataloader = get_data()


# Get the model and load the weights.
weights_path = "../model_checkpoints/" + model_name + ".pth"

model = get_architecture(args["architecture"], 
                         in_channels=DATA_HYPERPARAMETERS["IN_CHANNELS"], 
                         out_classes=DATA_HYPERPARAMETERS["NUM_CLASSES"], 
                         pretrained=False)

model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()

if args["explainer"] == "gradcam":
    target_layer = layer_selection(args["architecture"], model)
    attr_post_processing_function = None
    if type(target_layer) == tuple:
        target_layer, attr_post_processing_function = target_layer

    if target_layer is None:
        sys.exit("No function to get the target layer for the GradCAM found.")
    
    os.makedirs(f"../results/gradcam_{model_name}", exist_ok=True)

    while True:
        try:
            gradcam = GradCAM(model=model, 
                              target_layer=target_layer, 
                              dataloader=test_dataloader, 
                              classes=DATA_HYPERPARAMETERS["CLASSES"],
                              save_dir_path=f"../results/gradcam_{model_name}",
                              args=args,
                              attr_post_processing_function=attr_post_processing_function)
            gradcam.explain()
            gradcam.evaluate()
            break
        except Exception as e:
            if "CUDA out of memory" in str(e):
                new_batch_size = test_dataloader.batch_size // 2
                if new_batch_size <= 0:
                    raise e

                test_dataloader = DataLoader(test_dataloader.dataset, 
                                             batch_size=new_batch_size, 
                                             shuffle=False,
                                             num_workers=test_dataloader.num_workers)
                torch.cuda.empty_cache()
            else:
                raise e
                











