#!/bin/bash

weights_path="../model_checkpoints"
dobras_dir="../data/dobras"


# Remove the previous results
rm -rf ./results_gradcam_multilayer_resnet18
mkdir -p ./results_gradcam_multilayer_resnet18


# Parse the command line arguments
while getopts "k:" flag; 
do
   case "${flag}" in
      k) ndobras=${OPTARG}
         ;;
   esac
done


for ((i=1;i<=$ndobras;i+=1)); do
    weights_name="1_resnet18_sgd_0.001.pth"
    folds_name="fold_${i}"
    mkdir -p ./results
    mkdir -p ./results/results_gradcam_multilayer_resnet18/${folds_name}

    # Run the ResNet model
    echo "Running explanations for ResNet18"
    echo "Using weights: ${weights_path}/${weights_name}"
    echo "Using test data: ${dobras_dir}/${folds_name}"

    python3 resnet_explainer.py -f ${i} -w ${weights_path}/${weights_name} -td ${dobras_dir}/${folds_name} -o ./results/results_gradcam_multilayer_resnet18/${folds_name}

done


# Run the ResNet explainer
python3 resnet_explainer.py -w ${weight_path} -td ${test_data_path}
