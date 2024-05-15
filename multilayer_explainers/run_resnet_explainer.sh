#!/bin/bash

# Remove the previous results
rm -rf ./results_gradcam_multilayer/*

# Run the ResNet explainer
python3 resnet_explainer.py -w ../model_checkpoints/1_resnet18_sgd_0.001.pth -td ../data/dobras/fold_1