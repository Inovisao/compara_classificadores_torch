#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Architectures are defined in this file.
"""
import os
import wget
import timm
from torch import nn
from torchvision import models
import numpy as np
from hyperparameters import DATA_HYPERPARAMETERS
from IELT.models.vit import get_b16_config
from IELT.models.IELT import InterEnsembleLearningTransformer


def alexnet(in_channels, out_classes, pretrained):
    # Get the model with or without pretrained weights.
    model = models.alexnet(weights=models.alexnet if pretrained else None)

    # Adjust the last layer.
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, 
                                    out_features=out_classes, 
                                    bias=True)

    # Adjust the first layer.
    model.features[0] = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=model.features[0].out_channels, 
                                  kernel_size=(11, 11), 
                                  stride=(4, 4), 
                                  padding=(2, 2))

    return model


def get_alexnet_gradcam_layer(model):
    return model.features[-3]


def vgg19(in_channels, out_classes, pretrained):
    # Get the model with or without pretrained weights.
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)

    # Adjust the last layer.
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, 
                                    out_features=out_classes, 
                                    bias=True)

    # Adjust the first layer.
    model.features[0] = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=model.features[0].out_channels, 
                                  kernel_size=(3, 3), 
                                  stride=(1, 1), 
                                  padding=(1, 1))
    
    return model


def get_vgg19_gradcam_layer(model):
    return model.features[-3]

def maxvit_rmlp_tiny_rw_256(in_channels, out_classes, pretrained):
    """
    Multi-axis vision transformer: https://arxiv.org/abs/2204.01697
    """
    model = timm.create_model("maxvit_rmlp_tiny_rw_256",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)

    return model


def get_maxvit_rmlp_tiny_rw_256_gradcam_layer(model):
    return model.stages[3].blocks[1].conv.conv3_1x1


def coat_tiny(in_channels, out_classes, pretrained):
    """
    Co-Scale Conv-Attentional Image Transformers: https://arxiv.org/abs/2104.06399
    """
    model = timm.create_model("coat_tiny",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)

    return model


def get_coat_tiny_gradcam_layer(model):
    return model.parallel_blocks[5].factoratt_crpe4.crpe.conv_list[2]


def get_lambda_resnet26rpt_256(in_channels, out_classes, pretrained):
    model = timm.create_model("lambda_resnet26rpt_256", 
                              pretrained=pretrained, 
                              in_chans=in_channels, 
                              num_classes=out_classes)
    
    return model


def get_lambda_resnet26rpt_256_gradcam_layer(model):
    return model.stages[3][1].conv1_1x1.conv


def get_vit_relpos_base_patch32_plus_rpn_256(in_channels, out_classes, pretrained):
    model = timm.create_model("vit_relpos_base_patch32_plus_rpn_256",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)

    return model

def get_vit_relpos_base_patch32_plus_rpn_256_gradcam_layer(model):
    print("Gradcam not available for vit_relpos_base_patch32_plus_rpn_256.")


def get_sebotnet33ts_256(in_channels, out_classes, pretrained):
    model = timm.create_model("sebotnet33ts_256",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)
    
    return model


def get_sebotnet33ts_256_gradcam_layer(model):
    return model.final_conv.conv


def get_lamhalobotnet50ts_256(in_channels, out_classes, pretrained):
    model = timm.create_model("lamhalobotnet50ts_256",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)

    return model


def get_lamhalobotnet50ts_256_gradcam_layer(model):
    return model.stages[3][2].conv3_1x1.conv


def get_swinv2_base_window16_256(in_channels, out_classes, pretrained):
    model = timm.create_model("swinv2_base_window16_256",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)
    
    return model
    

def get_swinv2_base_window16_256_gradcam_layer(model):
    print("Gradcam not available for swinv2_base_window16_256.")


def get_swinv2_cr_base_224(in_channels, out_classes, pretrained):
    model = timm.create_model("swinv2_cr_base_224",
                              pretrained=pretrained,
                              in_chans=in_channels,
                              num_classes=out_classes)
    return model


def get_swinv2_cr_base_224_gradcam_layer(model):
    print("Gradcam not available for winv2_cr_base_224.")


def get_convnext_base(in_channels, out_classes, pretrained):
    """
    ConvNeXt model from vanilla PyTorch.
    """
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights) if pretrained else models.convnext_base()
    model.classifier[2] = nn.Linear(in_features=1024, out_features=out_classes, bias=True)
    model.features[0][0] = nn.Conv2d(in_channels, 128, kernel_size=(4, 4), stride=(4, 4))

    return model


def get_convnext_base_gradcam_layer(model):
    return model.features[7][2].block[0]


def get_resnet18(in_channels, out_classes, pretrained):
    """
    ResNet18 from PyTorch vanilla.
    """
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(in_features=512, out_features=out_classes, bias=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def get_resnet18_gradcam_layer(model):
    return model.layer4[1].conv2


def get_resnet50(in_channels, out_classes, pretrained):
    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(in_features=2048, out_features=out_classes, bias=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model

def get_resnet50_gradcam_layer(model):
    return model.layer4[1].conv2


def get_resnet101(in_channels, out_classes, pretrained):
    model = models.resnet101(weights="DEFAULT")
    model.fc = nn.Linear(in_features=2048, out_features=out_classes, bias=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model


def get_resnet101_gradcam_layer(model):
    return model.layer4[1].conv2

def get_mobilenetV3(in_channels, out_classes, pretrained):
    model = models.mobilenet_v3_large(pretrained=pretrained)
    model.classifier[3]=nn.Linear(in_features=1280, out_features=out_classes, bias=True)
    model.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1), bias=False)
    return model

def get_mobilenetV3_gradcam_layer(model):
    return model.features[-1]



 def get_densenet201(in_channels, out_classes, pretrained):
    model = models.densenet201(weights="DEFAULT")
    model.fc = nn.Linear(in_features=1920, out_features=out_classes, bias=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model

def get_densenet201_gradcam_layer(model):
    return model.features.transition3.conv
    
       
def get_ielt(in_channels, out_classes, pretrained):
    
    if not os.path.exists("./IELT/pretrained"):
        os.mkdir("./IELT/pretrained")
    
    if not os.path.exists("./IELT/pretrained/ViT-B_16.npz"):
        weights_url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
        path_pretrained = "./IELT/pretrained/ViT-B_16.npz"
        wget.download(weights_url, path_pretrained)
        
    config = get_b16_config()
    model = InterEnsembleLearningTransformer(config, img_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"], num_classes=out_classes)
    
    if pretrained:
        model.load_from(np.load("./IELT/pretrained/ViT-B_16.npz"))
    
    model.embeddings.patch_embeddings = nn.Conv2d(in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
    model.softmax = nn.Identity()
    
    return model

def get_ielt_gradcam_layer(model):
    print("GradCAM not available for IELT.")
    

def get_default_siamese(in_channels, out_classes, pretrained):
    embedding_model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=10, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(64, 128, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(128, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(128, 256, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(9216, out_classes)
    )
    return embedding_model

def get_siamese_gradcam_layer(model):
    print("GradCAM not available for Siamese.")






















