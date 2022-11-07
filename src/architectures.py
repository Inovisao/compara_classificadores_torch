#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Architectures are defined in this file.
"""
import timm
from torch import nn
from torchvision import models


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

































