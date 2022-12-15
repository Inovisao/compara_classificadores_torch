#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains the functions responsible for linking the main file to the architectures and optimizers.    
    
"""
import architectures as arch
import optimizers as optim

# Three lists, one with the architectures and another with the optimizers.
# If needed, more functions must be programmed in architectures.py and optimizers.py.
# After creating new functions, update these three lists.
architectures = {
    "alexnet": arch.alexnet,
    "coat_tiny": arch.coat_tiny,
    "maxvit_rmlp_tiny_rw_256": arch.maxvit_rmlp_tiny_rw_256,
    "vgg19": arch.vgg19,
    "lambda_resnet26rpt_256": arch.get_lambda_resnet26rpt_256,
    "vit_relpos_base_patch32_plus_rpn_256": arch.get_vit_relpos_base_patch32_plus_rpn_256,
    "sebotnet33ts_256": arch.get_sebotnet33ts_256,
    "lamhalobotnet50ts_256": arch.get_lamhalobotnet50ts_256,
    "swinv2_base_window16_256": arch.get_swinv2_base_window16_256,
    "convnext_base": arch.get_convnext_base,


}

optimizers = {
    "adam": optim.adam,
    "sgd": optim.sgd,
}

gradcam_layer_getters = {
    "alexnet": arch.get_alexnet_gradcam_layer,
    "coat_tiny": arch.get_coat_tiny_gradcam_layer,
    "maxvit_rmlp_tiny_rw_256": arch.get_maxvit_rmlp_tiny_rw_256_gradcam_layer,
    "vgg19": arch.get_vgg19_gradcam_layer,
    "lambda_resnet26rpt_256": arch.get_maxvit_rmlp_tiny_rw_256_gradcam_layer,
    "vit_relpos_base_patch32_plus_rpn_256": arch.get_vit_relpos_base_patch32_plus_rpn_256_gradcam_layer,
    "sebotnet33ts_256": arch.get_sebotnet33ts_256_gradcam_layer,
    "lamhalobotnet50ts_256": arch.get_lamhalobotnet50ts_256_gradcam_layer,
    "swinv2_base_window16_256": arch.get_swinv2_base_window16_256_gradcam_layer,
    "convnext_base": arch.get_convnext_base_gradcam_layer,

}


def get_optimizer(optimizer, model, learning_rate):
    # Return the optimizer.
    return optimizers[optimizer.casefold()](params=model.parameters(),
                                            learning_rate=learning_rate)


def get_architecture(architecture, in_channels, out_classes, pretrained):
    # Return the model.
    return architectures[architecture.casefold()](in_channels=in_channels,
                                                  out_classes=out_classes,
                                                  pretrained=pretrained)
