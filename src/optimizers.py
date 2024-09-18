#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Optimizers are defined in this file.

"""

from lion_pytorch import Lion
from sam import SAM
from torch import optim


SAM_BASE_OPTIM = optim.SGD

def get_optimizer(optimizer, model, learning_rate):
    def adam(params, learning_rate):
        return optim.Adam(params=params,
                          lr=learning_rate,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0,
                          amsgrad=False,
                          foreach=None,
                          maximize=False,
                          capturable=False)


    def sgd(params, learning_rate):
        return optim.SGD(params=params, 
                         lr=learning_rate,
                         momentum=0,
                         dampening=0,
                         weight_decay=0,
                         nesterov=False,
                         maximize=False,
                         foreach=None)


    def adagrad(params, learning_rate):
        return optim.Adagrad(params=params,
                             lr=learning_rate,
                             lr_decay=0,
                             weight_decay=0,
                             initial_accumulator_value=0,
                             eps=1e-10,
                             foreach=None,
                             maximize=False)


    def adamw(params, learning_rate):
        return optim.AdamW(params,
                           lr=learning_rate,
                           betas=(0.9, 0.99),
                           eps=1e-10,
                           weight_decay=0,
                           foreach=None,
                           maximize=False)


    def lion(params, learning_rate):
        return Lion(params=params, 
                    lr=learning_rate, 
                    betas=(0.9, 0.99), 
                    weight_decay=0.0)


    def sam(params, learning_rate):
        optimizer = SAM(params, 
                        SAM_BASE_OPTIM, 
                        lr=learning_rate, 
                        momentum=0.9)

        optimizer.__name__ = "sam"

        return optimizer
    

    # Chama a função adequada e retorna o otimizador.
    return locals()[optimizer.casefold()](model.parameters(), learning_rate)



