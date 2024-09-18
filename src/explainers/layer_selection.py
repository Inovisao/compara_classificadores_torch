

def layer_selection(model_name, model):
    if model_name == "resnet50":
        return model.layer4[-1]
    
    if model_name == "resnet18":
        return model.layer4[-1]

    if model_name == "maxvit_rmlp_tiny_rw_256":
        return model.stages[-1]

    return None
    


    