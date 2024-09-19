import torch


def layer_selection(model_name, model):
    if model_name == "resnet50":
        return model.layer4[-1]
    
    if model_name == "resnet18":
        return model.layer4[-1]
    
    if model_name == "resnet101":
        return model.layer4[-1]

    if model_name == "maxvit_rmlp_tiny_rw_256":
        return model.stages[-1]
    
    if model_name == "swinv2_base_window16_256":
        def post_process(attr: torch.tensor) -> torch.tensor:
            return attr.sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        return model.layers[-1], post_process

    return None
    


    