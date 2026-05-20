import torch
import timm

def get_net(model_name, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model