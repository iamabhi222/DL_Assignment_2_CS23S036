import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
def modified_model():
    # input_size = (3, 224, 224)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False # freeze
    model.fc.requires_grad = True

    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    return model

    # freeze = len(model.layers) - frozen_before
    # for i, param in model.childern():
    #     if(i < freeze):
    #         param.requires_grad = False
    #     else:
    #         break

    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc.requires_grad = True

    # model.fc = torch.nn.Linear(model.fc.in_features, dense_size)
    # new_model = nn.Sequential(
    #     model,
    #     nn.Dropout(dropout),
    #     nn.ReLU(),
    #     nn.Linear(dense_size, 10),  # Output layer with 10 classes
    #     nn.Softmax(dim=1)  # Softmax activation function for multiclass classification
    # )

    # return new_model
