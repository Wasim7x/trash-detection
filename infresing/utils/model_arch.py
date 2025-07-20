import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Transfer, self).__init__()

        # Load pretrained ResNet50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Remove the final FC layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])

        # Custom classifier (mimicking Keras: Flatten -> Dense(512) -> Dense(num_classes))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Equivalent to Flattening
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)  # for multi-class classification
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x
