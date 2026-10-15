from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class OsteoEfficientNetB0(nn.Module):
    """EfficientNet-B0 branch for histopathology or radiograph classification.

    Histopathology branch:
        num_classes = 3
        labels = Non tumor, Viable tumor, Necrotic tumor

    Radiograph branch:
        num_classes = 2
        labels = Other, Tumor
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        unfreeze_last_n_feature_blocks: int = 2,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n_feature_blocks > 0:
            for param in self.backbone.features[-unfreeze_last_n_feature_blocks:].parameters():
                param.requires_grad = True

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
