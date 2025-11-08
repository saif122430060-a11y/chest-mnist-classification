import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 Fine-Tuned untuk ChestMNIST (28x28 grayscale).
    - Menggunakan torchvision.models.efficientnet_b0
    - Modifikasi conv stem untuk 1-channel input dan stride=1
    - Custom classifier untuk binary classification
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load EfficientNet-B0
        effnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Modifikasi conv pertama untuk grayscale input (1 channel) dan stride=1
        old_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,  # input kecil (28x28), jangan stride 2
            padding=old_conv.padding,
            bias=False
        )
        
        # Freeze backbone jika diperlukan (untuk fine-tuning ringan)
        if freeze_backbone:
            for param in effnet.features.parameters():
                param.requires_grad = False
        
        self.features = effnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Ambil jumlah features dari classifier layer
        num_features = effnet.classifier[1].in_features
        
        # Custom classifier head
        out_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    model = EfficientNetB0(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("Model EfficientNet-B0:")
    print(model)

    dummy = torch.randn(4, IN_CHANNELS, 28, 28)
    out = model(dummy)
    print(f"\nInput shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print("Test passed!")