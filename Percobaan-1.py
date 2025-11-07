# ...existing code...
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Improved CNN for small medical images (e.g., ChestMNIST 28x28).
    - Conv-BatchNorm-ReLU blocks
    - MaxPool + Dropout for regularization
    - AdaptiveAvgPool to produce fixed-size features
    - Outputs single logit for binary (num_classes==2) or num_classes logits
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28 -> 14
            nn.Dropout2d(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 14 -> 7
            nn.Dropout2d(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 7 -> 3
            nn.Dropout2d(dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 128, 1, 1)

        hidden_dim = 128
        out_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    print("--- Menguji Model 'SimpleCNN' ---")
    model = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)

    dummy_input = torch.randn(8, IN_CHANNELS, 28, 28)
    output = model(dummy_input)

    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'SimpleCNN' berhasil.")
# ...existing code...