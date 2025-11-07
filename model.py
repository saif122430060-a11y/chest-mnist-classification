import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    """
    DenseNet-121 untuk klasifikasi medical images (ChestMNIST 28x28).
    - Pre-trained DenseNet-121 dari ImageNet
    - Input layer dimodifikasi untuk 1-channel grayscale
    - Classifier disesuaikan untuk binary classification
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pre-trained DenseNet-121
        densenet = models.densenet121(pretrained=pretrained)
        
        # Modifikasi convolution pertama: stride=1 (tidak 2) untuk input 28x28
        densenet.features[0] = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        
        # Hapus MaxPool yang pertama karena input sudah kecil
        densenet.features[1] = nn.BatchNorm2d(64)
        densenet.features[2] = nn.ReLU(inplace=True)
        # Skip MaxPool2d di features[3]
        
        self.features = nn.Sequential(*list(densenet.features.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        num_features = 1024  # Output dari DenseNet-121 features
        out_dim = 1 if num_classes == 2 else num_classes
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'DenseNet-121' ---")
    
    model = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(8, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'DenseNet-121' berhasil.")