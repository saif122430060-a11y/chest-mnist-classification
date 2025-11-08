import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class MobileNetV3Large(nn.Module):
    """
    MobileNet-V3 Large Fine-Tuned untuk ChestMNIST (28x28 grayscale).
    - Menggunakan torchvision.models.mobilenet_v3_large
    - Modifikasi conv stem untuk 1-channel input dan stride=1
    - Custom classifier untuk binary classification
    - Model ringan dengan performa tinggi
    - Dengan data augmentation untuk training
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load MobileNet-V3 Large
        mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Modifikasi conv pertama untuk grayscale input (1 channel) dan stride=1
        old_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,  # input kecil (28x28), jangan stride 2
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Freeze backbone jika diperlukan (untuk fine-tuning ringan)
        if freeze_backbone:
            for param in mobilenet.features.parameters():
                param.requires_grad = False
        
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Ambil jumlah features dari classifier layer
        num_features = mobilenet.classifier[0].in_features
        
        # Custom classifier head (tanpa inplace=True untuk menghindari gradient issues)
        out_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(inplace=False),
            nn.Dropout(0.4, inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3, inplace=False),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# Data Augmentation untuk Training
# ============================================================
def get_train_transforms():
    """
    Augmentasi data untuk training set dengan segmentasi dan transformasi
    """
    return transforms.Compose([
        # Resize ke 28x28 (standar ChestMNIST)
        transforms.Resize((28, 28)),
        
        # Augmentasi geometrik
        transforms.RandomRotation(15),  # Rotasi random 0-15 derajat
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translasi random
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip 50%
        
        # Augmentasi intensitas
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness & contrast
        
        # Gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        
        # Normalisasi
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])


def get_val_transforms():
    """
    Transformasi data untuk validation set (tanpa augmentasi)
    """
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])


# ============================================================
# Segmentasi untuk preprocessing (opsional)
# ============================================================
class SegmentationPreprocessor(nn.Module):
    """
    Preprocessing layer untuk segmentasi chest X-ray
    Menggunakan thresholding dan morphological operations
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x):
        """
        x: tensor grayscale (B, 1, H, W)
        output: tensor dengan background dihapus
        """
        # Normalisasi ke range [0, 1]
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Adaptive thresholding
        binary_mask = (x_normalized > self.threshold).float()
        
        # Morphological operations (dilation)
        kernel = torch.ones(1, 1, 3, 3, device=x.device)
        padded = torch.nn.functional.pad(binary_mask, (1, 1, 1, 1), mode='constant', value=0)
        dilated = torch.nn.functional.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
        
        # Apply mask ke original image
        segmented = x * dilated
        
        return segmented


if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    # Test Model
    model = MobileNetV3Large(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("Model MobileNet-V3 Large dengan Augmentation & Segmentation:")
    print(model)
    
    # Test dengan dummy input
    dummy = torch.randn(4, IN_CHANNELS, 28, 28)
    out = model(dummy)
    print(f"\nInput shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print("Test passed!")
    
    # Test transforms
    print("\n--- Data Augmentation Transforms ---")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    print("Train transforms:", train_transform)
    print("\nVal transforms:", val_transform)
    
    # Test segmentation
    print("\n--- Segmentation Preprocessor ---")
    segmentor = SegmentationPreprocessor(threshold=0.4)
    segmented = segmentor(dummy)
    print(f"Segmented output shape: {segmented.shape}")