import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


# ============================================================
# Model: DenseNet121
# ============================================================
class DenseNet121(nn.Module):
    """
    DenseNet-121 untuk klasifikasi medical images (ChestMNIST 28x28).
    - Pre-trained DenseNet-121 dari ImageNet
    - Input layer dimodifikasi untuk 1-channel grayscale
    - Classifier disesuaikan untuk binary classification
    - Dengan data augmentation dan segmentation support
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pre-trained DenseNet-121
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modifikasi convolution pertama: stride=1 untuk input 28x28
        densenet.features[0] = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        
        # Hapus MaxPool pertama (features[3]) karena input sudah kecil
        features_list = list(densenet.features.children())
        self.features = nn.Sequential(
            *features_list[:3],
            *features_list[4:]
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        num_features = 1024
        out_dim = 1 if num_classes == 2 else num_classes
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ============================================================
# Data Augmentation untuk Training
# ============================================================
def get_train_transforms():
    """
    Augmentasi data untuk training set
    """
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
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
# Segmentasi untuk preprocessing
# ============================================================
class SegmentationPreprocessor(nn.Module):
    """
    Preprocessing layer untuk segmentasi chest X-ray
    """
    def __init__(self, threshold: float = 0.4, use_morphology: bool = True):
        super().__init__()
        self.threshold = threshold
        self.use_morphology = use_morphology
        
    def forward(self, x):
        """
        x: tensor grayscale (B, 1, H, W)
        output: tensor dengan background dihapus
        """
        # Normalisasi ke range [0, 1] - Ganti view dengan reshape
        x_flat = x.reshape(x.size(0), -1)
        x_min = x_flat.min(dim=1)[0].reshape(-1, 1, 1, 1)
        x_max = x_flat.max(dim=1)[0].reshape(-1, 1, 1, 1)
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Adaptive thresholding
        binary_mask = (x_normalized > self.threshold).float()
        
        if self.use_morphology:
            kernel_size = 3
            padded = torch.nn.functional.pad(binary_mask, (1, 1, 1, 1), mode='constant', value=0)
            dilated = torch.nn.functional.max_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
            padded2 = torch.nn.functional.pad(dilated, (1, 1, 1, 1), mode='constant', value=1)
            eroded = 1 - torch.nn.functional.max_pool2d(1 - padded2, kernel_size=kernel_size, stride=1, padding=0)
            binary_mask = eroded
        
        segmented = x * binary_mask
        return segmented


def apply_segmentation_to_batch(batch, segmentor: SegmentationPreprocessor):
    """
    Apply segmentation ke batch dari dataloader
    """
    images, labels = batch
    segmented_images = segmentor(images)
    return segmented_images, labels


# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    print("--- Menguji Model 'DenseNet121' dengan Augmentation & Segmentation ---\n")
    
    # Test Model
    model = DenseNet121(in_channels=1, num_classes=2, pretrained=True)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("✓ Pengujian model 'DenseNet121' berhasil.\n")
    
    # Test Transforms
    print("--- Testing Data Augmentation Transforms ---")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    print("✓ Train transforms loaded")
    print("✓ Val transforms loaded\n")
    
    # Test Segmentation
    print("--- Testing Segmentation Preprocessor ---")
    segmentor = SegmentationPreprocessor(threshold=0.4, use_morphology=True)
    segmented = segmentor(dummy_input)
    print(f"Original input shape: {dummy_input.shape}")
    print(f"Segmented output shape: {segmented.shape}")
    print(f"Segmentation intensity range: [{segmented.min():.4f}, {segmented.max():.4f}]")
    print("✓ Segmentation preprocessor test passed!\n")
    
    print("="*60)
    print("Semua test berhasil! Model siap untuk training.")
    print("="*60)