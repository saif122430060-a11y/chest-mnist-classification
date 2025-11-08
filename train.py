# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
from efficientnet_b0 import EfficientNetB0
from mobilenet_v3 import (
    MobileNetV3Large, 
    get_train_transforms, 
    get_val_transforms,
    SegmentationPreprocessor
)
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Pilih Model ---
# Ubah ke 'densenet' untuk menggunakan DenseNet-121
# Ubah ke 'efficientnet' untuk menggunakan EfficientNet-B0
# Ubah ke 'mobilenet' untuk menggunakan MobileNet-V3 Large
MODEL_CHOICE = 'mobilenet'  # 'densenet', 'efficientnet', atau 'mobilenet'

# --- Hyperparameter ---
EPOCHS = 60
BATCH_SIZE = 16

# Learning rate berbeda untuk setiap model
if MODEL_CHOICE == 'densenet':
    LEARNING_RATE = 1e-4
elif MODEL_CHOICE == 'efficientnet':
    LEARNING_RATE = 3e-4
else:  # mobilenet
    LEARNING_RATE = 1e-3

def train():
    # 1. Memuat Data dengan Augmentation
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # Initialize segmentation preprocessor (opsional)
    segmentor = SegmentationPreprocessor(threshold=0.4) if MODEL_CHOICE == 'mobilenet' else None
    
    # 2. Inisialisasi Model
    if MODEL_CHOICE == 'densenet':
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes, pretrained=True)
        print("Model DenseNet-121 berhasil dimuat!")
        model_name = 'DenseNet-121'
        save_name = 'densenet121_chest.pth'
    elif MODEL_CHOICE == 'efficientnet':
        model = EfficientNetB0(in_channels=in_channels, num_classes=num_classes, pretrained=True)
        print("Model EfficientNet-B0 berhasil dimuat!")
        model_name = 'EfficientNet-B0'
        save_name = 'efficientnet_b0_chest.pth'
    elif MODEL_CHOICE == 'mobilenet':
        model = MobileNetV3Large(in_channels=in_channels, num_classes=num_classes, pretrained=True)
        print("Model MobileNet-V3 Large berhasil dimuat!")
        model_name = 'MobileNet-V3 Large'
        save_name = 'mobilenet_v3_chest.pth'
    else:
        raise ValueError(f"Model '{MODEL_CHOICE}' tidak dikenali. Gunakan 'densenet', 'efficientnet', atau 'mobilenet'")
    
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler (tanpa verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print(f"\n--- Memulai Training dengan {model_name} ---")
    print(f"Learning Rate: {LEARNING_RATE}\n")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            # Handle label shape untuk BCEWithLogitsLoss
            labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("\n--- Training Selesai ---")
    
    # Hitung rata-rata akurasi validasi
    avg_val_accuracy_over_epochs = sum(val_accs_history) / len(val_accs_history) if val_accs_history else 0.0
    
    # Tampilkan akurasi tertinggi dan rata-rata validasi
    max_train_acc = max(train_accs_history) if train_accs_history else 0.0
    max_val_acc = max(val_accs_history) if val_accs_history else 0.0
    best_epoch_train = train_accs_history.index(max_train_acc) + 1 if train_accs_history else None
    best_epoch_val = val_accs_history.index(max_val_acc) + 1 if val_accs_history else None
    
    print(f"\n{'='*60}")
    print(f"HASIL TRAINING - {model_name}")
    print(f"{'='*60}")
    print(f"Akurasi Training Tertinggi: {max_train_acc:.2f}% (Epoch {best_epoch_train})")
    print(f"Akurasi Validasi Tertinggi: {max_val_acc:.2f}% (Epoch {best_epoch_val})")
    print(f"Rata-rata Akurasi Validasi (selama training): {avg_val_accuracy_over_epochs:.2f}%")
    print(f"{'='*60}\n")
    
    # Save model
    torch.save(model.state_dict(), save_name)
    print(f"Model disimpan sebagai '{save_name}'\n")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()