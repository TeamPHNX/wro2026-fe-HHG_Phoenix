import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Configuration
DATA_FILE = 'processed_data.npz'
BATCH_SIZE = 16 
EPOCHS = 50 
LEARNING_RATE = 1e-3 
IMAGE_SIZE = (100, 213) # (height, width) - Note: PyTorch uses (C, H, W)
NUM_CLASSES = 3  # Background (0), Green (1), Red (2)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BlockDataset(Dataset):
    def __init__(self, X, y_class, y_box):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        self.y_class = torch.LongTensor(np.argmax(y_class, axis=1)) # Convert one-hot to class index
        self.y_box = torch.FloatTensor(y_box)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_box[idx]

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with np.load(file_path, allow_pickle=True) as data:
        X = data['images']
        boxes_ragged = data['boxes']
        labels_ragged = data['labels']
    
    y_class = []
    y_box = []
    
    valid_indices = []
    
    for i in range(len(boxes_ragged)):
        current_boxes = boxes_ragged[i]
        current_labels = labels_ragged[i]
        
        if len(current_boxes) > 0:
            # Find the largest box by area
            areas = (current_boxes[:, 2] - current_boxes[:, 0]) * (current_boxes[:, 3] - current_boxes[:, 1])
            largest_idx = np.argmax(areas)
            
            y_box.append(current_boxes[largest_idx])
            y_class.append(current_labels[largest_idx])
            valid_indices.append(i)
        else:
            # No object
            y_box.append([0, 0, 0, 0])
            y_class.append(0) # Background
            valid_indices.append(i)

    X = X[valid_indices]
    y_box = np.array(y_box, dtype=np.float32)
    y_class = np.array(y_class, dtype=np.int32)

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y_box = y_box[indices]
    y_class = y_class[indices]
    
    # Create one-hot (for consistency logic, though we convert back for PyTorch CrossEntropy)
    y_class_one_hot = np.eye(NUM_CLASSES)[y_class]
    
    # Split into train/val
    split_idx = int(0.8 * len(X))
    
    train_dataset = BlockDataset(
        X[:split_idx], 
        y_class_one_hot[:split_idx], 
        y_box[:split_idx]
    )
    
    val_dataset = BlockDataset(
        X[split_idx:], 
        y_class_one_hot[split_idx:], 
        y_box[split_idx:]
    )
    
    return train_dataset, val_dataset

class BlockDetector(nn.Module):
    def __init__(self):
        super(BlockDetector, self).__init__()
        
        # Feature Extractor - Increased complexity (~3x parameters)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # No BatchNorm here to preserve features for global pooling
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
        
        # Bounding Box Head
        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        class_out = self.class_head(x)
        box_out = self.box_head(x)
        
        return class_out, box_out

def train():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run preprocess_data.py first.")
        return

    train_dataset, val_dataset = load_data(DATA_FILE)
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BlockDetector().to(device)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if not os.path.exists('models'):
        os.makedirs('models')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, boxes in train_loader:
            inputs, labels, boxes = inputs.to(device), labels.to(device), boxes.to(device)
            
            optimizer.zero_grad()
            
            class_out, box_out = model(inputs)
            
            loss_class = class_criterion(class_out, labels)
            loss_box = box_criterion(box_out, boxes)
            
            # Weighted loss
            loss = loss_class * 1.0 + loss_box * 10.0
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, boxes in val_loader:
                inputs, labels, boxes = inputs.to(device), labels.to(device), boxes.to(device)
                
                class_out, box_out = model(inputs)
                
                loss_class = class_criterion(class_out, labels)
                loss_box = box_criterion(box_out, boxes)
                
                loss = loss_class * 1.0 + loss_box * 10.0
                val_loss += loss.item()
                
                _, predicted = torch.max(class_out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/new_best_model.pth')
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
    torch.save(model.state_dict(), 'models/final_model.pth')
    print("Training finished.")

if __name__ == '__main__':
    train()
