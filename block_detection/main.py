import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU to avoid CUDA library mismatch

import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = 'processed_data.npz'
BATCH_SIZE = 8 # Can be increased as the model is smaller
EPOCHS = 50 # Increased as the model is simpler and might need more epochs
LEARNING_RATE = 1e-3 # Increased from 1e-4 for faster convergence
IMAGE_SIZE = (100, 213) # (height, width)
NUM_CLASSES = 3  # Background (0), Green (1), Red (2)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with np.load(file_path, allow_pickle=True) as data:
        X = data['images']
        boxes_ragged = data['boxes']
        labels_ragged = data['labels']
    
    # Process targets: For this simple model, we will detect the LARGEST object in the image.
    # If you need to detect multiple objects, you'll need a more complex architecture (like SSD or YOLO).
    
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
    
    # One-hot encode classes
    y_class_one_hot = keras.utils.to_categorical(y_class, num_classes=NUM_CLASSES)
    
    # Split into train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_class_train, y_class_val = y_class_one_hot[:split_idx], y_class_one_hot[split_idx:]
    y_box_train, y_box_val = y_box[:split_idx], y_box[split_idx:]
    
    return (X_train, y_class_train, y_box_train), (X_val, y_class_val, y_box_val)

def create_model():
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    # Feature Extractor
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification Head
    class_x = layers.Dense(256, activation='relu')(x)
    class_x = layers.BatchNormalization()(class_x)
    class_x = layers.Dropout(0.3)(class_x)
    class_x = layers.Dense(128, activation='relu')(class_x)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(class_x)
    
    # Bounding Box Head
    box_x = layers.Dense(256, activation='relu')(x)
    box_x = layers.BatchNormalization()(box_x)
    box_x = layers.Dropout(0.3)(box_x)
    box_x = layers.Dense(128, activation='relu')(box_x)
    box_output = layers.Dense(4, activation='sigmoid', name='box_output')(box_x)
    
    model = models.Model(inputs=inputs, outputs=[class_output, box_output])
    return model

def train():
    if not tf.io.gfile.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run preprocess_data.py first.")
        return

    (X_train, y_class_train, y_box_train), (X_val, y_class_val, y_box_val) = load_data(DATA_FILE)
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
    
    model = create_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'class_output': 'categorical_crossentropy',
            'box_output': 'mse' # Mean Squared Error for regression
        },
        loss_weights={
            'class_output': 1.0,
            'box_output': 10.0 # Weight box loss higher to emphasize accurate localization
        },
        metrics={
            'class_output': 'accuracy',
            'box_output': 'mse'
        }
    )
    
    history = model.fit(
        X_train,
        {'class_output': y_class_train, 'box_output': y_box_train},
        validation_data=(X_val, {'class_output': y_class_val, 'box_output': y_box_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.ModelCheckpoint('models/new_best_model.keras', save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(patience=5),
            keras.callbacks.EarlyStopping(patience=5)
        ]
    )
    
    model.save('models/final_model.keras')
    print("Training finished. Model saved.")

if __name__ == '__main__':
    train()
