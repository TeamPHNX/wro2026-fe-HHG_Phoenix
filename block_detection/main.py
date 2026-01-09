import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = 'processed_data.npz'
BATCH_SIZE = 8 # Reduced to save VRAM
EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
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
    y_class_one_hot = tf.keras.utils.to_categorical(y_class, num_classes=NUM_CLASSES)
    
    # Split into train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_class_train, y_class_val = y_class_one_hot[:split_idx], y_class_one_hot[split_idx:]
    y_box_train, y_box_val = y_box[:split_idx], y_box[split_idx:]
    
    return (X_train, y_class_train, y_box_train), (X_val, y_class_val, y_box_val)

def create_model():
    # Use MobileNetV2 as backbone
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True # Fine-tune
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification Head
    class_output = layers.Dense(128, activation='relu')(x)
    class_output = layers.Dropout(0.5)(class_output)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(class_output)
    
    # Bounding Box Head
    box_output = layers.Dense(128, activation='relu')(x)
    box_output = layers.Dropout(0.5)(box_output)
    box_output = layers.Dense(4, activation='sigmoid', name='box_output')(box_output) # Sigmoid because coordinates are 0-1
    
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
            tf.keras.callbacks.ModelCheckpoint('models/new_best_model.keras', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3),
            tf.keras.callbacks.EarlyStopping(patience=5)
        ]
    )
    
    model.save('models/final_model.keras')
    print("Training finished. Model saved.")

if __name__ == '__main__':
    train()
