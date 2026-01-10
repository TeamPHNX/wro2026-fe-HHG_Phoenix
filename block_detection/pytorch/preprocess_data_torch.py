import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET_PATH = '../16.11._more_data - unfinished'
IMAGE_SIZE = (213, 100) # (width, height)
OUTPUT_FILE = 'processed_data.npz'

# Class mapping
CLASS_MAP = {
    'green_block': 1,
    'red_block': 2
}

def load_and_preprocess_image(path):
    """Loads an image, resizes it, and normalizes it."""
    try:
        img = Image.open(path).convert('RGB')
        original_size = img.size  # (width, height)
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        return img_array, original_size
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None, None

def parse_annotation(json_path, original_size):
    """Parses LabelMe JSON to extract bounding boxes and labels."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    boxes = []
    labels = []
    
    w_scale = IMAGE_SIZE[0] / original_size[0]
    h_scale = IMAGE_SIZE[1] / original_size[1]
    
    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in CLASS_MAP:
            continue
            
        points = shape['points']
        # LabelMe points are [[x1, y1], [x2, y2]] (top-left, bottom-right) or polygon
        
        # Extract x and y coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        xmin = min(xs) * w_scale
        xmax = max(xs) * w_scale
        ymin = min(ys) * h_scale
        ymax = max(ys) * h_scale
        
        # Standard TF Object Detection API uses [ymin, xmin, ymax, xmax] relative to image size [0, 1]
        # We will keep this format for consistency, but PyTorch often uses [xmin, ymin, xmax, ymax]
        # Let's stick to the original format [ymin, xmin, ymax, xmax] normalized to [0, 1]
        # to match the data loading expectation, or convert in main.py.
        # Original: box_norm = [ymin / IMAGE_SIZE[1], xmin / IMAGE_SIZE[0], ymax / IMAGE_SIZE[1], xmax / IMAGE_SIZE[0]]
        
        box_norm = [ymin / IMAGE_SIZE[1], xmin / IMAGE_SIZE[0], ymax / IMAGE_SIZE[1], xmax / IMAGE_SIZE[0]]
        
        boxes.append(box_norm)
        labels.append(CLASS_MAP[label])
        
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

def process_dataset():
    images = []
    all_boxes = []
    all_labels = []
    
    # Process subdirectories
    subdirs = ['green_blocks', 'red_blocks', 'background']
    
    print(f"Scanning {DATASET_PATH}...")
    
    file_list = []
    if not os.path.exists(DATASET_PATH):
         print(f"Error: Dataset path {DATASET_PATH} not found.")
         return

    for subdir in subdirs:
        dir_path = os.path.join(DATASET_PATH, subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} does not exist. Skipping.")
            continue
            
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dir_path, filename)
                # Check for corresponding JSON
                json_path = os.path.splitext(file_path)[0] + '.json'
                has_json = os.path.exists(json_path)
                
                file_list.append({
                    'image_path': file_path,
                    'json_path': json_path if has_json else None,
                    'subdir': subdir
                })
    
    print(f"Found {len(file_list)} images. processing...")
    
    for item in tqdm(file_list):
        img_array, original_size = load_and_preprocess_image(item['image_path'])
        if img_array is None:
            continue
            
        images.append(img_array)
        
        if item['json_path']:
            boxes, labels = parse_annotation(item['json_path'], original_size)
        else:
            # 'nothing' class or missing JSON
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int32)
            
        all_boxes.append(boxes)
        all_labels.append(labels)
    
    # Convert images to a single numpy array
    X = np.array(images, dtype=np.float32)
    
    output_path = OUTPUT_FILE
    np.savez_compressed(output_path, images=X, boxes=np.array(all_boxes, dtype=object), labels=np.array(all_labels, dtype=object))
    print(f"Saved processed data to {output_path}")
    print(f"Images shape: {X.shape}")
    print(f"Number of annotated samples: {len(all_boxes)}")

if __name__ == '__main__':
    process_dataset()
