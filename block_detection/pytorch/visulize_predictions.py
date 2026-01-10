import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import os
import random
import glob
import cv2
import time

# Configuration
MODEL_PATH = 'models/new_best_model.pth'
DATASET_PATH = 'testing_data'
IMAGE_SIZE = (213, 100) # (width, height)
NUM_CLASSES = 3

# Class mapping
CLASS_MAP = {
    0: 'Background',
    1: 'Green Block',
    2: 'Red Block'
}

COLORS = {
    0: 'gray',
    1: '#00ff00', # Green
    2: '#ff0000'  # Red
}

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (must match training)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BlockDetector(nn.Module):
    def __init__(self):
        super(BlockDetector, self).__init__()
        
        # Ultra High-Capacity Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.MaxPool2d(2),
            
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            nn.MaxPool2d(2),
            
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            nn.MaxPool2d(2),
            
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            
            ResidualBlock(512, 1024, stride=1),
            ResidualBlock(1024, 1024, stride=1),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Massive Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_CLASSES)
        )
        
        # Massive Bounding Box Head
        self.box_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        class_out = self.class_head(x)
        box_out = self.box_head(x)
        
        return class_out, box_out

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phoenix Block Detection - Advanced Viewer (PyTorch)")
        self.root.geometry("1000x850")
        self.root.configure(bg="#1e1e1e") # Dark theme

        # Load Model
        try:
            print("Loading model...")
            self.model = BlockDetector().to(device)
            # Check if model file exists before loading
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                self.model.eval()
                print("Model loaded successfully.")
            else:
                print(f"Warning: Model file {MODEL_PATH} not found.")
                self.model = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model = None
        
        # State
        self.video_running = False
        self.is_paused = False
        self.cap = None
        self.frames = None
        self.video_mode = None
        self.current_frame_idx = 0
        
        # Settings
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.show_overlay = tk.BooleanVar(value=True)
        self.playback_speed = tk.IntVar(value=33) # ms delay

        # Gather all dataset images for random selection
        self.all_images = glob.glob(os.path.join(DATASET_PATH, '**', '*.png'), recursive=True)
        if not self.all_images:
            print("Warning: No images found in dataset path.")

        self.setup_ui()

    def setup_ui(self):
        # Main Layout: Sidebar and Main Content
        self.main_container = tk.Frame(self.root, bg="#1e1e1e")
        self.main_container.pack(fill="both", expand=True)

        # Sidebar for Controls
        self.sidebar = tk.Frame(self.main_container, bg="#2d2d2d", width=250, padx=15, pady=20)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Header in Sidebar
        header = tk.Label(self.sidebar, text="CONTROLS", font=("Helvetica", 14, "bold"), bg="#2d2d2d", fg="#ffffff")
        header.pack(pady=(0, 20))

        # group: File Actions
        file_frame = tk.LabelFrame(self.sidebar, text="File Operations", bg="#2d2d2d", fg="#aaaaaa", padx=10, pady=10)
        file_frame.pack(fill="x", pady=10)

        side_btn_style = {"font": ("Helvetica", 10), "bg": "#3d3d3d", "fg": "white", "bd": 0, "pady": 5, "cursor": "hand2", "activebackground": "#555555"}
        
        tk.Button(file_frame, text="🎲 Random Image", command=self.load_random_image, **side_btn_style).pack(fill="x", pady=2)
        tk.Button(file_frame, text="📂 Load image", command=self.load_manual_image, **side_btn_style).pack(fill="x", pady=2)
        tk.Button(file_frame, text="🎥 Load Video/Data", command=self.load_video, **side_btn_style).pack(fill="x", pady=2)

        # group: Threshold & Options
        options_frame = tk.LabelFrame(self.sidebar, text="Settings", bg="#2d2d2d", fg="#aaaaaa", padx=10, pady=10)
        options_frame.pack(fill="x", pady=10)

        tk.Label(options_frame, text="Confidence Threshold", bg="#2d2d2d", fg="#ffffff", font=("Helvetica", 9)).pack(anchor="w")
        tk.Scale(options_frame, variable=self.conf_threshold, from_=0, to=1, resolution=0.05, orient="horizontal", bg="#2d2d2d", fg="white", highlightthickness=0, troughcolor="#1e1e1e").pack(fill="x", pady=(0, 10))

        tk.Checkbutton(options_frame, text="Show Detections", variable=self.show_overlay, bg="#2d2d2d", fg="#ffffff", selectcolor="#1e1e1e", activebackground="#2d2d2d", activeforeground="white").pack(anchor="w")

        # Main Content Area
        self.content_area = tk.Frame(self.main_container, bg="#1e1e1e")
        self.content_area.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Title
        tk.Label(self.content_area, text="Phoenix Block Logic Visualizer (PyTorch)", font=("Helvetica", 18, "bold"), bg="#1e1e1e", fg="#3498db").pack(pady=(0, 10))

        # Image Display Area
        self.image_frame = tk.Frame(self.content_area, bg="#121212", bd=0)
        self.image_frame.pack(pady=10, fill="both", expand=True)
        
        self.canvas_size = (700, 500)
        self.canvas = tk.Canvas(self.image_frame, width=self.canvas_size[0], height=self.canvas_size[1], bg="#121212", highlightthickness=0)
        self.canvas.pack(expand=True)

        # Playback Controls (Bottom of Center)
        playback_container = tk.Frame(self.content_area, bg="#1e1e1e")
        playback_container.pack(fill="x", pady=10)

        # Scrub Bar
        self.scrub_var = tk.DoubleVar()
        self.scrub_bar = tk.Scale(playback_container, variable=self.scrub_var, from_=0, to=100, orient="horizontal", 
                                bg="#1e1e1e", fg="#aaaaaa", highlightthickness=0, troughcolor="#2d2d2d", 
                                showvalue=False, command=self.on_scrub)
        self.scrub_bar.pack(fill="x", padx=10)

        btn_row = tk.Frame(playback_container, bg="#1e1e1e")
        btn_row.pack(pady=10)

        pb_btn_style = {"font": ("Helvetica", 12), "width": 5, "bg": "#2d2d2d", "fg": "white", "bd": 0, "cursor": "hand2"}
        
        self.btn_prev = tk.Button(btn_row, text="⏮", command=self.prev_frame, **pb_btn_style)
        self.btn_prev.grid(row=0, column=0, padx=5)

        self.btn_play = tk.Button(btn_row, text="▶", width=8, bg="#3498db", fg="white", font=("Helvetica", 12, "bold"), bd=0, command=self.toggle_playback)
        self.btn_play.grid(row=0, column=1, padx=5)

        self.btn_next = tk.Button(btn_row, text="⏭", command=self.next_frame, **pb_btn_style)
        self.btn_next.grid(row=0, column=2, padx=5)

        tk.Button(btn_row, text="⏹ Stop", command=self.stop_video, **pb_btn_style).grid(row=0, column=3, padx=20)

        # Results Label
        self.result_label = tk.Label(self.content_area, text="Ready", font=("Helvetica", 14), bg="#1e1e1e", fg="#999")
        self.result_label.pack(pady=5)

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Model Status: Loaded" if self.model else "Model Status: Not Loaded", 
                                 bd=1, relief="sunken", anchor="w", bg="#2d2d2d", fg="#888", font=("Helvetica", 8))
        self.status_bar.pack(side="bottom", fill="x")

    def process_image(self, file_path):
        try:
            original_image = Image.open(file_path).convert('RGB')
            self.predict_and_visualize(original_image)
        except Exception as e:
            print(f"Error processing image: {e}")
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def predict_and_visualize(self, original_image):
        try:
            # 1. Load and Preprocess for Model
            img_resized = original_image.resize(IMAGE_SIZE)
            img_array = np.array(img_resized, dtype=np.float32)
            # Normalize
            img_array = img_array / 255.0
            
            # PyTorch: (H, W, C) -> (C, H, W)
            # Add batch dimension: (1, C, H, W)
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

            display_image = original_image.copy()
            
            # Smart Scaling to fit canvas
            img_w, img_h = display_image.size
            canvas_w, canvas_h = self.canvas_size
            
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            display_image = display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            label_text = "Background"
            confidence = 0.0
            predicted_class_idx = 0
            inference_time = None

            # 2. Inference
            if self.model:
                start_time = time.time()
                with torch.no_grad():
                    class_out, box_out = self.model(img_tensor)
                    
                    # Apply Softmax for class probabilities
                    class_probs = torch.softmax(class_out, dim=1)
                    
                    # Get max confidence
                    max_conf, max_idx = torch.max(class_probs, 1)
                    confidence = max_conf.item()
                    predicted_class_idx = max_idx.item()
                    
                    # Get Box
                    box_norm = box_out[0].cpu().numpy()
                
                inference_time = (time.time() - start_time) * 1000

                if confidence >= self.conf_threshold.get() and predicted_class_idx > 0:
                    label_text = CLASS_MAP.get(predicted_class_idx, "Unknown")
                    
                    if self.show_overlay.get():
                        draw = ImageDraw.Draw(display_image)
                        
                        ymin, xmin, ymax, xmax = box_norm
                        
                        # Scale back to display image size
                        # Box is relative [0,1]
                        left = xmin * new_w
                        right = xmax * new_w
                        top = ymin * new_h
                        bottom = ymax * new_h
                        
                        color = COLORS.get(predicted_class_idx, 'white')
                        
                        draw.rectangle([left, top, right, bottom], outline=color, width=3)
                        
                        # Draw label background
                        try:
                            font = ImageFont.truetype("arial.ttf", 16)
                        except IOError:
                            font = ImageFont.load_default()
                            
                        text_w = font.getlength(f"{label_text} {confidence:.0%}")
                        text_h = 16
                        
                        draw.rectangle([left, top - text_h - 4, left + text_w + 4, top], fill=color)
                        draw.text((left + 2, top - text_h - 2), f"{label_text} {confidence:.0%}", fill='black', font=font)

            self.update_display(display_image, label_text, confidence, predicted_class_idx, inference_time)
                
        except Exception as e:
            print(f"Error in prediction/visualization: {e}")
            import traceback
            traceback.print_exc()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video or Data files", "*.mp4 *.avi *.mov *.mkv *.npz")])
        if file_path:
            self.stop_video()
            
            if file_path.lower().endswith('.npz'):
                try:
                    with np.load(file_path, allow_pickle=True) as data:
                        if 'images' in data:
                            self.frames = data['images'] # (N, H, W, 3)
                            self.video_mode = 'npz'
                            self.current_frame_idx = 0
                            self.scrub_bar.config(to=len(self.frames)-1)
                            self.status_bar.config(text=f"Loaded NPZ: {len(self.frames)} frames")
                            self.show_current_frame()
                        else:
                            messagebox.showerror("Error", "NPZ does not contain 'images' key")
                except Exception as e:
                     messagebox.showerror("Error", f"Failed to load NPZ: {e}")
            else:
                self.cap = cv2.VideoCapture(file_path)
                if self.cap.isOpened():
                    self.video_mode = 'video'
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.scrub_bar.config(to=total_frames)
                    self.status_bar.config(text=f"Loaded Video: {total_frames} frames")
                    self.show_current_frame()

    def toggle_playback(self):
        if not self.video_running:
            self.video_running = True
            self.is_paused = False
            self.play_video()
            return
        
        self.is_paused = not self.is_paused
        self.btn_play.config(text="▶" if self.is_paused else "⏸")
        if not self.is_paused:
            self.play_video()

    def next_frame(self):
        if not self.video_running and self.video_mode:
             # Allow stepping even if not running usually, but let's assume we need to start it first or just update pointer
             self.video_running = True 
             self.is_paused = True

        self.is_paused = True
        self.btn_play.config(text="▶")
        
        if self.video_mode == 'video' and self.cap:
            curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, curr + 1)
            self.show_current_frame()
        elif self.video_mode == 'npz':
            self.current_frame_idx = min(len(self.frames)-1, self.current_frame_idx + 1)
            self.show_current_frame()

    def prev_frame(self):
        if not self.video_running and self.video_mode:
             self.video_running = True
             self.is_paused = True

        self.is_paused = True
        self.btn_play.config(text="▶")
        
        if self.video_mode == 'video' and self.cap:
            curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr - 2)) 
            self.show_current_frame()
        elif self.video_mode == 'npz':
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            self.show_current_frame()

    def show_current_frame(self):
        """Displays the frame at the current pointer without starting loop."""
        if self.video_mode == 'video' and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # CV2 is BGR, PIL is RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.predict_and_visualize(img)
                
                curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.scrub_var.set(curr)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, curr) # Rewind the read
        elif self.video_mode == 'npz' and self.frames is not None:
            frame_data = self.frames[self.current_frame_idx]
            
            # De-normalize if needed (assuming stored as 0-1 float or 0-255 uint8)
            if frame_data.dtype == np.float32 or (frame_data.max() <= 1.05 and frame_data.min() >= -0.05):
                frame_data = (frame_data * 255).astype(np.uint8)
            else:
                frame_data = frame_data.astype(np.uint8)
            
            # NPZ usually stores as RGB if saved from PIL
            img = Image.fromarray(frame_data)
            # Resize if needed to match original aspect, but dataset might be resized already
            # processed_data.npz stores resized images usually (213, 100)
            
            self.predict_and_visualize(img)
            self.scrub_var.set(self.current_frame_idx)


    def on_scrub(self, value):
        if not self.video_running and self.video_mode:
            self.video_running = True
            self.is_paused = True
            
        val = int(float(value))
        
        if self.video_mode == 'video' and self.cap:
             self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
             self.show_current_frame()
        elif self.video_mode == 'npz':
             self.current_frame_idx = min(len(self.frames)-1, max(0, val))
             self.show_current_frame()
        
        if self.is_paused:
             self.btn_play.config(text="▶")

    def play_video(self):
        if not self.video_running or self.is_paused:
            return

        if self.video_mode == 'video' and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.predict_and_visualize(img)
                
                curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.scrub_var.set(curr)
                
                self.root.after(self.playback_speed.get(), self.play_video)
            else:
                self.stop_video()
                
        elif self.video_mode == 'npz' and self.frames is not None:
            if self.current_frame_idx < len(self.frames):
                self.show_current_frame()
                self.current_frame_idx += 1
                self.root.after(self.playback_speed.get(), self.play_video)
            else:
                self.stop_video()

    def stop_video(self):
        self.video_running = False
        self.is_paused = False
        self.video_mode = None
        self.frames = None
        self.btn_play.config(text="▶")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_bar.config(text="Status: Stopped")

    def update_display(self, image, label, conf, class_idx, inference_time=None):
        self.tk_image = ImageTk.PhotoImage(image)
        
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        img_w = self.tk_image.width()
        img_h = self.tk_image.height()
        
        x_pos = (c_w - img_w) // 2
        y_pos = (c_h - img_h) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self.tk_image)
        
        color = COLORS.get(class_idx, '#ffffff')
        perf_text = ""
        if inference_time is not None:
            perf_text = f" | {inference_time:.1f}ms"
            
        self.result_label.config(text=f"{label} ({conf:.1%}){perf_text}", fg=color)

    def load_random_image(self):
        self.stop_video()
        if not self.all_images:
            messagebox.showinfo("Info", "No images found in dataset.")
            return
        
        file_path = random.choice(self.all_images)
        self.status_bar.config(text=f"Loaded Random: {os.path.basename(file_path)}")
        self.process_image(file_path)

    def load_manual_image(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
             self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
             self.process_image(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
