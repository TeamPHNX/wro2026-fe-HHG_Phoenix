import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import os
import random
import glob
import cv2

# Configuration
MODEL_PATH = 'models/new_best_model.keras'
DATASET_PATH = 'testing_data'
IMAGE_SIZE = (224, 224)

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

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phoenix Block Detection - Advanced Viewer")
        self.root.geometry("1000x850")
        self.root.configure(bg="#1e1e1e") # Dark theme

        # Load Model
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
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
        tk.Label(self.content_area, text="Phoenix Block Logic Visualizer", font=("Helvetica", 18, "bold"), bg="#1e1e1e", fg="#3498db").pack(pady=(0, 10))

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
        self.status_bar = tk.Label(self.root, text="Model Status: Loaded" if self.model else "Model Status: Error", 
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
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            img_batch = np.expand_dims(img_array, axis=0)

            display_image = original_image.copy()
            
            # Smart Scaling to fit canvas (Upscale or Downscale)
            img_w, img_h = display_image.size
            canvas_w, canvas_h = self.canvas_size
            
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            display_image = display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            label_text = "Background"
            confidence = 0.0
            predicted_class_idx = 0

            # 2. Inference
            if self.model:
                predictions = self.model.predict(img_batch, verbose=0)
                class_probs = predictions[0][0]
                box_pred = predictions[1][0]
                
                predicted_class_idx = np.argmax(class_probs)
                confidence = np.max(class_probs)
                
                # 3. Visualization
                if self.show_overlay.get() and confidence >= self.conf_threshold.get():
                    draw = ImageDraw.Draw(display_image)
                    w, h = display_image.size
                    
                    ymin, xmin, ymax, xmax = box_pred
                    
                    left = max(0, min(xmin, 1)) * w
                    right = max(0, min(xmax, 1)) * w
                    top = max(0, min(ymin, 1)) * h
                    bottom = max(0, min(ymax, 1)) * h
                    
                    color = COLORS.get(predicted_class_idx, 'blue')
                    label_text = CLASS_MAP.get(predicted_class_idx, 'Unknown')
                    
                    # Draw Box
                    if predicted_class_idx != 0:
                        # Better box: semi-transparent or just thick
                        draw.rectangle([left, top, right, bottom], outline=color, width=3)
                        
                        try:
                            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
                        except:
                            font = ImageFont.load_default()
                            
                        text = f"{label_text} {confidence:.1%}"
                        
                        try:
                            text_bbox = draw.textbbox((left, top), text, font=font)
                            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                            draw.text((left, top-2), text, fill="white", font=font)
                        except:
                            draw.text((left + 2, top + 2), text, fill=color, font=font)
                
                if confidence < self.conf_threshold.get():
                    label_text = "Background (below threshold)"

            self.update_display(display_image, label_text, confidence, predicted_class_idx)
                
        except Exception as e:
            print(f"Error in prediction/visualization: {e}")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video or Data files", "*.mp4 *.avi *.mov *.mkv *.npz")])
        if file_path:
            self.stop_video()
            
            if file_path.lower().endswith('.npz'):
                try:
                    data = np.load(file_path)
                    if 'images' in data:
                        self.frames = data['images']
                        self.video_mode = 'npz'
                        self.current_frame_idx = 0
                        self.scrub_bar.config(from_=0, to=len(self.frames)-1)
                        self.video_running = True
                        self.is_paused = False
                        self.btn_play.config(text="⏸")
                        self.play_video()
                        self.status_bar.config(text=f"Loaded NPZ: {os.path.basename(file_path)} ({len(self.frames)} frames)")
                    else:
                        messagebox.showerror("Error", "NPZ file does not contain 'images' key.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load NPZ file: {e}")
            else:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video file.")
                    return
                
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    self.scrub_bar.config(from_=0, to=total_frames-1)
                
                self.video_mode = 'video'
                self.video_running = True
                self.is_paused = False
                self.btn_play.config(text="⏸")
                self.play_video()
                self.status_bar.config(text=f"Loaded Video: {os.path.basename(file_path)}")

    def toggle_playback(self):
        if not self.video_running:
            return
        
        self.is_paused = not self.is_paused
        self.btn_play.config(text="▶" if self.is_paused else "⏸")
        if not self.is_paused:
            self.play_video()

    def next_frame(self):
        if not self.video_running: return
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
        if not self.video_running: return
        self.is_paused = True
        self.btn_play.config(text="▶")
        
        if self.video_mode == 'video' and self.cap:
            curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr - 2)) # -2 because read() advances one
            self.show_current_frame()
        elif self.video_mode == 'npz':
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            self.show_current_frame()

    def show_current_frame(self):
        """Displays the frame at the current pointer without starting loop."""
        if self.video_mode == 'video' and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.predict_and_visualize(Image.fromarray(frame_rgb))
                self.scrub_bar.set(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        elif self.video_mode == 'npz' and self.frames is not None:
            frame_data = self.frames[self.current_frame_idx]
            if frame_data.dtype == np.float32 or (frame_data.max() <= 1.05 and frame_data.min() >= -1.05):
                frame_data = ((frame_data + 1) / 2.0 * 255).astype(np.uint8)
            else:
                frame_data = frame_data.astype(np.uint8)
            
            # Restore original aspect ratio for NPZ data (213x100)
            img = Image.fromarray(frame_data)
            img = img.resize((213, 100))
            
            self.predict_and_visualize(img)
            self.scrub_bar.set(self.current_frame_idx)

    def on_scrub(self, value):
        if not self.video_running: return
        val = int(float(value))
        
        if self.video_mode == 'video' and self.cap:
             self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        elif self.video_mode == 'npz':
             self.current_frame_idx = val
        
        if self.is_paused:
            self.show_current_frame()

    def play_video(self):
        if not self.video_running or self.is_paused:
            return

        if self.video_mode == 'video' and self.cap and self.cap.isOpened():
            curr_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.scrub_bar.set(curr_pos)

            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.predict_and_visualize(Image.fromarray(frame_rgb))
                self.root.after(self.playback_speed.get(), self.play_video)
            else:
                self.stop_video()
                
        elif self.video_mode == 'npz' and self.frames is not None:
            if self.current_frame_idx < len(self.frames):
                self.scrub_bar.set(self.current_frame_idx)
                
                frame_data = self.frames[self.current_frame_idx]
                if frame_data.dtype == np.float32 or (frame_data.max() <= 1.05 and frame_data.min() >= -1.05):
                    frame_data = ((frame_data + 1) / 2.0 * 255).astype(np.uint8)
                else:
                    frame_data = frame_data.astype(np.uint8)
                
                # Restore original aspect ratio for NPZ data (213x100)
                img = Image.fromarray(frame_data)
                img = img.resize((213, 100))
                
                self.predict_and_visualize(img)
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

    def update_display(self, image, label, conf, class_idx):
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
        self.result_label.config(text=f"{label} ({conf:.1%})", fg=color)

    def load_random_image(self):
        self.stop_video()
        if not self.all_images:
            messagebox.showwarning("Warning", "No images found in dataset directory.")
            return
        
        file_path = random.choice(self.all_images)
        self.status_bar.config(text=f"Loaded Random: {os.path.basename(file_path)}")
        self.process_image(file_path)

    def load_manual_image(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.status_bar.config(text=f"Loaded File: {os.path.basename(file_path)}")
            self.process_image(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
