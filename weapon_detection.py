import cv2
import numpy as np
import os
import threading
import time
import datetime
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('weapon_detection.log'), logging.StreamHandler()]
)
logger = logging.getLogger('WeaponDetection')

class WeaponDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Weapon Detection System")
        self.root.geometry("1200x700")
        self.root.minsize(900, 600)
        
        # Set app theme
        ctk.set_appearance_mode("System")  # Use system theme (dark/light)
        ctk.set_default_color_theme("blue")
        
        # Variables
        self.video_source = None
        self.cap = None
        self.detection_active = False
        self.is_paused = False
        self.detection_thread = None
        self.current_frame = None
        self.detection_sensitivity = tk.DoubleVar(value=0.5)
        self.show_confidence = tk.BooleanVar(value=True)
        self.video_path = tk.StringVar()
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self.get_available_cameras()
        self.detection_history = []
        self.model_loaded = False
        self.net = None
        self.classes = ["Weapon"]
        self.status_text = tk.StringVar(value="Ready")
        self.model_path = tk.StringVar(value="./models")
        self.alert_sound_enabled = tk.BooleanVar(value=True)
        self.recording = False
        self.out = None
        self.record_path = None
        self.auto_record = tk.BooleanVar(value=False)
        
        # Default paths
        self.config_file = "config.json"
        self.load_config()
        
        # Create UI
        self.create_ui()
        
        # Attempt to load model
        self.load_model()
        
        # Protocol handler for closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_ui(self):
        # Create main frames
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for video display
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create bottom frame for controls
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create tab view for settings
        self.tab_view = ctk.CTkTabview(self.control_frame)
        self.tab_view.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.source_tab = self.tab_view.add("Source")
        self.settings_tab = self.tab_view.add("Settings")
        self.detection_tab = self.tab_view.add("Detection")
        self.history_tab = self.tab_view.add("History")
        
        # Source tab
        self.create_source_tab()
        
        # Settings tab
        self.create_settings_tab()
        
        # Detection tab
        self.create_detection_tab()
        
        # History tab
        self.create_history_tab()
        
        # Status bar
        self.create_status_bar()

    def create_source_tab(self):
        # Source section
        source_frame = ctk.CTkFrame(self.source_tab)
        source_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Webcam selection
        camera_frame = ctk.CTkFrame(source_frame)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(camera_frame, text="Camera Source:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for camera selection if available
        if self.available_cameras:
            camera_options = [f"Camera {idx}" for idx in self.available_cameras]
            self.camera_dropdown = ctk.CTkComboBox(
                camera_frame, 
                values=camera_options,
                command=self.on_camera_selected
            )
            self.camera_dropdown.pack(side=tk.LEFT, padx=5)
            self.camera_dropdown.set(camera_options[0])
        else:
            ctk.CTkLabel(camera_frame, text="No cameras detected").pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(camera_frame, text="Use Webcam", command=self.use_webcam).pack(side=tk.LEFT, padx=5)
        
        # File selection
        file_frame = ctk.CTkFrame(source_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(file_frame, text="Video File:").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(file_frame, textvariable=self.video_path, width=300).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(file_frame, text="Use File", command=self.use_file).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_buttons_frame = ctk.CTkFrame(source_frame)
        control_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ctk.CTkButton(
            control_buttons_frame, 
            text="Start Detection", 
            command=self.toggle_detection,
            fg_color="green"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ctk.CTkButton(
            control_buttons_frame, 
            text="Pause", 
            command=self.toggle_pause,
            state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.record_button = ctk.CTkButton(
            control_buttons_frame, 
            text="Start Recording", 
            command=self.toggle_recording,
            state=tk.DISABLED
        )
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            control_buttons_frame, 
            text="Take Screenshot", 
            command=self.take_screenshot
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            control_buttons_frame, 
            text="Exit", 
            command=self.on_close,
            fg_color="darkred"
        ).pack(side=tk.RIGHT, padx=5)

    def create_settings_tab(self):
        # Settings section
        settings_frame = ctk.CTkFrame(self.settings_tab)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model settings
        model_frame = ctk.CTkFrame(settings_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(model_frame, text="Model Directory:").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(model_frame, textvariable=self.model_path, width=300).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(model_frame, text="Browse", command=self.browse_model_dir).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Detection settings
        detection_settings_frame = ctk.CTkFrame(settings_frame)
        detection_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(detection_settings_frame, text="Detection Sensitivity:").pack(side=tk.LEFT, padx=5)
        sensitivity_slider = ctk.CTkSlider(
            detection_settings_frame,
            from_=0.1,
            to=0.9,
            variable=self.detection_sensitivity,
            width=200
        )
        sensitivity_slider.pack(side=tk.LEFT, padx=5)
        sensitivity_value = ctk.CTkLabel(detection_settings_frame, text="0.5")
        sensitivity_value.pack(side=tk.LEFT, padx=5)
        
        # Update sensitivity value label when slider changes
        def update_sensitivity_label(value):
            sensitivity_value.configure(text=f"{float(value):.1f}")
        
        sensitivity_slider.configure(command=update_sensitivity_label)
        
        # Display settings
        display_frame = ctk.CTkFrame(settings_frame)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkCheckBox(
            display_frame, 
            text="Show Confidence Values", 
            variable=self.show_confidence
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkCheckBox(
            display_frame, 
            text="Enable Alert Sound", 
            variable=self.alert_sound_enabled
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkCheckBox(
            display_frame, 
            text="Auto Record on Detection", 
            variable=self.auto_record
        ).pack(side=tk.LEFT, padx=5)
        
        # Save/load config
        config_frame = ctk.CTkFrame(settings_frame)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkButton(config_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(config_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)

    def create_detection_tab(self):
        # Detection info section
        detection_frame = ctk.CTkFrame(self.detection_tab)
        detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a treeview for detections
        columns = ("timestamp", "object", "confidence")
        self.detection_tree = ttk.Treeview(detection_frame, columns=columns, show="headings")
        
        # Define headings
        self.detection_tree.heading("timestamp", text="Timestamp")
        self.detection_tree.heading("object", text="Object")
        self.detection_tree.heading("confidence", text="Confidence")
        
        # Define columns
        self.detection_tree.column("timestamp", width=150)
        self.detection_tree.column("object", width=100)
        self.detection_tree.column("confidence", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(detection_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscroll=scrollbar.set)
        
        # Pack
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Button to clear detection history
        ctk.CTkButton(
            detection_frame,
            text="Clear Detection History",
            command=self.clear_detection_history
        ).pack(side=tk.BOTTOM, padx=5, pady=5)

    def create_history_tab(self):
        # History section
        history_frame = ctk.CTkFrame(self.history_tab)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create list for saved videos/screenshots
        columns = ("filename", "type", "date", "size")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        
        # Define headings
        self.history_tree.heading("filename", text="Filename")
        self.history_tree.heading("type", text="Type")
        self.history_tree.heading("date", text="Date")
        self.history_tree.heading("size", text="Size")
        
        # Define columns
        self.history_tree.column("filename", width=200)
        self.history_tree.column("type", width=100)
        self.history_tree.column("date", width=150)
        self.history_tree.column("size", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # Pack
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ctk.CTkFrame(history_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkButton(
            button_frame,
            text="Open Selected File",
            command=self.open_selected_file
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Open Output Folder",
            command=self.open_output_folder
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Refresh List",
            command=self.refresh_history_list
        ).pack(side=tk.LEFT, padx=5)
        
        # Initial load of history
        self.refresh_history_list()

    def create_status_bar(self):
        # Status bar
        status_frame = ctk.CTkFrame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Status message on the left
        self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_text)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Model status indicator
        self.model_status = ctk.CTkLabel(
            status_frame,
            text="Model: Not Loaded",
            text_color="red"
        )
        self.model_status.pack(side=tk.RIGHT, padx=5)
        
        # FPS counter
        self.fps_label = ctk.CTkLabel(status_frame, text="FPS: --")
        self.fps_label.pack(side=tk.RIGHT, padx=5)

    def get_available_cameras(self):
        """Get list of available camera indices"""
        available = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def on_camera_selected(self, selection):
        """Handle camera selection from dropdown"""
        if selection:
            try:
                # Extract camera index from selection string
                camera_idx = int(selection.split(" ")[1])
                self.camera_index.set(camera_idx)
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing camera selection: {e}")

    def browse_video(self):
        """Open file dialog to select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            self.video_path.set(file_path)

    def browse_model_dir(self):
        """Open directory dialog to select model directory"""
        dir_path = filedialog.askdirectory(
            title="Select Model Directory"
        )
        
        if dir_path:
            self.model_path.set(dir_path)

    def use_webcam(self):
        """Set video source to webcam"""
        if not self.available_cameras:
            messagebox.showerror("Error", "No webcam detected.")
            return
            
        self.video_source = "webcam"
        camera_idx = self.camera_index.get()
        
        # Close existing capture if any
        self.close_video_source()
        
        try:
            self.cap = cv2.VideoCapture(camera_idx)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera {camera_idx}")
                self.video_source = None
                self.status_text.set("Failed to open webcam")
                return
                
            self.status_text.set(f"Using webcam {camera_idx}")
        except Exception as e:
            logger.error(f"Error opening webcam: {e}")
            messagebox.showerror("Error", f"Error opening webcam: {e}")
            self.video_source = None

    def use_file(self):
        """Set video source to file"""
        file_path = self.video_path.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
            
        self.video_source = "file"
        
        # Close existing capture if any
        self.close_video_source()
        
        try:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open file: {file_path}")
                self.video_source = None
                self.status_text.set("Failed to open file")
                return
                
            self.status_text.set(f"Using file: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error opening file: {e}")
            messagebox.showerror("Error", f"Error opening file: {e}")
            self.video_source = None

    def close_video_source(self):
        """Close current video source if open"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def load_model(self):
        """Load the YOLO model"""
        try:
            model_dir = self.model_path.get()
            weights_path = os.path.join(model_dir, "yolov4.weights")
            config_path = os.path.join(model_dir, "yolov4.cfg")
            
            # Check if files exist
            if not os.path.exists(weights_path) or not os.path.exists(config_path):
                self.status_text.set("Model files not found. Please check paths.")
                self.model_status.configure(text="Model: Not Found", text_color="red")
                logger.warning(f"Model files not found at {model_dir}")
                return
            
            self.status_text.set("Loading YOLO model...")
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.output_layer_names = self.net.getUnconnectedOutLayersNames()
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            self.model_loaded = True
            self.model_status.configure(text="Model: Loaded", text_color="green")
            self.status_text.set("Model loaded successfully")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            self.model_status.configure(text="Model: Error", text_color="red")
            error_msg = f"Error loading model: {e}"
            self.status_text.set(error_msg)
            logger.error(error_msg)
            messagebox.showerror("Model Error", error_msg)

    def toggle_detection(self):
        """Start or stop the detection process"""
        if not self.detection_active:
            # Start detection
            if not self.model_loaded:
                messagebox.showerror("Error", "Model not loaded. Please load model first.")
                return
                
            if not self.cap or not self.cap.isOpened():
                messagebox.showerror("Error", "No video source selected. Please select webcam or video file.")
                return
                
            self.detection_active = True
            self.start_button.configure(text="Stop Detection", fg_color="red")
            self.pause_button.configure(state=tk.NORMAL)
            self.record_button.configure(state=tk.NORMAL)
            self.status_text.set("Detection started")
            
            # Start detection in a separate thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
        else:
            # Stop detection
            self.detection_active = False
            self.start_button.configure(text="Start Detection", fg_color="green")
            self.pause_button.configure(state=tk.DISABLED)
            self.record_button.configure(state=tk.DISABLED)
            self.status_text.set("Detection stopped")
            
            # Stop recording if active
            if self.recording:
                self.toggle_recording()

    def toggle_pause(self):
        """Pause or resume the detection"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_button.configure(text="Resume")
            self.status_text.set("Detection paused")
        else:
            self.pause_button.configure(text="Pause")
            self.status_text.set("Detection resumed")

    def toggle_recording(self):
        """Start or stop recording"""
        if not self.recording:
            # Start recording
            if not self.cap or not self.current_frame is None:
                return
                
            # Create output directory if it doesn't exist
            output_dir = "recordings"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.mp4"
            self.record_path = os.path.join(output_dir, filename)
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:  # Default to 30fps if unable to get FPS
                fps = 30
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.record_path, fourcc, fps, (width, height))
            
            self.recording = True
            self.record_button.configure(text="Stop Recording", fg_color="red")
            self.status_text.set(f"Recording to {filename}")
        else:
            # Stop recording
            if self.out is not None:
                self.out.release()
                self.out = None
            
            self.recording = False
            self.record_button.configure(text="Start Recording", fg_color=None)
            self.status_text.set("Recording stopped")
            
            # Refresh history list to show new recording
            self.refresh_history_list()

    def take_screenshot(self):
        """Capture current frame as screenshot"""
        if self.current_frame is None:
            messagebox.showinfo("Info", "No frame available to capture")
            return
        
        # Create output directory if it doesn't exist
        output_dir = "screenshots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, self.current_frame)
        
        self.status_text.set(f"Screenshot saved to {filename}")
        logger.info(f"Screenshot saved to {filepath}")
        
        # Refresh history list to show new screenshot
        self.refresh_history_list()

    def detection_loop(self):
        """Main detection loop running in separate thread"""
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while self.detection_active:
            if self.is_paused:
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
            
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                # Handle end of video or read errors
                if not ret:
                    if self.video_source == "file":
                        # End of file, restart video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # Webcam error
                        logger.error("Error reading from webcam")
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to get frame from camera"))
                        self.root.after(0, self.toggle_detection)
                        break
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Record if active
                if self.recording and self.out is not None:
                    self.out.write(frame)
                
                # Process frame (weapon detection)
                processed_frame, detections = self.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - prev_time
                
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = current_time
                    # Update FPS label
                    self.root.after(0, lambda f=fps: self.fps_label.configure(text=f"FPS: {f:.1f}"))
                
                # Check if detection found any weapons
                if detections:
                    for detection in detections:
                        # Add to detection history
                        self.add_detection_to_history(detection)
                        
                        # Auto record if enabled and not already recording
                        if self.auto_record.get() and not self.recording:
                            self.root.after(0, self.toggle_recording)
                
                # Convert frame to RGB for tkinter
                cv_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Resize image to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Maintain aspect ratio
                    h, w = cv_img.shape[:2]
                    aspect_ratio = w / h
                    
                    if canvas_width / canvas_height > aspect_ratio:
                        # Canvas is wider than image aspect ratio
                        new_height = canvas_height
                        new_width = int(new_height * aspect_ratio)
                    else:
                        # Canvas is taller than image aspect ratio
                        new_width = canvas_width
                        new_height = int(new_width / aspect_ratio)
                    
                    cv_img = cv2.resize(cv_img, (new_width, new_height))
                
                # Convert to PIL Image format
                pil_img = Image.fromarray(cv_img)
                
                # Convert PIL image to ImageTk format
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # Update canvas
                self.root.after(0, lambda img=img_tk: self.update_canvas(img))
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                self.status_text.set(f"Error: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error

    def update_canvas(self, img):
        """Update canvas with new image"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Clear previous image
        self.canvas.delete("all")
        
        # Keep reference to prevent garbage collection
        self.photo_image = img
        
        # Calculate position to center image
        x = (canvas_width - img.width()) // 2
        y = (canvas_height - img.height()) // 2
        
        # Add image to canvas
        self.canvas.create_image(x, y, anchor=tk.NW, image=img)

    def process_frame(self, frame):
        """Process a frame and detect weapons"""
        height, width, _ = frame.shape
        detections = []
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layer_names)
        
        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        
        # Sensitivity threshold from settings
        threshold = self.detection_sensitivity.get()
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > threshold:
                    # Object detected
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        detected_weapons = []
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 0, 255)  # Red color for weapon detection
                
                # Create detection record
                detection_info = {
                    'label': label,
                    'confidence': confidence,
                    'timestamp': datetime.datetime.now()
                }
                detected_weapons.append(detection_info)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Show confidence if enabled
                if self.show_confidence.get():
                    confidence_text = f"{label}: {confidence:.2f}"
                else:
                    confidence_text = label
                    
                cv2.putText(frame, confidence_text, (x, y - 10), font, 0.5, color, 2)
                
                # Play alert sound if enabled
                if self.alert_sound_enabled.get():
                    # Use root's bell sound as a simple alert
                    # (In a production app, you might want to use a more explicit sound)
                    self.root.after(0, self.root.bell)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, height - 10), font, 0.5, (255, 255, 255), 1)
        
        return frame, detected_weapons

    def add_detection_to_history(self, detection):
        """Add detection to history and update UI"""
        self.detection_history.append(detection)
        
        # Update treeview
        timestamp_str = detection['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        confidence_str = f"{detection['confidence']:.2f}"
        
        # Schedule UI update to run in main thread
        self.root.after(0, lambda: self.detection_tree.insert("", 0, values=(timestamp_str, detection['label'], confidence_str)))

    def clear_detection_history(self):
        """Clear detection history and treeview"""
        self.detection_history = []
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)

    def refresh_history_list(self):
        """Refresh the list of saved files in history tab"""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Get screenshots
        screenshot_dir = "screenshots"
        if os.path.exists(screenshot_dir):
            for file in os.listdir(screenshot_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(screenshot_dir, file)
                    self.add_file_to_history(file_path, "Screenshot")
        
        # Get recordings
        recording_dir = "recordings"
        if os.path.exists(recording_dir):
            for file in os.listdir(recording_dir):
                if file.lower().endswith(('.mp4', '.avi')):
                    file_path = os.path.join(recording_dir, file)
                    self.add_file_to_history(file_path, "Recording")

    def add_file_to_history(self, file_path, file_type):
        """Add file to history treeview"""
        try:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_size_str = self.format_file_size(file_size)
            
            # Get file creation/modification time
            timestamp = os.path.getmtime(file_path)
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to treeview
            self.history_tree.insert("", "end", values=(file_name, file_type, date_str, file_size_str))
        except Exception as e:
            logger.error(f"Error adding file to history: {e}")

    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        # Convert bytes to KB, MB, etc.
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def open_selected_file(self):
        """Open selected file from history"""
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "No file selected")
            return
            
        item = self.history_tree.item(selected[0])
        file_name = item['values'][0]
        file_type = item['values'][1]
        
        # Determine directory based on file type
        if file_type == "Screenshot":
            directory = "screenshots"
        else:  # Recording
            directory = "recordings"
            
        file_path = os.path.join(directory, file_name)
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
            
        # Open file with default application
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS, Linux
                subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', file_path))
        except Exception as e:
            logger.error(f"Error opening file: {e}")
            messagebox.showerror("Error", f"Could not open file: {e}")

    def open_output_folder(self):
        """Open output folder in file explorer"""
        folders = ["screenshots", "recordings"]
        
        for folder in folders:
            if os.path.exists(folder):
                try:
                    import subprocess
                    if os.name == 'nt':  # Windows
                        os.startfile(folder)
                    elif os.name == 'posix':  # macOS, Linux
                        subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', folder))
                    return
                except Exception as e:
                    logger.error(f"Error opening folder: {e}")
                    messagebox.showerror("Error", f"Could not open folder: {e}")
                
        # If no folders exist
        messagebox.showinfo("Info", "No output folders found")

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                    # Load settings
                    if 'detection_sensitivity' in config:
                        self.detection_sensitivity.set(config['detection_sensitivity'])
                    if 'show_confidence' in config:
                        self.show_confidence.set(config['show_confidence'])
                    if 'alert_sound_enabled' in config:
                        self.alert_sound_enabled.set(config['alert_sound_enabled'])
                    if 'auto_record' in config:
                        self.auto_record.set(config['auto_record'])
                    if 'model_path' in config:
                        self.model_path.set(config['model_path'])
                        
                    logger.info("Configuration loaded")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default values if config can't be loaded

    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'detection_sensitivity': self.detection_sensitivity.get(),
                'show_confidence': self.show_confidence.get(),
                'alert_sound_enabled': self.alert_sound_enabled.get(),
                'auto_record': self.auto_record.get(),
                'model_path': self.model_path.get()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            self.status_text.set("Configuration saved")
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            messagebox.showerror("Error", f"Could not save configuration: {e}")

    def reset_defaults(self):
        """Reset settings to default values"""
        if messagebox.askyesno("Reset Defaults", "Are you sure you want to reset all settings to default values?"):
            self.detection_sensitivity.set(0.5)
            self.show_confidence.set(True)
            self.alert_sound_enabled.set(True)
            self.auto_record.set(False)
            self.model_path.set("./models")
            
            self.status_text.set("Settings reset to defaults")
            logger.info("Settings reset to defaults")

    def on_close(self):
        """Clean up resources and close application"""
        # Stop detection if running
        if self.detection_active:
            self.detection_active = False
            
            # Allow some time for thread to terminate
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)
        
        # Stop recording if active
        if self.recording and self.out is not None:
            self.out.release()
        
        # Close video source
        self.close_video_source()
        
        # Save configuration
        self.save_config()
        
        # Destroy root window
        self.root.destroy()

def main():
    # Configure logging folder
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create directories for outputs
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create root window
    root = ctk.CTk()
    app = WeaponDetectionApp(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    try:
        import customtkinter as ctk
    except ImportError:
        print("CustomTkinter is required for this application.")
        print("Installing CustomTkinter...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "customtkinter"])
            import customtkinter as ctk
            print("CustomTkinter installed successfully.")
        except Exception as e:
            print(f"Error installing CustomTkinter: {e}")
            print("Please install manually with: pip install customtkinter")
            exit(1)
    
    main()