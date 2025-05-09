import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
import time
import os
import threading
import json
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ObjectDetectionApp:
    """
    Modern object detection application with webcam support, video playback,
    restricted area monitoring, and notification capabilities.
    """
    
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Smart Object Detection System")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # App configuration
        self.config_file = "app_config.json"
        self.default_config = {
            "pushbullet_api_key": "",  # Default empty, user should enter their key
            "model_path": "yolov8s.pt",
            "classes_file": "coco.txt",
            "log_file": "detection_log.csv",
            "restricted_area_file": "restricted_area.json",
            "notification_cooldown": 5,
            "log_cooldown": 5,
            "frame_skip_threshold": 3,
            "crowd_threshold": 10,
            "confidence_threshold": 0.5,
            "last_used_video": ""
        }
        self.config = self.load_config()
        
        # Object detection related variables
        self.model = None
        self.cap = None
        self.is_camera_on = False
        self.video_paused = False
        self.frame_count = 0
        self.restricted_area_pts = []
        self.restricted_area_enabled = len(self.restricted_area_pts) > 0
        self.show_outline = True
        self.current_crowd_count = 0
        self.last_notification_time = 0
        self.last_log_time = 0
        self.class_list = self.read_classes_from_file()
        self.pushbullet_client = None
        self.processing_thread = None
        self.stop_thread = False
        
        # Create the main UI
        self.create_ui()
        
        # Initialize the model
        self.load_model()
        
        # Load restricted area if available
        self.load_restricted_area()
        
        # Ensure csv log file exists with headers
        self.ensure_log_file()
        
        # Set up the notification system if key is provided
        self.setup_notification()

    def create_ui(self):
        """Create the user interface"""
        # Create style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Use a modern theme
        
        # Configure colors
        bg_color = "#f0f0f0"
        accent_color = "#3498db"
        self.root.configure(bg=bg_color)
        
        self.style.configure("Accent.TButton", background=accent_color, foreground="white")
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color)
        
        # Create main frames
        self.top_frame = ttk.Frame(self.root, padding="10")
        self.top_frame.pack(fill='x')
        
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill='x')
        
        self.advanced_frame = ttk.Frame(self.root, padding="10")
        self.advanced_frame.pack(fill='x')
        
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', side='bottom')
        
        # Create the canvas for video display
        self.canvas = tk.Canvas(self.canvas_frame, width=1020, height=500, bg="black")
        self.canvas.pack(fill='both', expand=True)
        
        # Settings section
        settings_label = ttk.Label(self.top_frame, text="Settings", font=("Arial", 12, "bold"))
        settings_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Class selection
        self.class_selection = tk.StringVar()
        self.class_selection.set("All")
        class_label = ttk.Label(self.top_frame, text="Filter Class:")
        class_label.grid(row=1, column=0, sticky="w", padx=(0, 5))
        self.class_combo = ttk.Combobox(self.top_frame, textvariable=self.class_selection, 
                                        values=["All"] + self.class_list, width=15)
        self.class_combo.grid(row=1, column=1, sticky="w", padx=(0, 10))
        
        # Crowd threshold
        threshold_label = ttk.Label(self.top_frame, text="Crowd Threshold:")
        threshold_label.grid(row=1, column=2, sticky="w", padx=(10, 5))
        self.crowd_threshold_var = tk.StringVar(value=str(self.config["crowd_threshold"]))
        self.crowd_threshold_entry = ttk.Entry(self.top_frame, textvariable=self.crowd_threshold_var, width=5)
        self.crowd_threshold_entry.grid(row=1, column=3, sticky="w")
        
        # Confidence threshold
        conf_label = ttk.Label(self.top_frame, text="Confidence:")
        conf_label.grid(row=1, column=4, sticky="w", padx=(10, 5))
        self.confidence_var = tk.StringVar(value=str(self.config["confidence_threshold"]))
        self.confidence_entry = ttk.Entry(self.top_frame, textvariable=self.confidence_var, width=5)
        self.confidence_entry.grid(row=1, column=5, sticky="w")
        
        threshold_button = ttk.Button(self.top_frame, text="Apply Settings", 
                                      command=self.apply_settings)
        threshold_button.grid(row=1, column=6, sticky="w", padx=10)
        
        # API Key input
        api_label = ttk.Label(self.top_frame, text="Pushbullet API Key:")
        api_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.api_key_var = tk.StringVar(value=self.config["pushbullet_api_key"])
        self.api_key_entry = ttk.Entry(self.top_frame, textvariable=self.api_key_var, width=40)
        self.api_key_entry.grid(row=2, column=1, columnspan=5, sticky="w", pady=(10, 0))
        api_button = ttk.Button(self.top_frame, text="Save API Key", 
                               command=self.save_api_key)
        api_button.grid(row=2, column=6, sticky="w", padx=10, pady=(10, 0))
        
        # Control buttons
        self.create_control_buttons()
        
        # Advanced buttons
        self.create_advanced_buttons()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to start detection")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        # Load initial splash image
        self.load_splash_image()

    def create_control_buttons(self):
        """Create the main control buttons"""
        # Create a notebook for tabbed controls
        control_notebook = ttk.Notebook(self.control_frame)
        control_notebook.pack(fill='x')
        
        # Video source tab
        source_tab = ttk.Frame(control_notebook)
        control_notebook.add(source_tab, text="Video Source")
        
        # Video control tab
        playback_tab = ttk.Frame(control_notebook)
        control_notebook.add(playback_tab, text="Playback Controls")
        
        # Restricted area tab
        area_tab = ttk.Frame(control_notebook)
        control_notebook.add(area_tab, text="Monitoring Zone")
        
        # Source buttons
        self.webcam_button = ttk.Button(source_tab, text="Start Webcam", 
                                       command=self.initialize_webcam)
        self.webcam_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(source_tab, text="Stop Camera/Video", 
                                     command=self.stop_video_source, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.file_button = ttk.Button(source_tab, text="Select Video File", 
                                     command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Last used video
        if self.config["last_used_video"] and os.path.exists(self.config["last_used_video"]):
            last_video_label = ttk.Label(source_tab, text="Recent:")
            last_video_label.pack(side=tk.LEFT, padx=(20, 5), pady=5)
            
            video_name = os.path.basename(self.config["last_used_video"])
            recent_button = ttk.Button(source_tab, text=video_name, 
                                      command=lambda: self.load_video(self.config["last_used_video"]))
            recent_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Playback controls
        self.pause_button = ttk.Button(playback_tab, text="Pause/Resume", 
                                      command=self.pause_resume_video, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.outline_button = ttk.Button(playback_tab, text="Toggle Outlines", 
                                        command=self.toggle_outline)
        self.outline_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add sliders for playback control if it's a video file
        self.playback_position = ttk.Scale(playback_tab, from_=0, to=100, orient=tk.HORIZONTAL)
        self.playback_position.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)
        self.playback_position.state(['disabled'])
        
        # Area controls
        create_area_button = ttk.Button(area_tab, text="Define Monitoring Zone", 
                                       command=self.define_restricted_area)
        create_area_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_area_button = ttk.Button(area_tab, text="Save Zone", 
                                         command=self.save_restricted_area)
        self.save_area_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        load_area_button = ttk.Button(area_tab, text="Load Zone", 
                                     command=self.load_restricted_area)
        load_area_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_area_button = ttk.Button(area_tab, text="Clear Zone", 
                                      command=self.clear_restricted_area)
        clear_area_button.pack(side=tk.LEFT, padx=5, pady=5)

    def create_advanced_buttons(self):
        """Create advanced control buttons"""
        # Statistics button
        stats_button = ttk.Button(self.advanced_frame, text="View Statistics", 
                                 command=self.show_statistics)
        stats_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Export logs button
        export_button = ttk.Button(self.advanced_frame, text="Export Logs", 
                                  command=self.export_logs)
        export_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Reset button
        reset_button = ttk.Button(self.advanced_frame, text="Reset All Settings", 
                                 command=self.reset_settings)
        reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Help button
        help_button = ttk.Button(self.advanced_frame, text="Help", 
                                command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Quit button
        quit_button = ttk.Button(self.advanced_frame, text="Quit", 
                                command=self.quit_app, style="Accent.TButton")
        quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def load_splash_image(self):
        """Load initial splash image"""
        try:
            # Try to use a default image if it exists
            splash_paths = ["splash.jpg", "1st.jpg", "splash.png"]
            splash_path = next((p for p in splash_paths if os.path.exists(p)), None)
            
            if splash_path:
                initial_image = Image.open(splash_path)
                # Resize to fit canvas
                initial_image = initial_image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()))
                initial_photo = ImageTk.PhotoImage(image=initial_image)
                self.canvas.img = initial_photo
                self.canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)
            else:
                # Create a default image with text
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text="Smart Object Detection System\nClick 'Start Webcam' or 'Select Video File' to begin", 
                    fill="white", 
                    font=("Arial", 24), 
                    justify=tk.CENTER
                )
        except Exception as e:
            logger.error(f"Error loading splash image: {e}")
            # Create a default image with text if no splash image
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="Smart Object Detection System\nClick 'Start Webcam' or 'Select Video File' to begin", 
                fill="white", 
                font=("Arial", 24), 
                justify=tk.CENTER
            )
        except Exception as e:
            logger.error(f"Error loading splash image: {e}")
            # Create default text
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="Smart Object Detection System", 
                fill="white", 
                font=("Arial", 24)
            )

    def load_config(self) -> dict:
        """Load application configuration from JSON file"""
        config = self.default_config.copy()
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    config.update(loaded_config)
                    logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # If loading fails, we'll use the default config
        return config
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            self.update_status(f"Error saving configuration: {e}")
    
    def apply_settings(self):
        """Apply the current settings from UI inputs"""
        try:
            # Update crowd threshold
            self.config["crowd_threshold"] = int(self.crowd_threshold_var.get())
            
            # Update confidence threshold
            self.config["confidence_threshold"] = float(self.confidence_var.get())
            
            # Save configuration
            self.save_config()
            self.update_status("Settings applied successfully")
        except ValueError as e:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for thresholds")
            logger.error(f"Invalid setting value: {e}")
    
    def save_api_key(self):
        """Save the API key from the entry field"""
        api_key = self.api_key_var.get().strip()
        self.config["pushbullet_api_key"] = api_key
        self.save_config()
        self.setup_notification()
        self.update_status("API key saved successfully")
    
    def setup_notification(self):
        """Set up the notification system if API key is available"""
        api_key = self.config["pushbullet_api_key"]
        if api_key:
            try:
                # Import pushbullet only when needed to make it optional
                from pushbullet import Pushbullet
                self.pushbullet_client = Pushbullet(api_key)
                self.update_status("Notification system initialized")
                logger.info("Pushbullet client initialized")
            except ImportError:
                messagebox.showwarning("Module Missing", 
                                       "Pushbullet module not found. Install with: pip install pushbullet.py")
                logger.warning("Pushbullet module not installed")
            except Exception as e:
                messagebox.showerror("API Error", f"Failed to initialize Pushbullet: {e}")
                logger.error(f"Pushbullet initialization error: {e}")
                self.pushbullet_client = None
        else:
            self.pushbullet_client = None
    
    def read_classes_from_file(self) -> List[str]:
        """Read object detection class names from file"""
        classes = []
        try:
            classes_file = self.config["classes_file"]
            if os.path.exists(classes_file):
                with open(classes_file, 'r') as file:
                    classes = [line.strip() for line in file if line.strip()]
                logger.info(f"Loaded {len(classes)} classes from {classes_file}")
            else:
                logger.warning(f"Classes file not found: {classes_file}")
        except Exception as e:
            logger.error(f"Error reading classes file: {e}")
        return classes
    
    def ensure_log_file(self):
        """Ensure the detection log file exists with headers"""
        log_file = self.config["log_file"]
        if not os.path.exists(log_file):
            try:
                with open(log_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["timestamp", "class", "confidence", "location"])
                logger.info(f"Created new log file: {log_file}")
            except Exception as e:
                logger.error(f"Error creating log file: {e}")
    
    def load_model(self):
        """Load the YOLO object detection model"""
        try:
            self.start_progress()
            model_path = self.config["model_path"]
            if not os.path.exists(model_path):
                self.update_status(f"Model file not found: {model_path}")
                messagebox.showerror("Model Not Found", 
                                    f"The model file {model_path} was not found. Please check the path.")
                return
            
            # Load model in a separate thread to prevent UI freeze
            threading.Thread(target=self._load_model_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.update_status(f"Error loading model: {e}")
            self.stop_progress()
    
    def _load_model_thread(self):
        """Thread function to load the model"""
        try:
            model_path = self.config["model_path"]
            self.model = YOLO(model_path)
            self.root.after(0, lambda: self.update_status(f"Model loaded: {model_path}"))
            logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Error in model loading thread: {e}")
            self.root.after(0, lambda: self.update_status(f"Error loading model: {e}"))
        finally:
            self.root.after(0, self.stop_progress)
    
    def initialize_webcam(self):
        """Initialize webcam for video capture"""
        if not self.is_camera_on:
            try:
                self.cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
                if not self.cap.isOpened():
                    messagebox.showerror("Camera Error", "Could not open webcam")
                    return
                
                self.is_camera_on = True
                self.video_paused = False
                
                # Start processing in a separate thread
                self.stop_thread = False
                self.processing_thread = threading.Thread(target=self.process_video_thread, daemon=True)
                self.processing_thread.start()
                
                self.update_status("Webcam started")
                self.update_button_states(camera_on=True)
            except Exception as e:
                logger.error(f"Error initializing webcam: {e}")
                self.update_status(f"Error initializing webcam: {e}")
    
    def select_file(self):
        """Select and open a video file"""
        if self.is_camera_on:
            self.stop_video_source()
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path):
        """Load a video file for processing"""
        try:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("File Error", "Could not open video file")
                return
            
            # Save as last used video
            self.config["last_used_video"] = file_path
            self.save_config()
            
            self.is_camera_on = True
            self.video_paused = False
            
            # Start processing in a separate thread
            self.stop_thread = False
            self.processing_thread = threading.Thread(target=self.process_video_thread, daemon=True)
            self.processing_thread.start()
            
            self.update_status(f"Video loaded: {os.path.basename(file_path)}")
            self.update_button_states(camera_on=True)
            
            # Enable playback position slider for video files
            self.playback_position.state(['!disabled'])
            # Get total frames for slider
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.playback_position.configure(to=total_frames)
                self.playback_position.set(0)
                # Connect slider to seek function
                self.playback_position.configure(command=self.seek_video)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            self.update_status(f"Error loading video: {e}")
    
    def seek_video(self, value):
        """Seek to a specific position in the video"""
        if self.cap and self.is_camera_on and not self.cap.isOpened():
            try:
                frame_pos = int(float(value))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            except Exception as e:
                logger.error(f"Error seeking video: {e}")
    
    def stop_video_source(self):
        """Stop the current video source (webcam or file)"""
        if self.is_camera_on:
            # Signal the processing thread to stop
            self.stop_thread = True
            
            # Wait for thread to finish if it's running
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            # Release the video capture device
            if self.cap:
                self.cap.release()
            
            self.is_camera_on = False
            self.video_paused = False
            self.update_status("Video source stopped")
            self.update_button_states(camera_on=False)
            
            # Disable playback position slider
            self.playback_position.state(['disabled'])
            
            # Reset canvas with splash image
            self.load_splash_image()
    
    def pause_resume_video(self):
        """Pause or resume the video playback"""
        self.video_paused = not self.video_paused
        status = "paused" if self.video_paused else "resumed"
        self.update_status(f"Video {status}")
    
    def toggle_outline(self):
        """Toggle object outlines on/off"""
        self.show_outline = not self.show_outline
        status = "enabled" if self.show_outline else "disabled"
        self.update_status(f"Object outlines {status}")
    
    def define_restricted_area(self):
        """Open a window to define a restricted area"""
        if not self.is_camera_on or not self.cap:
            messagebox.showinfo("Information", "Please start the camera or load a video first")
            return
        
        # Pause video processing while defining area
        was_paused = self.video_paused
        self.video_paused = True
        
        # Get current frame
        ret, frame = self.cap.read()
        if not ret:
            self.video_paused = was_paused
            messagebox.showerror("Error", "Could not get frame from video source")
            return
        
        # Create a new toplevel window
        area_window = tk.Toplevel(self.root)
        area_window.title("Define Monitoring Zone")
        area_window.geometry("1024x600")
        area_window.protocol("WM_DELETE_WINDOW", lambda: self.on_area_window_close(area_window, was_paused))
        
        # Resize frame for display
        frame = cv2.resize(frame, (1000, 500))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Canvas for image and drawing
        canvas = tk.Canvas(area_window, width=1000, height=500)
        canvas.pack(pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep a reference
        
        # Variables for tracking rectangle drawing
        self.drawing = False
        self.rect_start_x = 0
        self.rect_start_y = 0
        self.rect_id = None
        
        # Instructions label
        instructions = ttk.Label(area_window, 
                                text="Click and drag to define monitoring zone. Press Save when done.")
        instructions.pack(pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(area_window)
        btn_frame.pack(pady=10)
        
        # Buttons
        save_btn = ttk.Button(btn_frame, text="Save Zone", 
                             command=lambda: self.save_area_from_dialog(area_window, was_paused))
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(btn_frame, text="Cancel", 
                               command=lambda: self.on_area_window_close(area_window, was_paused))
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Mouse event bindings for rectangle drawing
        canvas.bind("<ButtonPress-1>", 
                   lambda event: self.on_press(event, canvas))
        canvas.bind("<B1-Motion>", 
                   lambda event: self.on_drag(event, canvas))
        canvas.bind("<ButtonRelease-1>", 
                   lambda event: self.on_release(event, canvas))
        
        # If we already have a restricted area, draw it
        if len(self.restricted_area_pts) == 2:
            x1, y1 = self.restricted_area_pts[0]
            x2, y2 = self.restricted_area_pts[1]
            canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    
    def on_press(self, event, canvas):
        """Handle mouse press for drawing rectangle"""
        self.drawing = True
        self.rect_start_x = event.x
        self.rect_start_y = event.y
        
        # Create new rectangle starting from press point
        self.rect_id = canvas.create_rectangle(
            self.rect_start_x, self.rect_start_y, 
            self.rect_start_x, self.rect_start_y,
            outline="red", width=2
        )
    
    def on_drag(self, event, canvas):
        """Handle mouse drag for updating rectangle"""
        if self.drawing and self.rect_id:
            # Update rectangle to current mouse position
            canvas.coords(
                self.rect_id,self.rect_start_x, self.rect_start_y, 
                event.x, event.y
            )
            
    def on_release(self, event, canvas):
        """Handle mouse release to finalize rectangle"""
        self.drawing = False
        # Store the area points
        self.restricted_area_pts = [
            (self.rect_start_x, self.rect_start_y),
            (event.x, event.y)
        ]
        # Make sure the points are ordered (min x,y to max x,y)
        self.normalize_area_points()
    
    def normalize_area_points(self):
        """Ensure the restricted area points are in the correct order (min to max)"""
        if len(self.restricted_area_pts) == 2:
            x1, y1 = self.restricted_area_pts[0]
            x2, y2 = self.restricted_area_pts[1]
            
            self.restricted_area_pts = [
                (min(x1, x2), min(y1, y2)),
                (max(x1, x2), max(y1, y2))
            ]
    
    def save_area_from_dialog(self, window, was_paused):
        """Save the defined area and close the dialog"""
        self.restricted_area_enabled = True
        self.save_restricted_area()
        self.on_area_window_close(window, was_paused)
    
    def on_area_window_close(self, window, was_paused):
        """Handle closing of the area definition window"""
        # Restore previous pause state
        self.video_paused = was_paused
        window.destroy()
    
    def save_restricted_area(self):
        """Save the restricted area to a file"""
        if len(self.restricted_area_pts) != 2:
            messagebox.showinfo("Information", "No monitoring zone defined")
            return
        
        try:
            with open(self.config["restricted_area_file"], 'w') as file:
                json.dump(self.restricted_area_pts, file)
            self.update_status("Monitoring zone saved")
            logger.info("Monitoring zone saved")
        except Exception as e:
            logger.error(f"Error saving monitoring zone: {e}")
            self.update_status(f"Error saving monitoring zone: {e}")
    
    def load_restricted_area(self):
        """Load the restricted area from a file"""
        try:
            if os.path.exists(self.config["restricted_area_file"]):
                with open(self.config["restricted_area_file"], 'r') as file:
                    self.restricted_area_pts = json.load(file)
                self.restricted_area_enabled = True
                self.update_status("Monitoring zone loaded")
                logger.info("Monitoring zone loaded")
            else:
                self.restricted_area_pts = []
                self.restricted_area_enabled = False
        except Exception as e:
            logger.error(f"Error loading monitoring zone: {e}")
            self.update_status(f"Error loading monitoring zone: {e}")
            self.restricted_area_pts = []
            self.restricted_area_enabled = False
    
    def clear_restricted_area(self):
        """Clear the current restricted area"""
        self.restricted_area_pts = []
        self.restricted_area_enabled = False
        
        # Remove the file if it exists
        try:
            if os.path.exists(self.config["restricted_area_file"]):
                os.remove(self.config["restricted_area_file"])
        except Exception as e:
            logger.error(f"Error removing monitoring zone file: {e}")
        
        self.update_status("Monitoring zone cleared")
    
    def process_video_thread(self):
        """Process video frames in a separate thread"""
        frame_count = 0
        
        while not self.stop_thread:
            if self.is_camera_on and not self.video_paused:
                try:
                    # Read a frame
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        # End of video file or camera disconnected
                        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            # Reached end of video file
                            self.root.after(0, lambda: self.update_status("End of video reached"))
                            self.root.after(0, lambda: self.stop_video_source())
                        else:
                            # Camera disconnected or other error
                            self.root.after(0, lambda: self.update_status("Video source error"))
                            self.root.after(0, lambda: self.stop_video_source())
                        break
                    
                    # Skip frames for performance if needed
                    frame_count += 1
                    if frame_count % self.config["frame_skip_threshold"] != 0:
                        continue
                    
                    # Resize frame for display
                    display_frame = cv2.resize(
                        frame, 
                        (self.canvas.winfo_width(), self.canvas.winfo_height())
                    )
                    
                    # Process the frame with object detection
                    processed_frame = self.process_frame(display_frame)
                    
                    # Convert to format for tkinter
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(processed_frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    # Update UI from main thread
                    self.root.after(0, lambda f=img_tk: self.update_canvas(f))
                    
                    # Update slider position for video files (not webcam)
                    if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.root.after(0, lambda pos=current_pos: self.update_slider(pos))
                    
                except Exception as e:
                    logger.error(f"Error in video processing: {e}")
                    self.root.after(0, lambda msg=str(e): self.update_status(f"Processing error: {msg}"))
            else:
                # If paused, wait a bit before checking again
                time.sleep(0.05)
    
    def update_canvas(self, img_tk):
        """Update the canvas with a new image"""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img = img_tk  # Keep a reference
    
    def update_slider(self, position):
        """Update the video playback slider position"""
        if not self.playback_position.instate(['disabled']):
            try:
                self.playback_position.set(position)
            except:
                pass
    
    def process_frame(self, frame):
        """Process a video frame with object detection"""
        if self.model is None:
            # If model isn't loaded, just return the original frame
            return frame
        
        # Create a copy for drawing
        output_frame = frame.copy()
        
        try:
            # Get confidence threshold
            conf_threshold = self.config["confidence_threshold"]
            
            # Run detection
            results = self.model.predict(
                frame, 
                conf=conf_threshold, 
                verbose=False
            )
            
            # Extract detections
            boxes = results[0].boxes
            
            # Reset crowd count
            self.current_crowd_count = 0
            
            # Process each detection
            for box in boxes:
                # Get box data
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get class name
                if cls_id < len(self.class_list):
                    cls_name = self.class_list[cls_id]
                else:
                    cls_name = f"Class {cls_id}"
                
                # Check if we're filtering by class
                selected_class = self.class_selection.get()
                if selected_class != "All" and selected_class != cls_name:
                    continue
                
                # Check if detection is in restricted area (if enabled)
                in_restricted_area = not self.restricted_area_enabled or self.is_in_restricted_area(
                    (x1, y1), (x2, y2)
                )
                
                if in_restricted_area:
                    # Count persons
                    if cls_name == "person":
                        self.current_crowd_count += 1
                    
                    # Draw bounding box if enabled
                    if self.show_outline:
                        # Use different colors for different classes
                        color = self.get_color_for_class(cls_id)
                        
                        # Draw rectangle
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with confidence
                        label = f"{cls_name}: {conf:.2f}"
                        
                        # Get text size
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Fill background for text
                        cv2.rectangle(
                            output_frame, 
                            (x1, y1 - text_size[1] - 5), 
                            (x1 + text_size[0] + 5, y1), 
                            color, 
                            -1
                        )
                        
                        # Put text
                        cv2.putText(
                            output_frame, 
                            label, 
                            (x1 + 3, y1 - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            1, 
                            cv2.LINE_AA
                        )
                    
                    # Log detection
                    self.log_detection(cls_name, conf, (x1, y1, x2, y2))
            
            # Draw restricted area if enabled
            if self.restricted_area_enabled and len(self.restricted_area_pts) == 2:
                cv2.rectangle(
                    output_frame, 
                    self.restricted_area_pts[0], 
                    self.restricted_area_pts[1], 
                    (0, 0, 255), 
                    2
                )
            
            # Display crowd count
            crowd_text = f"People count: {self.current_crowd_count}"
            cv2.putText(
                output_frame, 
                crowd_text, 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Check crowd threshold
            if self.current_crowd_count > self.config["crowd_threshold"]:
                alert_text = "ALERT: Crowd limit exceeded!"
                cv2.putText(
                    output_frame, 
                    alert_text, 
                    (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                # Send notification
                self.send_notification(
                    "Crowd Alert", 
                    f"Crowd limit exceeded! Detected {self.current_crowd_count} people."
                )
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Add error text to frame
            cv2.putText(
                output_frame, 
                f"Processing error: {str(e)[:50]}...", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
        
        return output_frame
    
    def get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get a unique color for a class ID"""
        # Create a color map for different classes
        colors = [
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 0, 255),    # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
            (128, 0, 0),    # dark blue
            (0, 128, 0),    # dark green
            (0, 0, 128),    # dark red
            (128, 128, 0),  # dark cyan
            (128, 0, 128),  # dark magenta
            (0, 128, 128),  # dark yellow
        ]
        return colors[class_id % len(colors)]
    
    def is_in_restricted_area(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> bool:
        """Check if a detection is inside the restricted area"""
        if not self.restricted_area_enabled or len(self.restricted_area_pts) != 2:
            return True
        
        # Get coordinates
        rx1, ry1 = self.restricted_area_pts[0]
        rx2, ry2 = self.restricted_area_pts[1]
        
        # Check if the detection overlaps with the restricted area
        return not (pt2[0] < rx1 or pt1[0] > rx2 or pt2[1] < ry1 or pt1[1] > ry2)
    
    def log_detection(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """Log detection to CSV file with cooldown"""
        current_time = time.time()
        
        # Check if cooldown time has passed since the last log
        if current_time - self.last_log_time >= self.config["log_cooldown"]:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                
                with open(self.config["log_file"], mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, class_name, f"{confidence:.4f}", bbox_str])
                
                self.last_log_time = current_time
            except Exception as e:
                logger.error(f"Error logging detection: {e}")
    
    def send_notification(self, title: str, body: str):
        """Send a Pushbullet notification with cooldown"""
        if not self.pushbullet_client:
            return
        
        current_time = time.time()
        
        # Check if cooldown time has passed since the last notification
        if current_time - self.last_notification_time >= self.config["notification_cooldown"]:
            try:
                self.pushbullet_client.push_note(title, body)
                logger.info(f"Notification sent: {title}")
                self.last_notification_time = current_time
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    def update_status(self, message: str):
        """Update the status bar message"""
        self.status_var.set(message)
        logger.info(message)
    
    def update_button_states(self, camera_on: bool = False):
        """Update the states of buttons based on camera status"""
        if camera_on:
            self.webcam_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.file_button.configure(state=tk.DISABLED)
            self.pause_button.configure(state=tk.NORMAL)
        else:
            self.webcam_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.file_button.configure(state=tk.NORMAL)
            self.pause_button.configure(state=tk.DISABLED)
    
    def start_progress(self):
        """Start the progress bar"""
        self.progress.start(10)
    
    def stop_progress(self):
        """Stop the progress bar"""
        self.progress.stop()
    
    def show_statistics(self):
        """Show detection statistics"""
        try:
            # Create a new window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Detection Statistics")
            stats_window.geometry("600x400")
            
            # Create a frame for stats
            stats_frame = ttk.Frame(stats_window, padding=10)
            stats_frame.pack(fill='both', expand=True)
            
            # Add a notebook for tabs
            notebook = ttk.Notebook(stats_frame)
            notebook.pack(fill='both', expand=True)
            
            # Summary tab
            summary_tab = ttk.Frame(notebook)
            notebook.add(summary_tab, text="Summary")
            
            # Detail tab
            detail_tab = ttk.Frame(notebook)
            notebook.add(detail_tab, text="Detection Log")
            
            # Process log file and display stats
            if os.path.exists(self.config["log_file"]):
                try:
                    # Read CSV file
                    df = pd.read_csv(self.config["log_file"])
                    
                    # Summary stats
                    if not df.empty:
                        # Convert timestamp column to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Count by class
                        class_counts = df['class'].value_counts()
                        
                        # Create a bar chart
                        fig = self.create_class_bar_chart(class_counts)
                        self.add_bar_chart_to_tab(fig, summary_tab)
                        
                        # Add summary text
                        summary_text = f"Total detections: {len(df)}\n"
                        summary_text += f"Unique classes: {len(class_counts)}\n"
                        summary_text += f"First detection: {df['timestamp'].min()}\n"
                        summary_text += f"Last detection: {df['timestamp'].max()}\n"
                        
                        summary_label = ttk.Label(summary_tab, text=summary_text, font=("Arial", 10))
                        summary_label.pack(pady=10)
                        
                        # Create table for detail tab
                        self.create_detection_table(df, detail_tab)
                    else:
                        ttk.Label(summary_tab, text="No detection data available").pack(pady=20)
                        ttk.Label(detail_tab, text="No detection data available").pack(pady=20)
                
                except Exception as e:
                    logger.error(f"Error processing statistics: {e}")
                    ttk.Label(summary_tab, text=f"Error processing statistics: {e}").pack(pady=20)
            else:
                ttk.Label(summary_tab, text="No detection log file found").pack(pady=20)
            
            # Close button
            close_btn = ttk.Button(stats_window, text="Close", command=stats_window.destroy)
            close_btn.pack(pady=10)
        
        except Exception as e:
            logger.error(f"Error showing statistics: {e}")
            messagebox.showerror("Error", f"Failed to show statistics: {e}")
    
    def create_class_bar_chart(self, class_counts):
        """Create a bar chart for class counts"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create figure
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot
            bars = ax.bar(class_counts.index, class_counts.values, color='skyblue')
            
            # Add labels and title
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Detections by Class')
            
            # Rotate x labels if there are many classes
            if len(class_counts) > 5:
                ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', ha='center', va='bottom')
            
            fig.tight_layout()
            return fig
        
        except ImportError:
            logger.warning("Matplotlib not installed, cannot create charts")
            return None
    
    def add_bar_chart_to_tab(self, fig, tab):
        """Add a matplotlib figure to a tab"""
        if fig:
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                canvas = FigureCanvasTkAgg(fig, master=tab)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)
                canvas.draw()
            except Exception as e:
                logger.error(f"Error adding chart to tab: {e}")
                ttk.Label(tab, text=f"Error displaying chart: {e}").pack(pady=20)
        else:
            ttk.Label(tab, text="Chart generation failed. Matplotlib may be missing.").pack(pady=20)
    
    def create_detection_table(self, df, tab):
        """Create a table to show detection log data"""
        try:
            # Create a frame for the table
            table_frame = ttk.Frame(tab)
            table_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create scrollbar
            scrollbar_y = ttk.Scrollbar(table_frame)
            scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            scrollbar_x = ttk.Scrollbar(table_frame, orient='horizontal')
            scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Create treeview
            columns = list(df.columns)
            tree = ttk.Treeview(
                table_frame, 
                columns=columns, 
                show='headings',
                yscrollcommand=scrollbar_y.set,
                xscrollcommand=scrollbar_x.set
            )
            
            # Set column headings
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Add data rows (limit to last 1000 entries for performance)
            for i, row in df.tail(1000).iterrows():
                values = list(row)
                tree.insert('', tk.END, values=values)
            
            # Configure scrollbars
            scrollbar_y.config(command=tree.yview)
            scrollbar_x.config(command=tree.xview)
            
            # Pack the tree
            tree.pack(fill='both', expand=True)
            
        except Exception as e:
            logger.error(f"Error creating detection table: {e}")
            ttk.Label(tab, text=f"Error creating table: {e}").pack(pady=20)
    
    def export_logs(self):
        """Export detection logs to a new file"""
        if not os.path.exists(self.config["log_file"]):
            messagebox.showinfo("Information", "No log file exists yet")
            return
        
        # Ask user for export location
        export_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Detection Log"
        )
        
        if export_path:
            try:
                # Copy the log file
                import shutil
                shutil.copy2(self.config["log_file"], export_path)
                self.update_status(f"Log exported to {export_path}")
                messagebox.showinfo("Success", f"Log exported to {export_path}")
            except Exception as e:
                logger.error(f"Error exporting log: {e}")
                messagebox.showerror("Error", f"Failed to export log: {e}")
    
    def reset_settings(self):
        """Reset all settings to default"""
        confirm = messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset all settings to default? This will not delete detection logs."
        )
        
        if confirm:
            # Reset config to defaults
            self.config = self.default_config.copy()
            self.save_config()
            
            # Clear restricted area
            self.clear_restricted_area()
            
            # Update UI elements
            self.crowd_threshold_var.set(str(self.config["crowd_threshold"]))
            self.confidence_var.set(str(self.config["confidence_threshold"]))
            self.api_key_var.set(self.config["pushbullet_api_key"])
            
            # Reset notification system
            self.pushbullet_client = None
            
            self.update_status("All settings have been reset to default")
    
    def show_help(self):
        """Show help and information dialog"""
        help_text = """
        Smart Object Detection System
        
        Basic Operations:
        - Start Webcam: Activates your webcam for live detection
        - Select Video File: Choose a video file for detection
        - Stop Camera/Video: Stops the current video source
        - Pause/Resume: Toggles playback of the current video
        
        Monitoring Zone:
        - Define Zone: Create a rectangular area for focused monitoring
        - Only objects within this zone will be counted and logged
        
        Detection Settings:
        - Filter Class: Choose a specific object type to detect
        - Crowd Threshold: Set the number of people that triggers an alert
        - Confidence: Minimum detection confidence (0.0-1.0)
        
        Notifications:
        - Enter a Pushbullet API key to receive alerts on your devices
        - Notifications are sent when crowd threshold is exceeded
        
        For more information, check the documentation or log files.
        """
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help & Information")
        help_window.geometry("600x500")
        
        # Add help text
        text_widget = tk.Text(help_window, wrap="word", padx=15, pady=15)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")
        
        # About section
        about_frame = ttk.Frame(help_window, padding=10)
        about_frame.pack(fill="x")
        
        about_label = ttk.Label(
            about_frame, 
            text="Smart Object Detection System v1.0\n© 2025"
        )
        about_label.pack()
        
        # Close button
        close_btn = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_btn.pack(pady=10)
    
    def quit_app(self):
        """Safely quit the application"""
        # Stop any running processes
        self.stop_video_source()
        
        # Close all opencv windows
        cv2.destroyAllWindows()
        
        # Save configuration
        self.save_config()
        
        # Destroy the root window
        self.root.quit()
        self.root.destroy()


def main():
    # Configure exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Install exception handler
    import sys
    sys.excepthook = handle_exception
    
    # Create main window
    root = tk.Tk()
    root.title("Smart Object Detection System")
    root.geometry("1100x700")
    
    # Set icon if available
    try:
        icon_paths = ["icon.ico", "icon.png"]
        for path in icon_paths:
            if os.path.exists(path):
                if path.endswith(".ico"):
                    root.iconbitmap(path)
                else:
                    img = Image.open(path)
                    photo = ImageTk.PhotoImage(img)
                    root.iconphoto(True, photo)
                break
    except Exception as e:
        logger.error(f"Error setting app icon: {e}")
    
    # Create the application
    app = ObjectDetectionApp(root)
    
    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()