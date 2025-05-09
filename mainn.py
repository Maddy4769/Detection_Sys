import tkinter as tk
import customtkinter as ctk
import subprocess
import psutil
import os
from threading import Thread
import time
import datetime

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class SecuritySuite(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Security Management Suite")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Process tracking
        self.crowd_process = None
        self.weapon_process = None
        self.monitor_thread = None
        self.running = True

        # Create logs directory
        self.log_dir = "security_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # UI Elements
        self.create_widgets()
        self.start_status_monitor()

    def create_widgets(self):
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title
        ctk.CTkLabel(main_frame, text="Security Systems Controller", 
                    font=("Arial", 20, "bold")).pack(pady=10)

        # Button grid
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=20)

        # Control buttons
        self.crowd_button = ctk.CTkButton(button_frame, text="Start Crowd System",
                                         command=self.toggle_crowd)
        self.crowd_button.grid(row=0, column=0, padx=10, pady=5)

        self.weapon_button = ctk.CTkButton(button_frame, text="Start Weapon System",
                                          command=self.toggle_weapon)
        self.weapon_button.grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkButton(button_frame, text="Start Both Systems",
                     command=self.start_both, fg_color="green").grid(row=1, column=0, padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="Stop All Systems",
                     command=self.stop_all, fg_color="red").grid(row=1, column=1, padx=10, pady=5)

        # Quit button that properly closes the application
        ctk.CTkButton(button_frame, text="Quit", 
                     command=self.quit_app,  # Changed to use quit_app method
                     fg_color="darkred").grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        # Status indicators
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.pack(pady=10)

        self.crowd_status = ctk.CTkLabel(status_frame, text="Crowd System: STOPPED",
                                        text_color="red")
        self.crowd_status.grid(row=0, column=0, padx=20)

        self.weapon_status = ctk.CTkLabel(status_frame, text="Weapon System: STOPPED",
                                         text_color="red")
        self.weapon_status.grid(row=0, column=1, padx=20)

        # Log console
        self.log_console = ctk.CTkTextbox(main_frame, height=200)
        self.log_console.pack(pady=10, fill="both", expand=True)
        self.log_console.insert("end", "System Log:\n")
        self.log_console.configure(state="disabled")

    def quit_app(self):
        """New method to handle application quit"""
        self.running = False  # Stop the monitoring thread
        self.stop_all()       # Stop all running processes
        self.destroy()        # Close the application window

    def log_message(self, message):
        self.log_console.configure(state="normal")
        self.log_console.insert("end", f"{self.current_time()} - {message}\n")
        self.log_console.see("end")
        self.log_console.configure(state="disabled")

    def current_time(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def start_status_monitor(self):
        def monitor():
            while self.running:
                self.update_status()
                time.sleep(0.5)

        self.monitor_thread = Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def update_status(self):
        crowd_running = self.is_process_running(self.crowd_process)
        weapon_running = self.is_process_running(self.weapon_process)

        self.after(0, self._update_status_labels, crowd_running, weapon_running)
        self.after(0, self._update_button_states, crowd_running, weapon_running)

    def _update_status_labels(self, crowd, weapon):
        self.crowd_status.configure(
            text=f"Crowd System: {'RUNNING' if crowd else 'STOPPED'}",
            text_color="green" if crowd else "red"
        )
        self.weapon_status.configure(
            text=f"Weapon System: {'RUNNING' if weapon else 'STOPPED'}",
            text_color="green" if weapon else "red"
        )

    def _update_button_states(self, crowd, weapon):
        self.crowd_button.configure(
            text="Stop Crowd System" if crowd else "Start Crowd System",
            fg_color="red" if crowd else None
        )
        self.weapon_button.configure(
            text="Stop Weapon System" if weapon else "Start Weapon System",
            fg_color="red" if weapon else None
        )

    def is_process_running(self, process):
        return process and process.poll() is None

    def toggle_crowd(self):
        if self.is_process_running(self.crowd_process):
            self.stop_process("crowd")
        else:
            self.start_process("crowd")

    def toggle_weapon(self):
        if self.is_process_running(self.weapon_process):
            self.stop_process("weapon")
        else:
            self.start_process("weapon")

    def start_process(self, system):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"{system}_{timestamp}.log")
            
            cmd = {
                "crowd": "python crowd_management.py",
                "weapon": "python weapon_detection.py"
            }[system]

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )

            if system == "crowd":
                self.crowd_process = process
                self.log_message(f"Crowd system started (Log: {log_file})")
            else:
                self.weapon_process = process
                self.log_message(f"Weapon detection started (Log: {log_file})")

        except Exception as e:
            self.log_message(f"Error starting {system} system: {str(e)}")

    def stop_process(self, system):
        try:
            process = self.crowd_process if system == "crowd" else self.weapon_process
            if process and process.poll() is None:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                setattr(self, f"{system}_process", None)
                self.log_message(f"{system.capitalize()} system stopped")
        except Exception as e:
            self.log_message(f"Error stopping {system} system: {str(e)}")

    def start_both(self):
        if not self.is_process_running(self.crowd_process):
            self.start_process("crowd")
        if not self.is_process_running(self.weapon_process):
            self.start_process("weapon")

    def stop_all(self):
        self.stop_process("crowd")
        self.stop_process("weapon")
        self.log_message("All systems stopped")

    def on_close(self):
        if ctk.CTkMessageBox.askyesno("Quit", "Stop all systems and quit?"):
            self.quit_app()  # Now uses the same quit method as the button

if __name__ == "__main__":
    app = SecuritySuite()
    app.mainloop()