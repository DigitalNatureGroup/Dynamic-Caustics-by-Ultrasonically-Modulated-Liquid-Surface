import numpy as np
import serial
import pandas as pd
import math
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import scipy.io
import os
import time
import threading
import matplotlib.pyplot as plt

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

class SonicControlSystem:
    def __init__(self, serial_port="COM3", baudrate=230400, target_img_path=None, frames=3):
        """
        Initialize the Sonic Control System
        
        Args:
            serial_port (str): COM port for the sonic device
            baudrate (int): Serial baudrate
            target_img_path (str, optional): Path to target image
            frames (int): Number of frames for time-multiplexed patterns
        """
        # Serial connection setup
        self.ser = serial.Serial()
        self.ser.port = serial_port
        self.ser.baudrate = baudrate
        self.ser.timeout = 10
        
        # Constants
        self.f = 40e3  # frequency in Hz
        self.c_0 = 346  # speed of sound in air [m/s]
        
        # Board IDs and communication commands
        self.board_id_1 = np.array([192], dtype="uint8")
        self.board_id_2 = np.array([193], dtype="uint8")
        self.comm_init = np.array([254], dtype="uint8")
        self.comm_end = np.array([253], dtype="uint8")
        
        # Propagation constants
        self.tranducer_power_per_voltage = (2.214 / 10)  # transducer power [Pa at 1 m per voltage]
        self.freq = 40e3  # frequency in [Hz]
        self.c0 = 341  # speed of sound in air [m/s]
        self.l_ambda = self.c0 / self.freq  # Acoustic wavelength
        self.k = 2 * math.pi * self.freq / self.c0  # Acoustic wavenumber
        self.transducer_raidus = 4.5e-03  # Radius of transducer
        self.trans_q = [0, 0, 1]  # Transducer Normal
        
        # TensorFlow setup
        self.capture_img = tf.Variable(tf.zeros([192, 192]))
        
        # Target image setup
        self.target_img = None
        if target_img_path:
            self.load_target_image(target_img_path)
            
        # Webcam setup
        self.cap = None
        
        # Reference image for difference calculation
        self.flat_image = None
        
        # Phase optimization parameters
        self.frame = frames  # Frame count for time-averaged patterns
        self.phase_set_for_loop = None
        
        # Set up TensorFlow propagation model
        self._setup_propagation_model()
    
    def _setup_propagation_model(self):
        """Set up the TensorFlow propagation model for acoustic pressure calculation"""
        # CONSTANT
        self.PHASE_RESOLUTION = (2 * math.pi) / 32.0
        self.PHASE_RESOLUTION = tf.cast(self.PHASE_RESOLUTION, tf.float64)

        # Simulation Domain
        grid_size = self.l_ambda / 10
        self.x_c = tf.range(-48e-3, 48e-3, 0.0005)
        self.y_c = tf.range(-48e-3, 48e-3, 0.0005)
        self.z_plane = 200e-3
        self.x_c = tf.cast(self.x_c, tf.float64)
        self.y_c = tf.cast(self.y_c, tf.float64)
        self.z_plane = tf.cast(self.z_plane, tf.float64)

        # PAT SETTINGS
        INPUT_VOLTAGE = 20
        self.p_0 = self.tranducer_power_per_voltage * INPUT_VOLTAGE
        
        # Load transducer positions from .mat file
        try:
            Dictionary = scipy.io.loadmat("./facingPATs_transducers.mat")
            self.TRANS_X = Dictionary["array_x"].ravel()
            self.TRANS_Y = Dictionary["array_y"].ravel()
            self.TRANS_Z = Dictionary["array_z"].ravel()
            self.TRANS_NUM = len(self.TRANS_X)
            self.TRANS_X = tf.cast(self.TRANS_X, tf.float64)
            self.TRANS_Y = tf.cast(self.TRANS_Y, tf.float64)
            self.TRANS_Z = tf.cast(self.TRANS_Z, tf.float64)
            self.trans_q = tf.cast(self.trans_q, tf.float64)
        except Exception as e:
            print(f"Error loading transducer positions: {e}")
            # Create dummy values if file not found
            self.TRANS_X = tf.cast(np.linspace(-10e-3, 10e-3, 256), tf.float64)
            self.TRANS_Y = tf.cast(np.linspace(-10e-3, 10e-3, 256), tf.float64)
            self.TRANS_Z = tf.cast(np.zeros(256), tf.float64)
            self.TRANS_NUM = 256

        # XY plane setup
        XX, YY = tf.meshgrid(self.x_c, self.y_c)
        r_prep_x = XX[:, :, tf.newaxis] - self.TRANS_X
        r_prep_y = YY[:, :, tf.newaxis] - self.TRANS_Y
        r_prep_z = self.z_plane - self.TRANS_Z

        # pressure calculation setup
        self.d_pt = tf.sqrt(r_prep_x**2 + r_prep_y**2 + r_prep_z**2)
        dotproduct = r_prep_x * self.trans_q[0] + r_prep_y * self.trans_q[1] + r_prep_z * self.trans_q[2]
        theta = tf.acos(dotproduct / self.d_pt / tf.norm(self.trans_q, ord=2))
        self.D = self.directivity(self.k, self.transducer_raidus, theta)

        # For time-averaged mode
        self.d_pt = tf.expand_dims(self.d_pt, 0)
        self.d_pt = tf.tile(self.d_pt, [self.frame, 1, 1, 1])
        self.D = tf.expand_dims(self.D, 0)
        self.D = tf.tile(self.D, [self.frame, 1, 1, 1])

        # Convert to complex for pressure calculations
        self.k = tf.cast(self.k, tf.complex64)
        self.p_0 = tf.cast(self.p_0, tf.complex64)
        self.d_pt = tf.cast(self.d_pt, tf.complex64)
        self.D = tf.cast(self.D, tf.complex64)
    
    def load_target_image(self, image_path):
        """
        Load and prepare target image for optimization
        
        Args:
            image_path (str): Path to target image
            
        Returns:
            tf.Tensor: Processed target image
        """
        img = Image.open(image_path).convert('L')
        img = ImageOps.flip(img)
        binary = np.array(img).reshape(192, 192, 1).astype(np.float64)                                                                                                                                                          
        binary[binary < 128] = 0                                                                                                                                                                         
        binary[binary >= 128] = 1  
        self.target_img = self.target_resize_func_propagate(binary)
        return self.target_img
    
    def initialize_serial(self):
        """
        Initialize serial connection to the sonic device
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.ser.is_open:
                self.ser.open()
            return True
        except Exception as e:
            print(f"Error opening serial port: {e}")
            return False
    
    def initialize_webcam(self, camera_id=0):
        """
        Initialize webcam for image capture
        
        Args:
            camera_id (int): Camera device ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            return True
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            return False
    
    def capture_reference_image(self):
        """
        Capture a reference image (flat field) for differential imaging
        
        Returns:
            numpy.ndarray: Captured reference image
        """
        if self.cap is None:
            print("Error: Webcam not initialized")
            return None
        
        print("Capturing reference image...")
        
        # Capture multiple frames and average for better quality
        frames = []
        for _ in range(10):
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            time.sleep(0.1)
        
        if not frames:
            print("Error: Could not capture reference frames")
            return None
            
        # Average the frames
        self.flat_image = np.mean(frames, axis=0).astype(np.uint8)
        self.flat_image = cv2.resize(self.flat_image, (300, 300))
        
        # Crop the center region
        height, width = self.flat_image.shape
        center_x, center_y = width // 2, height // 2
        self.flat_image = self.flat_image[center_y-150:center_y+150, center_x-150:center_x+150]
        
        print("Reference image captured")
        return self.flat_image
    
    def capture_webcam_image(self, use_difference=True, rotation=None):
        """
        Capture image from webcam and process it for optimization
        
        Args:
            use_difference (bool): Apply difference with reference image
            rotation (int, optional): Rotation angle in degrees (90, 180, 270) or None
            
        Returns:
            numpy.ndarray: Processed image
        """
        if self.cap is None:
            print("Error: Webcam not initialized")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            return None
        
        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (300, 300))
        
        # Crop to center region
        height, width = frame.shape
        center_x, center_y = width // 2, height // 2
        cropped = frame[center_y-150:center_y+150, center_x-150:center_x+150]
        
        # Apply difference with reference image if requested
        if use_difference and self.flat_image is not None:
            diff_image = cropped.astype(np.int16) - self.flat_image.astype(np.int16)
            diff_image = diff_image - np.min(diff_image)  # Shift to positive values
            processed = diff_image.astype(np.uint8)
        else:
            processed = cropped
            
        # Apply rotation if specified
        if rotation == 90:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            processed = cv2.rotate(processed, cv2.ROTATE_180)
        elif rotation == 270:
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Flip and resize for TensorFlow
        flipped = cv2.flip(processed, 0)  # Vertical flip
        resized = cv2.resize(flipped, (192, 192))
        
        # Update the TensorFlow variable
        self.capture_img.assign(resized)
        
        return resized
    
    def close_all(self):
        """Close all connections and resources"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
    
    def set_phase(self, phase_arr):
        """
        Set the phase for the sonic transducers
        
        Args:
            phase_arr (numpy.ndarray): Phase array (0 to 2π)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ser.is_open:
            print("Error: Serial port not open")
            return False
        
        phase = np.mod(phase_arr, 2 * math.pi)
        phase = np.round(32 * phase / (2 * math.pi))
        phase = phase.astype("uint8")
        
        data = np.concatenate([self.comm_init, self.board_id_1, phase, self.board_id_2, phase, self.comm_end], dtype='uint8')
        values = data.tobytes()
        self.ser.write(values)
        return True
    
    def target_resize_func_propagate(self, img):
        """
        Resize target image for propagation calculation
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            tf.Tensor: Resized image
        """
        resized = tf.image.resize(img, [192, 192], method="bicubic")
        resized = tf.math.greater(resized, 0.5)
        resized = tf.cast(resized, tf.float32)
        resized = tf.reshape(resized, shape=(192, 192))
        return resized
    
    def directivity(self, k, a, theta_n):
        """
        Calculate directivity function for acoustic propagation
        
        Args:
            k: Wave number
            a: Transducer radius
            theta_n: Angle
            
        Returns:
            tf.Tensor: Directivity values
        """
        return 2 * tf.divide(tf.math.bessel_i1e(k * a * tf.sin(theta_n)), k * a * tf.sin(theta_n))
    
    def accuracy(self, target, capture, pressure):
        """
        Calculate accuracy between target and captured/calculated pressure field
        
        Args:
            target: Target image
            capture: Captured image
            pressure: Calculated pressure field
            
        Returns:
            tf.Tensor: Accuracy metric
        """
        replaced_pressure = (pressure + tf.stop_gradient(capture - pressure))
        denom = tf.sqrt(tf.reduce_sum(tf.pow(replaced_pressure, 2), axis=[0, 1]) * tf.reduce_sum(tf.pow(target, 2), axis=[0, 1]))
        return 1 - tf.reduce_mean((tf.reduce_sum(replaced_pressure * target, axis=[0, 1]) + 0.001) / (denom + 0.001))
    
    def accuracy_average(self, target, capture, pressure):
        """
        Calculate average-based accuracy for optimization
        
        Args:
            target: Target image
            capture: Captured image
            pressure: Calculated pressure field
            
        Returns:
            tf.Tensor: Average-based accuracy metric
        """
        replaced_pressure = (pressure + tf.stop_gradient(capture - pressure))
        average_loss = (-1) * tf.reduce_sum(target * replaced_pressure) / tf.reduce_sum(target)
        return average_loss
    
    def absolute_difference(self, target, capture, pressure):
        """
        Calculate absolute difference between normalized target and pressure
        
        Args:
            target: Target image
            capture: Captured image
            pressure: Calculated pressure field
            
        Returns:
            tf.Tensor: Absolute difference metric
        """
        target_normalized = (target - tf.reduce_min(target)) / (tf.reduce_max(target) - tf.reduce_min(target))
        capture_normalized = (capture - tf.reduce_min(capture)) / (tf.reduce_max(capture) - tf.reduce_min(capture))
        pressure_normalized = (pressure - tf.reduce_min(pressure)) / (tf.reduce_max(pressure) - tf.reduce_min(pressure))
        replaced_pressure = (pressure_normalized + tf.stop_gradient(capture_normalized - pressure_normalized))
        absolute_differences = tf.abs(target_normalized - replaced_pressure)
        difference_sum = tf.reduce_sum(absolute_differences)
        return difference_sum
    
    def loss_func(self, phase_set):
        """
        Calculate loss for phase optimization with camera feedback
        
        Args:
            phase_set: Phase settings
            
        Returns:
            tf.Tensor: Loss value with camera feedback
        """
        prop = self.calculate_pressure(phase_set)
        return self.accuracy(self.target_img, self.capture_img, prop)
        
    def simulation_only_loss(self, phase_set):
        """
        Calculate loss for phase optimization using simulation only (no camera feedback)
        
        Args:
            phase_set: Phase settings
            
        Returns:
            tf.Tensor: Loss value based only on simulation
        """
        # Calculate simulation result
        prop = self.calculate_pressure(phase_set)
        
        # Direct comparison without replacement
        denom = tf.sqrt(tf.reduce_sum(tf.pow(prop, 2), axis=[0, 1]) * 
                       tf.reduce_sum(tf.pow(self.target_img, 2), axis=[0, 1]))
        return 1 - tf.reduce_mean((tf.reduce_sum(prop * self.target_img, axis=[0, 1]) + 0.001) / 
                                 (denom + 0.001))
    
    def calculate_pressure(self, phase_set):
        """
        Calculate acoustic pressure field from phase settings
        
        Args:
            phase_set: Phase settings
            
        Returns:
            tf.Tensor: Calculated pressure field
        """
        reshaped_phase_set = tf.reshape(phase_set, [-1, 256])
        reshaped_phase_set = tf.reshape(reshaped_phase_set, (self.frame, 1, 1, 256))
        reshaped_phase_set = tf.tile(reshaped_phase_set, [1, 192, 192, 1])
        reshaped_phase_set = tf.cast(reshaped_phase_set, tf.complex64)
        p2 = tf.reduce_sum((self.p_0 / self.d_pt) * self.D * tf.exp(tf.complex(0.0, 1.0) * (self.k * self.d_pt + reshaped_phase_set)), axis=3)
        pressure = tf.abs(p2)
        total_pressure = tf.reduce_sum(pressure, axis=0)
        average_pressure = total_pressure / tf.cast(tf.shape(phase_set)[0], tf.float32)
        return average_pressure
    
    def load_phase_from_csv(self, csv_path):
        """
        Load phase settings from CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            numpy.ndarray: Loaded phase settings
        """
        try:
            phase_arr = np.loadtxt(csv_path, delimiter=',')
            return phase_arr
        except Exception as e:
            print(f"Error loading phase from CSV: {e}")
            return None
    
    def save_phase_to_csv(self, phase_arr, csv_path):
        """
        Save phase settings to CSV file
        
        Args:
            phase_arr (numpy.ndarray): Phase settings
            csv_path (str): Path to save CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            np.savetxt(csv_path, phase_arr, delimiter=",")
            return True
        except Exception as e:
            print(f"Error saving phase to CSV: {e}")
            return False
    
    def optimize_phase(self, initial_phase=None, iterations=300, learning_rate=0.005, save_path=None, simulation_only=False):
        """
        Optimize phase settings to match target image
        
        Args:
            initial_phase (numpy.ndarray, optional): Initial phase settings
            iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimizer
            save_path (str, optional): Path to save results
            simulation_only (bool): If True, use simulation only without camera feedback
            
        Returns:
            tuple: (optimized_phase, loss_history)
        """
        # Initialize parameters
        if initial_phase is None:
            # Random initialization
            initial_phase = tf.random.uniform(shape=[self.frame, 256], minval=0.0, maxval=2.0 * np.pi)
        
        params = tf.Variable(initial_phase)
        
        # Convert initial phase to device format and send
        phase_set_for_loop_tmp = params.numpy()
        phase_set_for_loop_tmp = np.mod(phase_set_for_loop_tmp, 2 * math.pi)
        phase_set_for_loop_tmp = np.round(32 * phase_set_for_loop_tmp / (2 * math.pi))
        self.phase_set_for_loop = phase_set_for_loop_tmp.astype("uint8")
        
        # Set up optimization
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_history = []
        
        # Set up threading for phase control
        exit_flag = threading.Event()
        
        def loop_phase_control():
            """Thread function to continuously update phase patterns"""
            while not exit_flag.is_set():
                for phase in self.phase_set_for_loop:
                    if exit_flag.is_set():
                        break
                    data = np.concatenate([self.comm_init, self.board_id_1, phase, self.board_id_2, phase, self.comm_end], dtype='uint8')
                    values = data.tobytes()
                    self.ser.write(values)
                    time.sleep(0.05)  # Small delay to not overwhelm the device
        
        def loop_optimization():
            """Thread function to run optimization"""
            for step in range(iterations):
                if not simulation_only:
                    # Get camera image for real-time optimization
                    self.capture_webcam_image(use_difference=True, rotation=90)
                    
                    # Use camera feedback for optimization - TensorFlow 2.x style
                    with tf.GradientTape() as tape:
                        loss = self.loss_func(params)
                    gradients = tape.gradient(loss, [params])
                    opt.apply_gradients(zip(gradients, [params]))
                    current_loss = loss.numpy()
                else:
                    # Use simulation only with a different loss function - TensorFlow 2.x style
                    with tf.GradientTape() as tape:
                        loss = self.simulation_only_loss(params)
                    gradients = tape.gradient(loss, [params])
                    opt.apply_gradients(zip(gradients, [params]))
                    current_loss = loss.numpy()
                
                print(f"Step {step}: Loss = {current_loss}")
                loss_history.append([step, current_loss])
                
                # Update phase settings for the control thread
                phase_set_for_loop_tmp = params.numpy()
                phase_set_for_loop_tmp = np.mod(phase_set_for_loop_tmp, 2 * math.pi)
                phase_set_for_loop_tmp = np.round(32 * phase_set_for_loop_tmp / (2 * math.pi))
                self.phase_set_for_loop = phase_set_for_loop_tmp.astype("uint8")
                
                # Optional: save intermediate results
                if save_path and step % 5 == 0:
                    # Save simulation result
                    pressure = self.calculate_pressure(params).numpy()
                    plt.figure(figsize=(8, 8))
                    plt.imshow(pressure, cmap='hot')
                    plt.title(f"Step {step} - Simulated Pressure")
                    plt.colorbar()
                    plt.savefig(f"{save_path}_sim_step{step}.png")
                    plt.close()
                    
                    # Save camera image if available
                    if not simulation_only:
                        capture = self.capture_img.numpy()
                        cv2.imwrite(f"{save_path}_camera_step{step}.png", capture)
            
            # Save final phase
            if save_path:
                self.save_phase_to_csv(params.numpy(), f"{save_path}_phase.csv")
                
                # Save loss history
                np.savetxt(f"{save_path}_loss_history.csv", loss_history, delimiter=",")
            
            # Signal phase control thread to exit
            exit_flag.set()
        
        if simulation_only:
            # Don't start phase control thread in simulation-only mode
            print("Starting simulation-only optimization...")
            loop_optimization()
        else:
            # Start both threads for digital twin mode
            print("Starting digital twin optimization with camera feedback...")
            thread_control = threading.Thread(target=loop_phase_control)
            thread_optimize = threading.Thread(target=loop_optimization)
            
            thread_control.start()
            time.sleep(3)  # Give time for control to initialize
            thread_optimize.start()
            
            # Wait for both threads to complete
            thread_control.join()
            thread_optimize.join()
        
        print("Optimization completed")
        return params.numpy(), loss_history
    
    def simulate_pressure(self, phase_set, save_path=None):
        """
        Simulate pressure field from phase settings
        
        Args:
            phase_set (numpy.ndarray): Phase settings
            save_path (str, optional): Path to save visualization
            
        Returns:
            numpy.ndarray: Pressure field
        """
        pressure = self.calculate_pressure(phase_set).numpy()
        
        if save_path:
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(self.x_c.numpy() * 1000, self.y_c.numpy() * 1000, pressure, cmap="hot")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.colorbar()
            plt.title("Simulated Acoustic Pressure Field")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            plt.savefig(save_path)
            plt.close()
        
        return pressure
        
    def output_phase_continuously(self, phase_set):
        """
        Continuously output phase patterns until a key is pressed
        
        Args:
            phase_set (numpy.ndarray): Phase patterns to output (frames x 256)
            
        Returns:
            None
        """
        if not self.ser.is_open:
            print("Error: Serial port not open")
            if not self.initialize_serial():
                print("Failed to initialize serial connection")
                return
        
        # Convert phases to device format
        phase_patterns = []
        for phase in phase_set:
            p = np.mod(phase, 2 * math.pi)
            p = np.round(32 * p / (2 * math.pi))
            phase_patterns.append(p.astype("uint8"))
        
        print("Continuously outputting phase patterns...")
        print("Press Enter to stop")
        
        # Start output in a separate thread
        exit_flag = threading.Event()
        
        def output_thread():
            cycle_count = 0
            while not exit_flag.is_set():
                for idx, phase in enumerate(phase_patterns):
                    if exit_flag.is_set():
                        break
                    
                    data = np.concatenate([self.comm_init, self.board_id_1, phase, self.board_id_2, phase, self.comm_end], dtype='uint8')
                    values = data.tobytes()
                    self.ser.write(values)
                    
                    # Print status every 100 cycles
                    if cycle_count % 100 == 0:
                        print(f"Outputting phase pattern {idx+1}/{len(phase_patterns)} (cycle {cycle_count})")
                    
                    time.sleep(0.05)  # Small delay
                cycle_count += 1
        
        thread = threading.Thread(target=output_thread)
        thread.start()
        
        # Wait for Enter key
        input()
        
        # Stop the thread
        exit_flag.set()
        thread.join()
        print("Phase output stopped")
