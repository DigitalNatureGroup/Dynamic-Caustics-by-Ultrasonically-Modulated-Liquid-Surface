import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import tensorflow as tf
from sonic_control_system import SonicControlSystem

# Display version information
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

def main():
    """
    Main function demonstrating the complete workflow:
    1. Numerical simulation optimization (3 frames)
    2. Serial output of simulation results (exit on keypress)
    3. Digital twin optimization using simulation results as initial values
    4. Serial output of digital twin results (exit on keypress)
    """
    # Create system with 3 frame time-multiplexing
    system = SonicControlSystem(
        serial_port="COM3",
        frames=3  # Use 3 frames for time-multiplexing
    )
    
    # Setup: Load target image
    print("Loading target image...")
    system.load_target_image("target.png")
    
    # Step 1: Numerical simulation (without camera/hardware)
    simulation_phase = run_simulation(system)
    
    # Step 2: Output simulation results via serial port
    print("\n=== Step 2: Outputting simulation results via serial ===")
    print("Initializing serial port...")
    if system.initialize_serial():
        system.output_phase_continuously(simulation_phase)
    else:
        print("Warning: Failed to initialize serial. Skipping output step.")
    
    # Step 3: Digital twin optimization (with camera feedback)
    digital_twin_phase = run_digital_twin(system, simulation_phase)
    
    # Step 4: Output digital twin results via serial port
    print("\n=== Step 4: Outputting digital twin results via serial ===")
    print("Initializing serial port...")
    if system.initialize_serial():
        system.output_phase_continuously(digital_twin_phase)
    else:
        print("Warning: Failed to initialize serial. Skipping output step.")
    
    # Clean up
    system.close_all()
    print("\nAll steps completed.")

def run_simulation(system):
    """Run numerical simulation optimization"""
    print("\n=== Step 1: Running numerical simulation optimization ===")
    
    # Random initialization for simulation
    print("Starting numerical simulation with 3 frames...")
    optimized_phase, loss_history = system.optimize_phase(
        iterations=100,  # Reduced for example purposes
        learning_rate=0.01,
        save_path="simulation_results",
        simulation_only=True  # Use simulation-only mode
    )
    
    # Save final simulation phase
    system.save_phase_to_csv(optimized_phase, "simulation_optimized_phase.csv")
    
    # Visualize final pressure field
    print("Generating pressure field visualization...")
    system.simulate_pressure(optimized_phase, save_path="simulation_pressure_field.png")
    
    # Plot loss history
    plot_loss_history(loss_history, "Simulation Optimization Progress", "simulation_loss_history.png")
    
    print("Simulation optimization completed")
    return optimized_phase

def run_digital_twin(system, initial_phase):
    """Run digital twin optimization with camera feedback"""
    print("\n=== Step 3: Running digital twin optimization ===")
    
    # Initialize webcam
    print("Initializing webcam...")
    if not system.initialize_webcam():
        print("Error: Failed to initialize webcam. Skipping digital twin step.")
        return initial_phase
    
    # Capture reference image
    print("Capturing reference image...")
    system.capture_reference_image()
    
    # Initialize serial connection
    print("Initializing serial connection...")
    if not system.initialize_serial():
        print("Error: Failed to initialize serial. Skipping digital twin step.")
        return initial_phase
    
    # Run digital twin optimization using simulation results as initial values
    print("Starting digital twin optimization...")
    optimized_phase, loss_history = system.optimize_phase(
        initial_phase=initial_phase,
        iterations=50,  # Reduced for example purposes
        learning_rate=0.005,
        save_path="digital_twin_results"
    )
    
    # Save final digital twin phase
    system.save_phase_to_csv(optimized_phase, "digital_twin_optimized_phase.csv")
    
    # Visualize final pressure field
    print("Generating pressure field visualization...")
    system.simulate_pressure(optimized_phase, save_path="digital_twin_pressure_field.png")
    
    # Plot loss history
    plot_loss_history(loss_history, "Digital Twin Optimization Progress", "digital_twin_loss_history.png")
    
    print("Digital twin optimization completed")
    return optimized_phase

def plot_loss_history(loss_history, title, save_path):
    """Plot and save loss history graph"""
    steps = [item[0] for item in loss_history]
    losses = [item[1] for item in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss history saved to {save_path}")

def create_dummy_data():
    """Create dummy data files for testing"""
    # Create dummy transducer positions
    import scipy.io
    
    # Arrange 256 transducers in a circle
    theta = np.linspace(0, 2*np.pi, 256)
    r = 0.05  # 5cm radius
    array_x = r * np.cos(theta)
    array_y = r * np.sin(theta)
    array_z = np.zeros_like(theta)
    
    # Create .mat file
    data = {'array_x': array_x, 'array_y': array_y, 'array_z': array_z}
    scipy.io.savemat('facingPATs_transducers.mat', data)
    print("Created dummy transducer positions file")
    
    # Create a simple target image (circle)
    target = np.zeros((192, 192), dtype=np.uint8)
    cv2.circle(target, (96, 96), 30, 255, -1)
    cv2.imwrite('target.png', target)
    print("Created dummy target image")

if __name__ == "__main__":
    # Check if required files exist, create dummy ones if not
    try:
        import os
        if not os.path.exists("facingPATs_transducers.mat") or not os.path.exists("target.png"):
            print("Required files not found. Creating dummy data...")
            create_dummy_data()
    except Exception as e:
        print(f"Error creating dummy data: {e}")
    
    # Run the main workflow
    main()
