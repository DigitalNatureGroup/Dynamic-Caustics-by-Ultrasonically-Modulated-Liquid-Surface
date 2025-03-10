# SonicControlSystem

A Python library for controlling acoustic phased arrays with digital twin optimization capabilities. This system allows for numerical simulation and real-time optimization of acoustic pressure fields using webcam feedback.

## Features

- **Time-Multiplexed Optimization**: Create complex acoustic patterns by time-multiplexing multiple phase patterns
- **Digital Twin Optimization**: Combines numerical simulation with camera feedback for optimal results
- **Simulation-Only Mode**: Test and develop patterns using only numerical simulation
- **Real-time Serial Control**: Send optimized patterns to acoustic hardware via serial communication
- **Visualization Tools**: Visualize pressure fields and optimization progress

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- SciPy
- PySerial
- Pillow (PIL)

## Installation

```bash
# Install dependencies
pip install numpy pyserial pandas opencv-python Pillow scipy matplotlib tensorflow
```

## Required Data Files

- `facingPATs_transducers.mat`: Contains transducer positions (x, y, z coordinates)
- `target.png`: Target image for pattern optimization

## Basic Usage

```python
from sonic_control_system import SonicControlSystem

# Create system with 3-frame time-multiplexing
system = SonicControlSystem(
    serial_port="COM3",
    frames=3  # Use 3 frames for time-multiplexing
)

# Load target image
system.load_target_image("target.png")

# Run simulation-only optimization
sim_phase, _ = system.optimize_phase(
    iterations=100,
    learning_rate=0.01,
    save_path="simulation_results",
    simulation_only=True
)

# Save and visualize results
system.save_phase_to_csv(sim_phase, "simulation_phase.csv")
system.simulate_pressure(sim_phase, "pressure_field.png")
```

## Complete Workflow

The system supports a complete workflow:

1. **Numerical Simulation**: Optimize phase patterns using only simulation
2. **Serial Output**: Send simulation results to acoustic hardware
3. **Digital Twin Optimization**: Refine patterns using camera feedback
4. **Final Output**: Send optimized patterns to acoustic hardware

## Digital Twin Process

The digital twin optimization combines simulation with real hardware:

1. Capture camera images of acoustic patterns
2. Compare camera images with simulation
3. Adjust simulation parameters based on real-world feedback
4. Continue optimizing patterns for best results

## Time-Multiplexing

Time-multiplexing creates more complex patterns by rapidly cycling through multiple phase patterns:

```python
# 3-frame time-multiplexing
system = SonicControlSystem(frames=3)

# Optimize with time-multiplexing
phase_patterns, _ = system.optimize_phase(iterations=100)

# The result will have shape [3, 256] - 3 frames, 256 transducers each
```

## Google Colab Support

You can run the simulation-only mode in Google Colab:

```python
# Install dependencies
!pip install numpy pyserial pandas opencv-python Pillow scipy matplotlib

# Create dummy data
# (See create_dummy_data function in examples)

# Run simulation only
system = SonicControlSystem(frames=3)
system.load_target_image("target.png")
phase, _ = system.optimize_phase(
    iterations=100,
    simulation_only=True
)
```

## Hardware Requirements

For complete functionality (digital twin mode):
- Acoustic phased array with serial control interface
- Webcam positioned to capture acoustic patterns

## License

MIT License
