# Gas detector app

C++ application for running real time inference on the raspberry pi 5.

### Key Application Components

#### 1. Capture Thread
- Reads IR frames from MLX90640 (4 FPS)
- Reads gas sensor values from MCP3008
- Buffers synchronised data in WindowQueue
- Configuration: 5-second windows (20 frames + 20 gas readings)

#### 2. Inference Thread
- Retrieves data windows from queue
- Runs TFLite model inference
- Outputs predictions and confidence scores
- Handles 4 classes: normal, aerosol, flame, breath

#### 3. Window Queue
- Thread-safe producer-consumer pattern
- Capacity: 2 windows
- Condition variables for synchronisation
- Prevents data loss during processing

### Dependencies

- **TensorFlow Lite** - Inference framework
- **Abseil-cpp** - Dependency of TFLite
- **Ruy** - Matrix operations library
- **XNNPACK** - CPU acceleration delegate
- **pthreadpool** - Thread pool implementation

---
