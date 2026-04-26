# Gas Detector Application (C++)

Real-time inference engine implemented in C++ for efficient inference on Raspberry Pi 5.

### Application Architecture

```mermaid
graph TD
    subgraph Sensors["Hardware Sensors"]
        MLX["MLX90640<br/>Thermal Sensor<br/>(I2C)"]
        MQ2["MQ2 Gas Sensor<br/>(ADC Ch 1)"]
        MQ7["MQ7 Gas Sensor<br/>(ADC Ch 3)"]
        MQ135["MQ135 Gas Sensor<br/>(ADC Ch 2)"]
        MCP["MCP3008<br/>ADC<br/>(SPI)"]
    end
    
    subgraph Capture["Data Capture Layer"]
        IR["IR Thread<br/>Read Frames<br/>@ 4 FPS"]
        GAS["Gas Thread<br/>Read ADC<br/>Values"]
    end
    
    subgraph Queue["Window Queue<br/>Thread-Safe Buffer"]
        QUEUE["CaptureWindow<br/>- irFrames: 32x24x20<br/>- gas: 20x3<br/>- timestamp"]
    end
    
    subgraph Inference["Inference Engine"]
        MODEL["TFLite Model<br/>model_float32.tflite<br/>4-class Classifier"]
        PREPROCESS["Preprocessing<br/>Normalization<br/>Scaling"]
        POSTPROCESS["Postprocessing<br/>Softmax<br/>Confidence Score"]
    end
    
    subgraph Output["Output & Analysis"]
        CONSOLE["Console Display<br/>┌─────────────┐<br/>│ Prediction  │<br/>└─────────────┘"]
        LOG["Logging<br/>File Output<br/>Metrics"]
    end
    
    MLX -->|32x24 frames| IR
    MQ2 -->|Analog| MCP
    MQ7 -->|Analog| MCP
    MQ135 -->|Analog| MCP
    MCP -->|Digital| GAS
    
    IR -->|window.irFrames<br/>20 frames| QUEUE
    GAS -->|window.gas<br/>20 readings| QUEUE
    
    QUEUE -->|Pop Window| PREPROCESS
    PREPROCESS -->|Normalized Data| MODEL
    MODEL -->|Raw Logits| POSTPROCESS
    POSTPROCESS -->|InferenceResult<br/>label, confidence,<br/>probabilities| CONSOLE
    POSTPROCESS -->|Metrics| LOG
    
    style Sensors fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style Capture fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Queue fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Inference fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style Output fill:#ffe0b2,stroke:#e65100,stroke-width:2px
```

### Building the Application

```bash
cd gas-detector/app
make clean
make
```

### Running the Application

```bash
cd gas-detector/app
./gas_detector
```

Output example:
```
┌─────────────────────────────┐
│ Prediction: normal (95.3%)  │
├─────────────────────────────┤
│ normal       95.3%          |
│ aerosol      3.2%           |
│ flame        1.2%           |
│ breath       0.3%           |
└─────────────────────────────┘
```

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
