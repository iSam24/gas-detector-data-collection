#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <csignal>
#include <csignal>
#include <atomic>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "MLX90640.h"
#include "MQSensor.h"
#include "MCP3008.h"
#include "inferenceEngine.h"
#include "windowQueue.h"

// Config
static constexpr int   SAMPLE_DURATION_SEC = 5;
static constexpr int   EXPECTED_FPS        = 4;
static constexpr int   TARGET_SAMPLES      = SAMPLE_DURATION_SEC * EXPECTED_FPS;  // 20
static constexpr int   TARGET_SAMPLES_IR   = TARGET_SAMPLES + 1;                  // 21 — first dropped
static constexpr const char* MODEL_PATH    =   "/home/isam/dev/MLX90640/python-inference/model_float32v2.tflite";
static constexpr int    NUM_QUEUE_WINDOWS  = 2;
static bool             DEBUG              = false;

static std::atomic<int> capture_count{0};
static std::atomic<bool> running{true};

void signalHandler(int sig) {
    std::cout << "Interrupt handle " << sig << "\n";
    running = false;
}

// Write IR frames (20, 32, 24) as (20, 768) flat CSV — matches debug_ir_frames.csv
void writeIrCsv(const std::string& path,
                const std::vector<std::vector<std::vector<float>>>& frames)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    f << std::fixed;
    f.precision(2);

    for (const auto& frame : frames) {           // 20 frames
        bool firstVal = true;
        for (const auto& row : frame) {          // 32 rows
            for (float v : row) {                // 24 cols
                if (!firstVal) f << ",";
                f << v;
                firstVal = false;
            }
        }
        f << "\n";
    }

    std::cout << "Saved IR CSV  → " << path
              << "  (" << frames.size() << " x 768)\n";
}

// Write gas data (20, 3) CSV — matches debug_gas.csv
void writeGasCsv(const std::string& path,
                 const std::vector<std::vector<float>>& gas)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    f << std::fixed;
    f.precision(2);

    for (const auto& sample : gas) {            // 20 samples
        for (size_t i = 0; i < sample.size(); ++i) {
            if (i > 0) f << ",";
            f << sample[i];
        }
        f << "\n";
    }

    std::cout << "Saved gas CSV → " << path
              << "  (" << gas.size() << " x " << (gas.empty() ? 0 : gas[0].size()) << ")\n";
}

void captureThread(MLX90640& ir, MQSensor& gas, WindowQueue& queue) {

    if (DEBUG) {
        std::filesystem::create_directories("debug_csv");
    }

    while (running)
    {
        capture_count++;
        CaptureWindow window;

        std::exception_ptr irError, gasError;

        // Launch IR and gas capture in parallel
        std::thread irThread([&]() {
            try {
                window.irFrames = ir.getIrData(TARGET_SAMPLES_IR);
            } catch (...) {
                irError = std::current_exception();
            }
        });

        std::thread gasThread([&]() {
            try {
                window.gas = gas.getGasData(SAMPLE_DURATION_SEC, EXPECTED_FPS);
            } catch (...) {
                gasError = std::current_exception();
            }
        });

        irThread.join();
        gasThread.join();

        if (irError || gasError) {
            std::cerr << "[CaptureThread] Sensor error, skipping window\n";
            continue;
        }

        if ((int)window.irFrames.size() != TARGET_SAMPLES || 
            (int)window.gas.size() != TARGET_SAMPLES) {
            std::cerr << "[CaptureThread] Ir or Gas sample size incorrect,\n";
            continue; 
        }

        if (DEBUG) {
            try {
                std::string irCsvPath = "debug_csv/debug_ir_cpp" + std::to_string(capture_count) + ".csv";
                std::string gasCsvPath = "debug_csv/debug_gas_cpp" + std::to_string(capture_count) + ".csv";
                writeIrCsv(irCsvPath,  window.irFrames);
                writeGasCsv(gasCsvPath, window.gas);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to write CSV: " << e.what() << "\n";
                continue;
            } 
        }

        queue.push(std::move(window));
    }
    queue.stop();   // !running
}

void inferenceThread(InferenceEngine& engine, WindowQueue& queue) {
    while (true) 
    {
        // pop gas data
        auto window = queue.pop();

        if (!window) break; // queue stopped and empty

        InferenceResult result = engine.run(window->irFrames, window->gas);

        std::cout << "\n┌─────────────────────────────┐\n";
        std::cout << "│ Prediction: "
                  << result.label << " (" << result.confidence * 100.0f << "%)\n";
        std::cout << "├─────────────────────────────┤\n";
        const char* labels[] = {"aerosol", "flame", "normal"};
        for (int i = 0; i < 3; ++i) {
            std::cout << "│ " << labels[i] << "\t"
                      << result.probabilities[i] * 100.0f << "%  " << "\n";
        }
        std::cout << "└─────────────────────────────┘\n";
    }
}

int main()
{
    std::signal(SIGINT, signalHandler);  // Ctrl + C handler

    MCP3008   adc("/dev/spidev0.0");
    MQSensor  gasSensor(adc);
    MLX90640  irSensor(EXPECTED_FPS);

    InferenceEngine engine(MODEL_PATH); 
    WindowQueue queue(NUM_QUEUE_WINDOWS);

    // start thread and pass args by ref
    std::thread capture(captureThread, std::ref(irSensor), std::ref(gasSensor), std::ref(queue));
    std::thread inference(inferenceThread, std::ref(engine), std::ref(queue));

    capture.join();
    inference.join();

    return 0;
}
